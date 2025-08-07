#include <libhackrf/hackrf.h>
#include <stdint.h>
#include <math.h>
#include <fftw3.h>

/* Global state used by the lightweight C helper.  These variables hold the
 * FFT configuration, window coefficients and work buffers.  They are
 * initialised by ``hs_prepare`` and freed by ``hs_cleanup``. */
static int g_fft_size = 0;
static int g_step_count = 0;
static int g_current_step = 0;
static float *g_window = NULL;
static fftwf_complex *g_in = NULL;
static fftwf_complex *g_out = NULL;
static fftwf_plan g_plan = NULL;
/* Буфер для хранения мощности в дБм, нужен при вычислении RSSI слейвов */
static float *g_power = NULL;
static int g_threads = 1;
/* Constant that shifts the log power output into the dBm range.  The value is
 * empirical and merely provides a rough reference level for distance
 * estimation. */
static const float RSSI_OFFSET_DBM = -70.0f;

void hs_prepare(int fft_size, int step_count, int threads) {
    g_fft_size = fft_size;
    g_step_count = step_count;
    g_current_step = 0;
    g_threads = threads > 0 ? threads : 1;
    if (g_window) { fftwf_free(g_window); g_window = NULL; }
    if (g_in) { fftwf_free(g_in); g_in = NULL; }
    if (g_out) { fftwf_free(g_out); g_out = NULL; }
    if (g_power) { fftwf_free(g_power); g_power = NULL; }
    if (g_plan) { fftwf_destroy_plan(g_plan); g_plan = NULL; }

    fftwf_init_threads();
    fftwf_plan_with_nthreads(g_threads);

    g_window = (float*)fftwf_malloc(sizeof(float)*fft_size);
    g_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fft_size);
    g_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fft_size);
    g_power = (float*)fftwf_malloc(sizeof(float)*fft_size);
    for (int i=0;i<fft_size;i++) {
        g_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
    }
    g_plan = fftwf_plan_dft_1d(fft_size, g_in, g_out, FFTW_FORWARD, FFTW_MEASURE);
}

void hs_cleanup(void) {
    if (g_window) { fftwf_free(g_window); g_window = NULL; }
    if (g_in) { fftwf_free(g_in); g_in = NULL; }
    if (g_out) { fftwf_free(g_out); g_out = NULL; }
    if (g_power) { fftwf_free(g_power); g_power = NULL; }
    if (g_plan) { fftwf_destroy_plan(g_plan); g_plan = NULL; }
    fftwf_cleanup_threads();
}

int hs_process(hackrf_transfer* transfer, float* sweep_buffer) {
    if (!g_window) return 0;
    int8_t* buf = (int8_t*)transfer->buffer;
    float mean_re = 0.0f;
    float mean_im = 0.0f;
    /* Single pass: update running mean while applying the Hann window and
     * scaling samples to floats.  This avoids a second read of the IQ buffer. */
    for (int i = 0; i < g_fft_size; i++) {
        float re = buf[2*i];
        float im = buf[2*i+1];
        mean_re += (re - mean_re) / (i + 1);
        mean_im += (im - mean_im) / (i + 1);
        re = (re - mean_re) / 128.0f;
        im = (im - mean_im) / 128.0f;
        g_in[i][0] = re * g_window[i];
        g_in[i][1] = im * g_window[i];
    }
    fftwf_execute(g_plan);
    float* dest = sweep_buffer + g_current_step * g_fft_size;
    for (int i=0;i<g_fft_size;i++) {
        float re = g_out[i][0];
        float im = g_out[i][1];
        float mag = sqrtf(re*re + im*im);
        /* Convert magnitude to dBm.  The small offset protects against log(0)
         * and RSSI_OFFSET_DBM roughly aligns readings with a practical dBm
         * scale. */
        dest[i] = 20.0f * log10f(mag + 1e-12f) + RSSI_OFFSET_DBM;
    }
    g_current_step++;
    if (g_current_step >= g_step_count) {
        g_current_step = 0;
        return 1;
    }
    return 0;
}

/* Простая функция для вычисления среднего по трём самым сильным бинам.
 * Используется слейвами, чтобы быстро оценивать уровень сигнала на нужной
 * частоте. */
float hs_rssi(hackrf_transfer* transfer) {
    if (!g_window) return 0.0f;
    int8_t* buf = (int8_t*)transfer->buffer;
    float mean_re = 0.0f;
    float mean_im = 0.0f;
    for (int i = 0; i < g_fft_size; i++) {
        float re = buf[2*i];
        float im = buf[2*i+1];
        mean_re += (re - mean_re) / (i + 1);
        mean_im += (im - mean_im) / (i + 1);
        re = (re - mean_re) / 128.0f;
        im = (im - mean_im) / 128.0f;
        g_in[i][0] = re * g_window[i];
        g_in[i][1] = im * g_window[i];
    }
    fftwf_execute(g_plan);
    for (int i = 0; i < g_fft_size; i++) {
        float re = g_out[i][0];
        float im = g_out[i][1];
        float mag = sqrtf(re*re + im*im);
        g_power[i] = 20.0f * log10f(mag + 1e-12f) + RSSI_OFFSET_DBM;
    }
    if (g_fft_size < 3) {
        float sum = 0.0f;
        for (int i = 0; i < g_fft_size; i++) sum += g_power[i];
        return sum / g_fft_size;
    }
    float window_sum = g_power[0] + g_power[1] + g_power[2];
    float max_mean = window_sum / 3.0f;
    for (int i = 3; i < g_fft_size; i++) {
        window_sum += g_power[i] - g_power[i-3];
        float mean = window_sum / 3.0f;
        if (mean > max_mean) max_mean = mean;
    }
    return max_mean;
}
