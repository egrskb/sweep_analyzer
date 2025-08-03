#include <libhackrf/hackrf.h>
#include <stdint.h>
#include <math.h>
#include <fftw3.h>

static int g_fft_size = 0;
static int g_step_count = 0;
static int g_current_step = 0;
static float *g_window = NULL;
static fftwf_complex *g_in = NULL;
static fftwf_complex *g_out = NULL;
static fftwf_plan g_plan = NULL;

void hs_prepare(int fft_size, int step_count) {
    g_fft_size = fft_size;
    g_step_count = step_count;
    g_current_step = 0;
    if (g_window) { fftwf_free(g_window); g_window = NULL; }
    if (g_in) { fftwf_free(g_in); g_in = NULL; }
    if (g_out) { fftwf_free(g_out); g_out = NULL; }
    if (g_plan) { fftwf_destroy_plan(g_plan); g_plan = NULL; }

    g_window = (float*)fftwf_malloc(sizeof(float)*fft_size);
    g_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fft_size);
    g_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fft_size);
    for (int i=0;i<fft_size;i++) {
        g_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
    }
    g_plan = fftwf_plan_dft_1d(fft_size, g_in, g_out, FFTW_FORWARD, FFTW_MEASURE);
}

void hs_cleanup(void) {
    if (g_window) { fftwf_free(g_window); g_window = NULL; }
    if (g_in) { fftwf_free(g_in); g_in = NULL; }
    if (g_out) { fftwf_free(g_out); g_out = NULL; }
    if (g_plan) { fftwf_destroy_plan(g_plan); g_plan = NULL; }
}

int hs_process(hackrf_transfer* transfer, float* sweep_buffer) {
    if (!g_window) return 0;
    int8_t* buf = (int8_t*)transfer->buffer;
    for (int i=0;i<g_fft_size;i++) {
        float re = (float)buf[2*i] * g_window[i];
        float im = (float)buf[2*i+1] * g_window[i];
        g_in[i][0] = re;
        g_in[i][1] = im;
    }
    fftwf_execute(g_plan);
    float* dest = sweep_buffer + g_current_step * g_fft_size;
    for (int i=0;i<g_fft_size;i++) {
        float re = g_out[i][0];
        float im = g_out[i][1];
        float mag = sqrtf(re*re + im*im);
        dest[i] = 20.0f * log10f(mag + 1e-12f);
    }
    g_current_step++;
    if (g_current_step >= g_step_count) {
        g_current_step = 0;
        return 1;
    }
    return 0;
}
