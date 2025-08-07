"""CFFI build script for the minimal HackRF sweep wrapper.

Running this script will compile the extension module
``hackrf_sweep._lib`` which exposes a subset of libhackrf required for the
example sweep implementation.
"""

from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """
    typedef struct hackrf_device hackrf_device;
    typedef struct {
        hackrf_device* device;
        unsigned char* buffer;
        int buffer_length;
        int valid_length;
        void* rx_ctx;
        void* tx_ctx;
        ...;
    } hackrf_transfer;

    int hackrf_init(void);
    int hackrf_exit(void);
    int hackrf_open_by_serial(const char* serial, hackrf_device** device);
    int hackrf_close(hackrf_device* device);

    int hackrf_set_sample_rate_manual(hackrf_device* device, uint32_t rate, uint32_t divider);
    int hackrf_set_baseband_filter_bandwidth(hackrf_device* device, uint32_t bandwidth);
    int hackrf_set_vga_gain(hackrf_device* device, uint32_t gain);
    int hackrf_set_lna_gain(hackrf_device* device, uint32_t gain);
    int hackrf_init_sweep(hackrf_device* device, uint16_t* freqs, int num_ranges,
                          uint32_t num_bytes, uint32_t step, uint32_t offset, uint8_t style);
    int hackrf_start_rx_sweep(hackrf_device* device,
                              int (*callback)(hackrf_transfer*), void* ctx);
    int hackrf_is_streaming(hackrf_device* device);

    int hackrf_set_freq(hackrf_device* device, uint64_t freq_hz);
    int hackrf_start_rx(hackrf_device* device,
                        int (*callback)(hackrf_transfer*), void* ctx);
    int hackrf_stop_rx(hackrf_device* device);

    enum hackrf_usb_board_id {
        USB_BOARD_ID_JAWBREAKER = 0x604B,
        USB_BOARD_ID_HACKRF_ONE = 0x6089,
        USB_BOARD_ID_RAD1O = 0xCC15,
        USB_BOARD_ID_INVALID = 0xFFFF
    };
    typedef struct {
        char** serial_numbers;
        enum hackrf_usb_board_id* usb_board_ids;
        int* usb_device_index;
        int devicecount;
        void** usb_devices;
        int usb_devicecount;
    } hackrf_device_list_t;
    hackrf_device_list_t* hackrf_device_list(void);
    void hackrf_device_list_free(hackrf_device_list_t* list);

    void hs_prepare(int fft_size, int step_count, int threads);
    int hs_process(hackrf_transfer* transfer, float* sweep_buffer);
    float hs_rssi(hackrf_transfer* transfer);
    void hs_cleanup(void);
    """
)

ffibuilder.set_source(
    "hackrf_sweep._lib",
    """
    #include <libhackrf/hackrf.h>
    void hs_prepare(int, int, int);
    int hs_process(hackrf_transfer*, float*);
    float hs_rssi(hackrf_transfer*);
    void hs_cleanup(void);
    """,
    sources=["sweep_callback.c"],
    libraries=["hackrf", "usb-1.0", "fftw3f", "fftw3f_threads", "pthread"],
    extra_compile_args=["-O3"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
