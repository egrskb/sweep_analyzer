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
    """
)

ffibuilder.set_source(
    "hackrf_sweep._lib",
    "#include <libhackrf/hackrf.h>",
    libraries=["hackrf", "usb-1.0", "fftw3f", "pthread"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
