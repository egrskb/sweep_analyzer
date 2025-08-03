"""High level interface for performing HackRF sweep and processing data.

This module provides :func:`start_sweep` which initialises the HackRF
hardware, sets up a sweep and processes each sweep with a Python
callback.  The implementation is intentionally minimal and only aims to
illustrate how the C API may be wrapped using :mod:`cffi`.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from . import _lib  # type: ignore

ffi = _lib.ffi
lib = _lib.lib

# Default scanning parameters -------------------------------------------------
DEFAULT_SAMPLE_RATE = int(20e6)
DEFAULT_BANDWIDTH = int(15e6)

# Size of FFT performed for each block.
FFT_SIZE = 256

# Mutable sweep configuration.  ``start_sweep`` updates these globals so that
# the callback knows how many steps to expect and what frequencies they map to.
STEP_COUNT = 16
_freq_start_mhz = 0.0
_step_mhz = 1.0

# Buffer holding power values for a single sweep.  Each row represents one
# frequency step.
_sweep_buffer = np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32)
_sweep_buffer_c = ffi.cast("float *", _sweep_buffer.ctypes.data)
_callback: Callable[[np.ndarray], None]
@ffi.callback("int(hackrf_transfer*)")
def _rx_callback(transfer) -> int:
    """C callback passed to ``hackrf_start_rx_sweep``.

    The heavy FFT processing is implemented in C for efficiency.  This
    callback merely forwards the incoming buffer to the native routine
    and invokes the Python handler when a full sweep has been collected.
    """
    finished = lib.hs_process(transfer, _sweep_buffer_c)
    if finished:
        _callback(_sweep_buffer.copy())
    return 0


def start_sweep(
    callback: Callable[[np.ndarray], None], *,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    bandwidth: float = DEFAULT_BANDWIDTH,
    freq_start_mhz: float = 50.0,
    freq_stop_mhz: float = 6000.0,
    step_mhz: float = 5.0,
    serial: Optional[str] = None,
) -> None:
    """Start sweeping with HackRF and call ``callback`` for each sweep.

    Parameters
    ----------
    callback:
        Callable invoked with ``numpy.ndarray`` containing power values in dB
        for an entire sweep.
    sample_rate:
        Sample rate passed to ``hackrf_set_sample_rate_manual``.
    bandwidth:
        Baseband filter bandwidth.
    freq_start_mhz, freq_stop_mhz:
        Frequency range to sweep, expressed in MHz.
    step_mhz:
        Step size between consecutive centre frequencies in MHz.
    serial:
        Optional serial number of the HackRF to open. ``None`` selects the
        first available device.
    """
    global _callback, STEP_COUNT, _sweep_buffer, _sweep_buffer_c, _freq_start_mhz, _step_mhz

    _callback = callback
    _freq_start_mhz = freq_start_mhz
    _step_mhz = step_mhz
    STEP_COUNT = int((freq_stop_mhz - freq_start_mhz) / step_mhz)
    _sweep_buffer = np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32)
    _sweep_buffer_c = ffi.cast("float *", _sweep_buffer.ctypes.data)

    lib.hs_prepare(FFT_SIZE, STEP_COUNT)

    if lib.hackrf_init() != 0:
        raise RuntimeError("hackrf_init failed")

    dev_pp = ffi.new("hackrf_device **")
    ser = serial.encode() if serial is not None else ffi.NULL
    if lib.hackrf_open_by_serial(ser, dev_pp) != 0:
        lib.hackrf_exit()
        raise RuntimeError("hackrf_open_by_serial failed")
    dev = dev_pp[0]

    try:
        lib.hackrf_set_sample_rate_manual(dev, int(sample_rate), 1)
        lib.hackrf_set_baseband_filter_bandwidth(dev, int(bandwidth))
        lib.hackrf_set_vga_gain(dev, 20)
        lib.hackrf_set_lna_gain(dev, 16)

        freqs = ffi.new(
            "uint16_t[2]",
            [int(freq_start_mhz), int(freq_stop_mhz)],
        )
        lib.hackrf_init_sweep(
            dev,
            freqs,
            1,
            16384,
            int(step_mhz * 1e6),
            7500000,
            0,
        )

        if lib.hackrf_start_rx_sweep(dev, _rx_callback, ffi.NULL) != 0:
            raise RuntimeError("hackrf_start_rx_sweep failed")

        try:
            while lib.hackrf_is_streaming(dev):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    finally:
        lib.hackrf_close(dev)
        lib.hackrf_exit()
        lib.hs_cleanup()

