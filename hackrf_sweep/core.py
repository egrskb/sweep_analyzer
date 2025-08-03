"""High level interface for performing HackRF sweep and processing data.

This module provides :func:`start_sweep` which initialises the HackRF
hardware, sets up a sweep and processes each sweep with a Python
callback.  The implementation is intentionally minimal and only aims to
illustrate how the C API may be wrapped using :mod:`cffi`.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import _lib  # type: ignore

ffi = _lib.ffi
lib = _lib.lib

# Default scanning parameters -------------------------------------------------
DEFAULT_SAMPLE_RATE = int(20e6)
DEFAULT_BANDWIDTH = int(15e6)
# Number of frequency steps expected in a single sweep.  The C example uses
# 16 blocks per transfer so we mimic that here.
STEP_COUNT = 16
# Size of FFT performed for each block.
FFT_SIZE = 256

# Window used before performing the FFT.
_WINDOW = np.hanning(FFT_SIZE).astype(np.float32)
# Buffer holding power values for a single sweep.  Each row represents one
# frequency step.
_sweep_buffer = np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32)
_current_step = 0
_callback: Callable[[np.ndarray], None]


@ffi.callback("int(hackrf_transfer*)")
def _rx_callback(transfer) -> int:
    """C callback passed to ``hackrf_start_rx_sweep``.

    The callback receives IQ samples from HackRF.  Each set of samples is
    windowed and transformed using NumPy's FFT implementation.  The power
    spectrum for each step is stored in ``_sweep_buffer`` until all steps are
    collected, at which point the user provided Python callback is invoked
    with a copy of the array.
    """
    global _current_step

    buf = ffi.buffer(transfer.buffer, transfer.valid_length)
    # Interpret the raw buffer as signed I/Q pairs.
    iq = np.frombuffer(buf, dtype=np.int8).astype(np.float32)
    if iq.size % 2:
        iq = iq[:-1]
    iq = iq.reshape(-1, 2)
    complex_samples = iq[:, 0] + 1j * iq[:, 1]
    # Apply window and FFT
    windowed = complex_samples[:FFT_SIZE] * _WINDOW
    spectrum = np.fft.fft(windowed)
    power = 20 * np.log10(np.abs(spectrum) + 1e-12)

    if _current_step < STEP_COUNT:
        _sweep_buffer[_current_step] = power
        _current_step += 1

    if _current_step >= STEP_COUNT:
        # One full sweep collected; call the Python handler with a copy of data
        _callback(_sweep_buffer.copy())
        _current_step = 0

    return 0


def start_sweep(callback: Callable[[np.ndarray], None], *,
                 sample_rate: float = DEFAULT_SAMPLE_RATE,
                 bandwidth: float = DEFAULT_BANDWIDTH) -> None:
    """Start sweeping with HackRF and call ``callback`` for each sweep.

    Parameters
    ----------
    callback:
        Callable invoked with ``numpy.ndarray`` of shape ``(STEP_COUNT,
        FFT_SIZE)`` containing power values in dB for an entire sweep.
    sample_rate:
        Sample rate passed to ``hackrf_set_sample_rate_manual``.
    bandwidth:
        Baseband filter bandwidth.
    """
    global _callback

    _callback = callback

    if lib.hackrf_init() != 0:
        raise RuntimeError("hackrf_init failed")

    dev_pp = ffi.new("hackrf_device **")
    if lib.hackrf_open_by_serial(ffi.NULL, dev_pp) != 0:
        lib.hackrf_exit()
        raise RuntimeError("hackrf_open_by_serial failed")
    dev = dev_pp[0]

    try:
        lib.hackrf_set_sample_rate_manual(dev, int(sample_rate), 1)
        lib.hackrf_set_baseband_filter_bandwidth(dev, int(bandwidth))
        lib.hackrf_set_vga_gain(dev, 20)
        lib.hackrf_set_lna_gain(dev, 16)

        # Minimal sweep initialisation: one range from 0 to 6000 MHz.  The
        # actual numbers are placeholders but allow the call to succeed.
        freqs = ffi.new("uint16_t[2]", [0, 6000])
        lib.hackrf_init_sweep(dev, freqs, 1, 16384, 1000000, 7500000, 0)

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
