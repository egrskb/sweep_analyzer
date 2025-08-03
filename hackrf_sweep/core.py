"""High level interface for performing HackRF sweep and processing data.

This module provides :func:`start_sweep` which initialises the HackRF
hardware, sets up a sweep and processes each sweep with a Python
callback.  The implementation is intentionally minimal and only aims to
illustrate how the C API may be wrapped using :mod:`cffi`.
"""

from __future__ import annotations

import json
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

# Double buffers holding power values for sweeps.  We flip between them each
# time a sweep completes to avoid copying data for the Python callback.
_buffers = [np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32),
            np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32)]
_buffer_ptrs = [ffi.cast("float *", _buffers[0].ctypes.data),
                ffi.cast("float *", _buffers[1].ctypes.data)]
_active_buf = 0

_callback: Callable[[np.ndarray], None]


def load_config(path: str = "config.json") -> dict:
    """Load sweep parameters from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@ffi.callback("int(hackrf_transfer*)")
def _rx_callback(transfer) -> int:
    """C callback passed to ``hackrf_start_rx_sweep``.

    The heavy FFT processing is implemented in C for efficiency.  This
    callback merely forwards the incoming buffer to the native routine
    and invokes the Python handler when a full sweep has been collected.
    """
    global _active_buf
    finished = lib.hs_process(transfer, _buffer_ptrs[_active_buf])
    if finished:
        ready = _active_buf
        _active_buf ^= 1
        _callback(_buffers[ready])
    return 0


def start_sweep(
    callback: Callable[[np.ndarray], None], *,
    config_path: str = "config.json",
    serial: Optional[str] = None,
) -> None:
    """Start sweeping with HackRF and call ``callback`` for each sweep."""

    global _callback, STEP_COUNT, _buffers, _buffer_ptrs, _freq_start_mhz, _step_mhz, _active_buf

    cfg = load_config(config_path)
    sample_rate = int(cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
    bandwidth = int(cfg.get("bandwidth", DEFAULT_BANDWIDTH))
    freq_start_mhz = float(cfg.get("freq_start_mhz", 50.0))
    freq_stop_mhz = float(cfg.get("freq_stop_mhz", 6000.0))
    step_mhz = float(cfg.get("step_mhz", 5.0))
    vga_gain = int(cfg.get("vga_gain", 20))
    lna_gain = int(cfg.get("lna_gain", 16))
    fft_threads = int(cfg.get("fft_threads", 1))

    _callback = callback
    _freq_start_mhz = freq_start_mhz
    _step_mhz = step_mhz
    STEP_COUNT = int((freq_stop_mhz - freq_start_mhz) / step_mhz)
    _buffers = [np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32),
                np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32)]
    _buffer_ptrs = [ffi.cast("float *", _buffers[0].ctypes.data),
                    ffi.cast("float *", _buffers[1].ctypes.data)]
    _active_buf = 0

    lib.hs_prepare(FFT_SIZE, STEP_COUNT, fft_threads)

    if lib.hackrf_init() != 0:
        raise RuntimeError("hackrf_init failed")

    dev_pp = ffi.new("hackrf_device **")
    ser = serial.encode() if serial is not None else ffi.NULL
    if lib.hackrf_open_by_serial(ser, dev_pp) != 0:
        lib.hackrf_exit()
        raise RuntimeError("hackrf_open_by_serial failed")
    dev = dev_pp[0]

    try:
        lib.hackrf_set_sample_rate_manual(dev, sample_rate, 1)
        lib.hackrf_set_baseband_filter_bandwidth(dev, bandwidth)
        lib.hackrf_set_vga_gain(dev, vga_gain)
        lib.hackrf_set_lna_gain(dev, lna_gain)

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

