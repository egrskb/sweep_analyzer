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

# Additional HackRF devices opened as slaves for RSSI measurements.
_slaves: list[ffi.CData] = []

# Temporary storage used by the RSSI callback.
_rssi_result: float = 0.0
_rssi_done: bool = False

# Empirical offset that roughly converts the FFT magnitudes into the dBm
# range.  The exact value depends on hardware calibration and is only intended
# to provide a stable reference for distance estimation.
RSSI_OFFSET_DBM = -70.0


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


@ffi.callback("int(hackrf_transfer*)")
def _rssi_callback(transfer) -> int:
    """Collect a single buffer of IQ samples and compute its power."""
    global _rssi_result, _rssi_done
    buf = ffi.buffer(transfer.buffer, transfer.valid_length)
    # Convert interleaved int8 IQ samples to float.
    iq = (
        np.frombuffer(buf, dtype=np.int8)
        .astype(np.float32)
        .reshape(-1, 2)
    )
    # Remove DC offset and normalise to the [-1, 1] range.
    iq -= iq.mean(axis=0)
    iq /= 128.0
    power = np.mean(iq ** 2)
    # Convert to dBm, protecting against log(0).
    _rssi_result = 10 * np.log10(power + 1e-12) + RSSI_OFFSET_DBM
    _rssi_done = True
    return 0


def measure_rssi(freq_hz: float) -> list[float]:
    """Measure RSSI at ``freq_hz`` using all slave devices."""
    results: list[float] = []
    for dev in _slaves:
        lib.hackrf_set_freq(dev, int(freq_hz))
        global _rssi_done
        _rssi_done = False
        lib.hackrf_start_rx(dev, _rssi_callback, ffi.NULL)
        while not _rssi_done:
            time.sleep(0.01)
        lib.hackrf_stop_rx(dev)
        results.append(_rssi_result)
    return results


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

    # Enumerate connected devices and open master/slave units.
    dev_list = lib.hackrf_device_list()
    serials = [
        ffi.string(dev_list.serial_numbers[i]).decode()
        for i in range(dev_list.devicecount)
    ]
    lib.hackrf_device_list_free(dev_list)

    if not serials:
        lib.hackrf_exit()
        raise RuntimeError("HackRF device not found")

    master_serial = serial or serials[0]
    slave_serials = [s for s in serials if s != master_serial][:2]

    dev_pp = ffi.new("hackrf_device **")
    if lib.hackrf_open_by_serial(master_serial.encode(), dev_pp) != 0:
        lib.hackrf_exit()
        raise RuntimeError("hackrf_open_by_serial failed")
    dev = dev_pp[0]

    # Open slave devices.
    _slaves.clear()
    for s in slave_serials:
        pp = ffi.new("hackrf_device **")
        if lib.hackrf_open_by_serial(s.encode(), pp) == 0:
            lib.hackrf_set_sample_rate_manual(pp[0], sample_rate, 1)
            lib.hackrf_set_baseband_filter_bandwidth(pp[0], bandwidth)
            lib.hackrf_set_vga_gain(pp[0], vga_gain)
            lib.hackrf_set_lna_gain(pp[0], lna_gain)
            _slaves.append(pp[0])

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
        for s in _slaves:
            lib.hackrf_close(s)
        lib.hackrf_close(dev)
        lib.hackrf_exit()
        lib.hs_cleanup()

