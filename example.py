"""Example scanner demonstrating baseline tracking and RSSI reporting.

The script sweeps the configured spectrum using a master HackRF and measures
RSSI on up to two additional slave devices.  Frequencies that deviate from the
baseline by more than ``THRESHOLD_DB`` are tracked and reported each sweep.
When the signal returns to the baseline level the frequency is removed from the
tracking list.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from hackrf_sweep import start_sweep, load_config, measure_rssi
from hackrf_sweep.core import FFT_SIZE, ffi, lib

# Load sweep parameters so we can map bins to frequencies.
CONFIG = load_config()
START_MHZ = CONFIG["freq_start_mhz"]
STEP_MHZ = CONFIG["step_mhz"]
BIN_WIDTH_MHZ = CONFIG["sample_rate"] / FFT_SIZE / 1e6

# Detection settings.  Signals that differ from the baseline by this many dB
# are tracked and reported.
THRESHOLD_DB = 10.0

# Baseline sweep captured on the first iteration.
BASELINE: np.ndarray | None = None

# Frequencies currently being monitored.  The key is the frequency in MHz and
# the value stores baseline and last RSSI for master and slave devices.
TRACKED: Dict[float, Dict[str, Any]] = {}

# Timestamp of the previous sweep to measure cycle time.
_last_sweep_time = time.time()


def _approx_distance(rssi_dbm: float) -> float:
    """Crude log-distance path loss model assuming -40 dBm at 1 m."""
    ref_rssi = -40.0  # reference RSSI at 1 m
    path_loss = 2.0   # free space
    return 10 ** ((ref_rssi - rssi_dbm) / (10 * path_loss))


def list_serials() -> List[Optional[str]]:
    """Return serial numbers of connected HackRF devices."""
    if lib.hackrf_init() != 0:
        raise RuntimeError("hackrf_init failed")
    try:
        lst = lib.hackrf_device_list()
        serials: List[Optional[str]] = []
        for i in range(lst.devicecount):
            sn = lst.serial_numbers[i]
            serials.append(ffi.string(sn).decode() if sn != ffi.NULL else None)
        lib.hackrf_device_list_free(lst)
        return serials
    finally:
        lib.hackrf_exit()


def process_sweep(sweep: np.ndarray) -> None:
    """Handle each completed sweep from the master device."""
    global BASELINE, _last_sweep_time

    now = time.time()
    cycle = now - _last_sweep_time
    _last_sweep_time = now

    if BASELINE is None:
        # Use the very first sweep as the reference noise floor.
        BASELINE = sweep.copy()
        return

    diff = sweep - BASELINE
    for step_idx in range(diff.shape[0]):
        for bin_idx in range(diff.shape[1]):
            freq_mhz = START_MHZ + step_idx * STEP_MHZ + bin_idx * BIN_WIDTH_MHZ
            base = BASELINE[step_idx, bin_idx]
            current = sweep[step_idx, bin_idx]
            delta = current - base

            tracked = freq_mhz in TRACKED
            if abs(delta) >= THRESHOLD_DB or tracked:
                # Measure RSSI on slave devices for trilateration purposes.
                slave_vals = measure_rssi(freq_mhz * 1e6)
                TRACKED[freq_mhz] = {
                    "baseline": base,
                    "master": current,
                    "slaves": slave_vals,
                }

                if abs(delta) < THRESHOLD_DB:
                    # Signal returned to baseline; stop tracking.
                    del TRACKED[freq_mhz]
                    continue

                # Decide whether the signal rose or fell relative to baseline.
                verb = "вырос на" if delta > 0 else "упал на"
                sign = "+" if delta > 0 else "-"
                distance = _approx_distance(current)

                print(
                    f"{sign} Частота: {freq_mhz/1000:.3f} GHz | RSSI: {verb} {abs(delta):.1f} dB "
                    f"(было {base:.1f} dBm → стало {current:.1f} dBm) | "
                    f"Расстояние: ~{distance:.1f} м | относительно SDR Master"
                )

    print(f"[i] Время свипа: {cycle:.2f} с")


if __name__ == "__main__":
    serials = list_serials()
    if not serials:
        raise RuntimeError("HackRF device not found")

    master = serials[0]
    slaves = serials[1:]

    print(f"SDR master - {master}")
    if slaves:
        for s in slaves:
            print(f"SDR slave - {s}")
    else:
        print("SDR slave - нет")

    start_sweep(process_sweep, serial=master)
