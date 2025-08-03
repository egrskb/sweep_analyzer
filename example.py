"""Example demonstrating baseline sweep and RSSI measurements on peaks.

Constants at the top control sweep range, step size and detection threshold.
Replace entries in ``SLAVE_SERIALS`` with serial numbers of additional HackRF
units used for RSSI measurements.
"""

from __future__ import annotations

import numpy as np

from hackrf_sweep import measure_rssi, start_sweep
from hackrf_sweep.core import DEFAULT_SAMPLE_RATE, FFT_SIZE

# --- Configurable constants -------------------------------------------------
FREQ_START_MHZ = 50      # sweep start frequency
FREQ_STOP_MHZ = 6000     # sweep stop frequency
STEP_MHZ = 5             # frequency step size
THRESHOLD_DB = 10        # detection threshold over baseline
SLAVE_SERIALS = [None, None]  # serial numbers of two slave HackRF units

BIN_WIDTH_MHZ = DEFAULT_SAMPLE_RATE / FFT_SIZE / 1e6

baseline: np.ndarray | None = None


def process_sweep(sweep: np.ndarray) -> None:
    """Handle each completed sweep.

    The first sweep is stored as the baseline.  Subsequent sweeps are compared
    against it and any bins exceeding ``THRESHOLD_DB`` above baseline trigger an
    RSSI measurement on the slave devices.
    """
    global baseline

    if baseline is None:
        baseline = sweep.copy()
        print("Базовый уровень сохранён")
        return

    diff = sweep - baseline
    for step_idx, row in enumerate(diff):
        idx = int(np.argmax(row))
        delta = row[idx]
        if delta > THRESHOLD_DB:
            freq_mhz = FREQ_START_MHZ + step_idx * STEP_MHZ + idx * BIN_WIDTH_MHZ
            amp_db = sweep[step_idx, idx]
            print(
                f"Сигнал {freq_mhz:.2f} МГц: {amp_db:.1f} дБ (∆{delta:.1f} дБ)"
            )
            for serial in SLAVE_SERIALS:
                rssi = measure_rssi(serial, freq_mhz)
                label = serial or "slave"
                print(f"  RSSI {label}: {rssi:.1f} дБ")


if __name__ == "__main__":
    start_sweep(
        process_sweep,
        freq_start_mhz=FREQ_START_MHZ,
        freq_stop_mhz=FREQ_STOP_MHZ,
        step_mhz=STEP_MHZ,
    )
