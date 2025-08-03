"""Example that demonstrates baseline detection and multi-SDR RSSI."""

from __future__ import annotations

import numpy as np

from typing import List, Optional

from hackrf_sweep import start_sweep, load_config, measure_rssi
from hackrf_sweep.core import FFT_SIZE, ffi, lib

# Load sweep parameters so we can map bins to frequencies.
CONFIG = load_config()
START_MHZ = CONFIG["freq_start_mhz"]
STEP_MHZ = CONFIG["step_mhz"]
BIN_WIDTH_MHZ = CONFIG["sample_rate"] / FFT_SIZE / 1e6

# Detection settings.
THRESHOLD_DB = 10.0
BASELINE: np.ndarray | None = None


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
    """Handle each sweep from the master device."""
    global BASELINE
    if BASELINE is None:
        # First sweep becomes the baseline noise level.
        BASELINE = sweep.copy()
        return
    diff = sweep - BASELINE
    for step_idx, row in enumerate(diff):
        peaks = np.where(row > THRESHOLD_DB)[0]
        for peak in peaks:
            freq_mhz = START_MHZ + step_idx * STEP_MHZ + peak * BIN_WIDTH_MHZ
            level = sweep[step_idx, peak]
            msg = f"Пик на частоте {freq_mhz:.2f} МГц: {level:.1f} дБ"
            rssi_vals = measure_rssi(freq_mhz * 1e6)
            if rssi_vals:
                msg += " | RSSI: " + ", ".join(f"{v:.1f} дБ" for v in rssi_vals)
            print(msg)


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
