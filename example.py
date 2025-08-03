"""Simple example that prints peak frequency of each sweep row."""

from __future__ import annotations

import numpy as np

from typing import List, Optional

from hackrf_sweep import start_sweep
from hackrf_sweep.core import DEFAULT_SAMPLE_RATE, FFT_SIZE, ffi, lib

START_MHZ = 50
STOP_MHZ = 6000
STEP_MHZ = 5
BIN_WIDTH_MHZ = DEFAULT_SAMPLE_RATE / FFT_SIZE / 1e6


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


def process_peaks(sweep: np.ndarray) -> None:
    for step_idx, row in enumerate(sweep):
        peak = int(np.argmax(row))
        freq_mhz = START_MHZ + step_idx * STEP_MHZ + peak * BIN_WIDTH_MHZ
        print(f"Пик на частоте {freq_mhz:.2f} МГц: {row[peak]:.1f} дБ")


if __name__ == "__main__":
    serials = list_serials()
    if not serials:
        raise RuntimeError("HackRF device not found")

    master = serials[0]
    slave = serials[1] if len(serials) > 1 else None

    print(f"SDR master - {master}")
    if slave:
        print(f"SDR slave - {slave}")
    else:
        print("SDR slave - нет")

    start_sweep(
        process_peaks,
        freq_start_mhz=START_MHZ,
        freq_stop_mhz=STOP_MHZ,
        step_mhz=STEP_MHZ,
        serial=master,
    )
