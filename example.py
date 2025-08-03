"""Simple example that prints peak frequency of each sweep row."""

from __future__ import annotations

import numpy as np

from typing import List, Optional

from hackrf_sweep import start_sweep, load_config
from hackrf_sweep.core import FFT_SIZE, ffi, lib

CONFIG = load_config()
START_MHZ = CONFIG["freq_start_mhz"]
STEP_MHZ = CONFIG["step_mhz"]
BIN_WIDTH_MHZ = CONFIG["sample_rate"] / FFT_SIZE / 1e6


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
        serial=master,
    )
