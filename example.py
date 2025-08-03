"""Simple example demonstrating use of the ``hackrf_sweep`` package.

The script defines :func:`process_peaks` which receives the power array for
an entire sweep and prints the frequency of the strongest bin in each step.
Run ``python build_hackrf_sweep.py`` beforehand to compile the CFFI extension.
"""

import numpy as np

from hackrf_sweep import start_sweep
from hackrf_sweep.core import FFT_SIZE, STEP_COUNT, DEFAULT_SAMPLE_RATE

BIN_WIDTH_MHZ = DEFAULT_SAMPLE_RATE / FFT_SIZE / 1e6


def process_peaks(sweep: np.ndarray) -> None:
    """Print peak information for each step of the sweep."""
    for row in sweep:
        idx = int(np.argmax(row))
        freq_mhz = idx * BIN_WIDTH_MHZ
        amp_db = row[idx]
        print(f"Пик на частоте {freq_mhz:.2f} МГц: {amp_db:.1f} дБ")


if __name__ == "__main__":
    start_sweep(process_peaks)
