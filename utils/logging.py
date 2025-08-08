"""Data logging utilities."""
from __future__ import annotations

import datetime as _dt
import numpy as np
from pathlib import Path
from typing import Iterable

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def save_csv(path: Path, freqs: np.ndarray, power: np.ndarray) -> None:
    """Save spectrum in HackRF sweep CSV format.

    Формат соответствует выходу `hackrf_sweep`:

    ``date, time, hz_low, hz_high, hz_bin_width, num_samples, dB...``
    """
    now = _dt.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S.%f")
    hz_low = float(freqs[0])
    hz_high = float(freqs[-1])
    bin_width = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    header = f"{date_str}, {time_str}, {hz_low:.0f}, {hz_high:.0f}, {bin_width:.0f}, {len(freqs)}"
    power_str = ", ".join(f"{p:.2f}" for p in power)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        if power_str:
            f.write(", " + power_str)
        f.write("\n")


def save_hdf5(path: Path, spectra: Iterable[np.ndarray]) -> None:
    """Save spectra sequence to HDF5 file."""
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py not installed")
    with h5py.File(path, "w") as h5:
        for i, spec in enumerate(spectra):
            h5.create_dataset(f"spec_{i}", data=spec)
