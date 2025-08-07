"""Data logging utilities."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterable

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def save_csv(path: Path, freqs: np.ndarray, power: np.ndarray) -> None:
    """Save spectrum to CSV file."""
    data = np.column_stack((freqs, power))
    np.savetxt(path, data, delimiter=",", header="freq,power_db")


def save_hdf5(path: Path, spectra: Iterable[np.ndarray]) -> None:
    """Save spectra sequence to HDF5 file."""
    if h5py is None:  # pragma: no cover
        raise RuntimeError("h5py not installed")
    with h5py.File(path, "w") as h5:
        for i, spec in enumerate(spectra):
            h5.create_dataset(f"spec_{i}", data=spec)
