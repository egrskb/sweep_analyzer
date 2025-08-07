"""SDR device abstraction layer using ``hackrf_sweep``."""
from __future__ import annotations

import numpy as np
from typing import List

try:  # pragma: no cover - module may be unavailable during tests
    from hackrf_sweep import _lib  # type: ignore
except Exception:  # pragma: no cover
    _lib = None  # type: ignore

ffi = _lib.ffi if _lib else None
lib = _lib.lib if _lib else None


class SDRDevice:
    """Represents a single HackRF One device."""

    def __init__(self, serial: str) -> None:
        self.serial = serial

    def open(self) -> None:
        """Open the device connection.

        The :mod:`hackrf_sweep` library handles device setup internally when
        starting a sweep, so this method is a no-op provided for API
        compatibility.
        """

    def close(self) -> None:
        """Close the device connection."""

    def read_samples(self, num_samples: int) -> np.ndarray:  # pragma: no cover - hardware specific
        """Read IQ samples from the device.

        Direct sample access is not implemented in this skeleton; sweeping and
        FFT processing are performed via :mod:`hackrf_sweep`.
        """
        raise NotImplementedError("Direct sample access not implemented")


def enumerate_devices() -> List[SDRDevice]:
    """Return list of available HackRF devices."""

    devices: List[SDRDevice] = []
    if lib is None:  # pragma: no cover - exercised when extension missing
        return devices
    dev_list = lib.hackrf_device_list()
    for i in range(dev_list.devicecount):
        serial = ffi.string(dev_list.serial_numbers[i]).decode()
        devices.append(SDRDevice(serial))
    lib.hackrf_device_list_free(dev_list)
    return devices


class MockSDR(SDRDevice):
    """Mock SDR device used for tests and development."""

    def __init__(self, serial: str = "MOCK") -> None:
        super().__init__(serial)

    def read_samples(self, num_samples: int) -> np.ndarray:  # pragma: no cover - deterministic
        t = np.arange(num_samples)
        signal = np.exp(2j * np.pi * 0.1 * t)
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.01
        return (signal + noise).astype(np.complex64)

