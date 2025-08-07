"""SDR device abstraction layer."""
from __future__ import annotations

import numpy as np
from typing import List, Optional

try:
    import SoapySDR  # type: ignore
except Exception:  # pragma: no cover
    SoapySDR = None  # type: ignore


class SDRDevice:
    """Represents a single SDR device using SoapySDR."""

    def __init__(self, serial: str) -> None:
        self.serial = serial
        self._device: Optional[object] = None

    def open(self) -> None:
        """Open the device connection."""
        if SoapySDR is None:  # pragma: no cover
            raise RuntimeError("SoapySDR not available")
        args = dict(serial=self.serial)
        self._device = SoapySDR.Device(args)

    def close(self) -> None:
        """Close the device connection."""
        self._device = None

    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from the device.

        Args:
            num_samples: Number of complex samples to retrieve.

        Returns:
            Array of complex64 IQ samples.
        """
        if self._device is None:  # pragma: no cover
            raise RuntimeError("Device not open")
        buf = np.empty(num_samples, dtype=np.complex64)
        # Here we simply read from channel 0; real implementation would
        # configure streams etc.
        self._device.readStream(self._device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32), [buf], num_samples)
        return buf


def enumerate_devices() -> List[SDRDevice]:
    """Return list of available SDR devices."""
    devices: List[SDRDevice] = []
    if SoapySDR is None:
        return devices
    for info in SoapySDR.Device.enumerate():
        serial = info.get("serial", "unknown")
        devices.append(SDRDevice(serial))
    return devices


class MockSDR(SDRDevice):
    """Mock SDR device used for tests and development."""

    def __init__(self, serial: str = "MOCK") -> None:
        super().__init__(serial)
        self._device = True

    def read_samples(self, num_samples: int) -> np.ndarray:  # pragma: no cover - deterministic
        t = np.arange(num_samples)
        signal = np.exp(2j * np.pi * 0.1 * t)
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.01
        return (signal + noise).astype(np.complex64)
