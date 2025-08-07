"""Абстракция SDR-устройства с использованием ``hackrf_sweep``."""
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
    """Представляет одно устройство HackRF One."""

    def __init__(self, serial: str) -> None:
        self.serial = serial

    def open(self) -> None:
        """Открыть соединение с устройством.

        Библиотека :mod:`hackrf_sweep` выполняет настройку устройства при
        запуске сканирования, поэтому этот метод присутствует лишь для
        совместимости API.
        """

    def close(self) -> None:
        """Закрыть соединение с устройством."""

    def read_samples(self, num_samples: int) -> np.ndarray:  # pragma: no cover - hardware specific
        """Считать IQ-сэмплы с устройства.

        Прямой доступ к сэмплам в данном каркасе не реализован; сканирование и
        обработка FFT выполняются через :mod:`hackrf_sweep`.
        """
        raise NotImplementedError("Прямой доступ к сэмплам не реализован")


def enumerate_devices() -> List[SDRDevice]:
    """Вернуть список доступных устройств HackRF.

    Библиотека ``libhackrf`` требует вызова :func:`hackrf_init` перед любой
    работой с устройством. В противном случае возможна сегментация на некоторых
    системах. Поэтому мы временно инициализируем библиотеку для получения
    списка устройств и затем корректно завершаем работу.
    """

    devices: List[SDRDevice] = []
    if lib is None:  # pragma: no cover - exercised when extension missing
        return devices

    if lib.hackrf_init() != 0:  # pragma: no cover - library initialisation
        return devices
    try:
        dev_list = lib.hackrf_device_list()
        for i in range(dev_list.devicecount):
            serial = ffi.string(dev_list.serial_numbers[i]).decode()
            devices.append(SDRDevice(serial))
        lib.hackrf_device_list_free(dev_list)
    finally:
        lib.hackrf_exit()
    return devices


class MockSDR(SDRDevice):
    """Мок-устройство SDR для тестов и разработки."""

    def __init__(self, serial: str = "MOCK") -> None:
        super().__init__(serial)

    def read_samples(self, num_samples: int) -> np.ndarray:  # pragma: no cover - deterministic
        t = np.arange(num_samples)
        signal = np.exp(2j * np.pi * 0.1 * t)
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.01
        return (signal + noise).astype(np.complex64)

