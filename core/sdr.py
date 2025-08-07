"""Абстракция SDR-устройства с использованием ``hackrf_sweep``."""
from __future__ import annotations

import numpy as np
from threading import Event
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
        self._dev = ffi.NULL if ffi else None

    def open(self) -> None:
        """Открыть соединение с устройством."""

        if lib is None:  # pragma: no cover - no hardware during tests
            raise RuntimeError("hackrf_sweep не установлен")
        if lib.hackrf_init() != 0:
            raise RuntimeError("hackrf_init failed")
        dev_pp = ffi.new("hackrf_device **")
        if lib.hackrf_open_by_serial(self.serial.encode(), dev_pp) != 0:
            lib.hackrf_exit()
            raise RuntimeError("hackrf_open_by_serial failed")
        self._dev = dev_pp[0]

    def close(self) -> None:
        """Закрыть соединение с устройством."""

        if lib is None or not self._dev:  # pragma: no cover
            return
        lib.hackrf_close(self._dev)
        lib.hackrf_exit()
        self._dev = ffi.NULL

    def read_samples(self, num_samples: int) -> np.ndarray:  # pragma: no cover - hardware specific
        """Считать ``num_samples`` IQ-сэмплов с устройства."""

        if lib is None or not self._dev:
            raise RuntimeError("Устройство не открыто")

        total = num_samples * 2
        buf = ffi.new("uint8_t[]", total)
        written = 0
        ready = Event()

        @ffi.callback("int(hackrf_transfer*)")
        def _cb(transfer):
            nonlocal written
            n = min(transfer.valid_length, total - written)
            ffi.memmove(buf + written, transfer.buffer, n)
            written += n
            if written >= total:
                ready.set()
            return 0

        if lib.hackrf_start_rx(self._dev, _cb, ffi.NULL) != 0:
            raise RuntimeError("hackrf_start_rx failed")
        ready.wait()
        lib.hackrf_stop_rx(self._dev)

        arr = np.frombuffer(ffi.buffer(buf, total), dtype=np.uint8)
        i = arr[0::2].astype(np.float32) - 128
        q = arr[1::2].astype(np.float32) - 128
        return (i + 1j * q).astype(np.complex64) / 128.0


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

