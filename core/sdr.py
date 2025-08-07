"""Абстракция SDR-устройства для HackRF One.

Модуль предоставляет простой класс :class:`SDRDevice`, который использует
библиотеку :mod:`hackrf_sweep` для выполнения спектральных свипов. Для
разработки и тестов предусмотрен класс :class:`MockSDR`, генерирующий
случайные спектры.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional
from threading import Event

try:  # pragma: no cover - module may be unavailable during tests
    from hackrf_sweep import _lib, start_sweep  # type: ignore
except Exception:  # pragma: no cover
    _lib = None  # type: ignore
    start_sweep = None  # type: ignore

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

    def sweep(
        self,
        callback: Callable[[np.ndarray], None],
        *,
        config_path: str = "config.json",
        stop_event: Optional[Event] = None,
    ) -> None:
        """Запустить непрерывный свип.

        Parameters
        ----------
        callback:
            Пользовательская функция, принимающая одномерный массив мощностей.
        config_path:
            Путь к JSON-файлу с параметрами свипа.
        stop_event:
            Событие, установка которого завершает свип.
        """

        if start_sweep is None:  # pragma: no cover - no extension during tests
            raise RuntimeError("hackrf_sweep не установлен")

        def _handle(data: np.ndarray) -> None:
            callback(data.ravel())

        start_sweep(
            _handle, config_path=config_path, serial=self.serial, stop_event=stop_event
        )


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

    def sweep(
        self,
        callback: Callable[[np.ndarray], None],
        *,
        config_path: str = "config.json",
        stop_event: Optional[Event] = None,
    ) -> None:
        stop_event = stop_event or Event()
        bins = 512
        while not stop_event.is_set():
            power = np.abs(np.fft.rfft(np.random.randn(bins))).astype(np.float32)
            callback(power)
            stop_event.wait(1.0)

