"""Простая обёртка для свипа HackRF и обработки данных.

Функция :func:`start_sweep` настраивает устройство, запускает свип
и вызывает переданный коллбэк для каждого готового свипа. Код минимален и
показывает, как обернуть C API через модуль :mod:`cffi`.
"""

from __future__ import annotations

import json
import os
import time
from threading import Event, Lock
from typing import Callable, Optional

import numpy as np

from . import _lib  # type: ignore

ffi = _lib.ffi
lib = _lib.lib

# --------------------------- параметры по умолчанию ---------------------------
DEFAULT_SAMPLE_RATE = int(20e6)  # частота дискретизации
DEFAULT_BANDWIDTH = int(15e6)    # полоса фильтра

# Размер FFT для одного блока
FFT_SIZE = 256

# Текущее описание свипа. start_sweep изменяет эти значения
STEP_COUNT = 16
_freq_start_mhz = 0.0
_step_mhz = 1.0

# Двойной буфер с результатами. Переключаемся между буферами чтобы не копировать
_buffers = [np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32),
            np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32)]
_buffer_ptrs = [ffi.cast("float *", _buffers[0].ctypes.data),
                ffi.cast("float *", _buffers[1].ctypes.data)]
_active_buf = 0

_callback: Callable[[np.ndarray], None]

# Список слейвов: каждая запись содержит устройство и словарь с состоянием
_slaves: list[tuple[ffi.CData, dict]] = []

# Блокировка, чтобы слейвы не измеряли частоту одновременно
rssi_lock = Lock()


def load_config(path: str = "config.json") -> dict:
    """Загрузить настройки свипа из JSON файла."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@ffi.callback("int(hackrf_transfer*)")
def _rx_callback(transfer) -> int:
    """C-коллбэк для ``hackrf_start_rx_sweep``.

    Функция передаёт буфер в C-код, который делает FFT. Когда полный свип
    готов, вызывается переданный Python-обработчик.
    """
    global _active_buf
    finished = lib.hs_process(transfer, _buffer_ptrs[_active_buf])
    if finished:
        ready = _active_buf
        _active_buf ^= 1
        _callback(_buffers[ready])
    return 0


@ffi.callback("int(hackrf_transfer*)")
def _rssi_callback(transfer) -> int:
    """Коллбэк для слейвов: C-код считает FFT, мы лишь забираем результат."""
    ctx = ffi.from_handle(transfer.rx_ctx)
    if not ctx["pending"]:
        return 0
    ctx["pending"] = False
    ctx["result"] = lib.hs_rssi(transfer)
    ctx["event"].set()
    return 0


def measure_rssi(freq_hz: float, timeout: float = 1.0) -> tuple[float, list[float]]:
    """Попросить все слейвы измерить уровень на ``freq_hz``."""
    ts = time.time()
    results: list[float] = []
    # Все операции выполняем под одной блокировкой
    with rssi_lock:
        # Настраиваем каждую плату на нужную частоту
        for dev, ctx in _slaves:
            ctx["event"].clear()
            ctx["pending"] = True
            lib.hackrf_set_freq(dev, int(freq_hz))
        # Ждём результаты
        for dev, ctx in _slaves:
            if not ctx["event"].wait(timeout):
                ctx["pending"] = False
                results.append(float("nan"))
            else:
                results.append(ctx["result"])
    return ts, results


def start_sweep(
    callback: Callable[[np.ndarray], None], *,
    config_path: str = "config.json",
    serial: Optional[str] = None,
    stop_event: Optional[Event] = None,
) -> None:
    """Запустить свип и вызывать ``callback`` для каждого готового свипа.

    Parameters
    ----------
    callback:
        Пользовательский обработчик одного спектра.
    config_path:
        Путь к JSON-файлу с параметрами свипа.
    serial:
        Серийный номер устройства HackRF One. Если ``None``, используется
        первое доступное устройство.
    stop_event:
        Объект :class:`threading.Event`, установка которого останавливает
        свип. Если не указан, функция блокируется бессрочно (как раньше).
    """

    global _callback, STEP_COUNT, _buffers, _buffer_ptrs, _freq_start_mhz, _step_mhz, _active_buf

    cfg = load_config(config_path)
    sample_rate = int(cfg.get("sample_rate", DEFAULT_SAMPLE_RATE))
    bandwidth = int(cfg.get("bandwidth", DEFAULT_BANDWIDTH))
    freq_start_mhz = float(cfg.get("freq_start_mhz", 50.0))
    freq_stop_mhz = float(cfg.get("freq_stop_mhz", 6000.0))
    step_mhz = float(cfg.get("step_mhz", 5.0))
    vga_gain = int(cfg.get("vga_gain", 20))
    lna_gain = int(cfg.get("lna_gain", 16))
    fft_threads = int(cfg.get("fft_threads", os.cpu_count() or 1))

    _callback = callback
    _freq_start_mhz = freq_start_mhz
    _step_mhz = step_mhz
    STEP_COUNT = int((freq_stop_mhz - freq_start_mhz) / step_mhz)
    _buffers = [np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32),
                np.zeros((STEP_COUNT, FFT_SIZE), dtype=np.float32)]
    _buffer_ptrs = [ffi.cast("float *", _buffers[0].ctypes.data),
                    ffi.cast("float *", _buffers[1].ctypes.data)]
    _active_buf = 0

    lib.hs_prepare(FFT_SIZE, STEP_COUNT, fft_threads)

    if lib.hackrf_init() != 0:
        raise RuntimeError("hackrf_init failed")

    # Определяем подключённые устройства и открываем мастер и слейвы
    dev_list = lib.hackrf_device_list()
    serials = [
        ffi.string(dev_list.serial_numbers[i]).decode()
        for i in range(dev_list.devicecount)
    ]
    lib.hackrf_device_list_free(dev_list)

    if not serials:
        lib.hackrf_exit()
        raise RuntimeError("HackRF device not found")

    master_serial = serial or serials[0]
    slave_serials = [s for s in serials if s != master_serial][:2]

    dev_pp = ffi.new("hackrf_device **")
    if lib.hackrf_open_by_serial(master_serial.encode(), dev_pp) != 0:
        lib.hackrf_exit()
        raise RuntimeError("hackrf_open_by_serial failed")
    dev = dev_pp[0]

    # Открываем слейвы
    _slaves.clear()
    for s in slave_serials:
        pp = ffi.new("hackrf_device **")
        if lib.hackrf_open_by_serial(s.encode(), pp) == 0:
            lib.hackrf_set_sample_rate_manual(pp[0], sample_rate, 1)
            lib.hackrf_set_baseband_filter_bandwidth(pp[0], bandwidth)
            lib.hackrf_set_vga_gain(pp[0], vga_gain)
            lib.hackrf_set_lna_gain(pp[0], lna_gain)
            ctx = {"event": Event(), "result": 0.0, "pending": False, "freq": 0}
            handle = ffi.new_handle(ctx)
            ctx["handle"] = handle  # держим ссылку, чтобы GC не удалил
            if lib.hackrf_start_rx(pp[0], _rssi_callback, handle) == 0:
                _slaves.append((pp[0], ctx))
            else:
                lib.hackrf_close(pp[0])

    try:
        lib.hackrf_set_sample_rate_manual(dev, sample_rate, 1)
        lib.hackrf_set_baseband_filter_bandwidth(dev, bandwidth)
        lib.hackrf_set_vga_gain(dev, vga_gain)
        lib.hackrf_set_lna_gain(dev, lna_gain)

        freqs = ffi.new(
            "uint16_t[2]",
            [int(freq_start_mhz), int(freq_stop_mhz)],
        )
        lib.hackrf_init_sweep(
            dev,
            freqs,
            1,
            16384,
            int(step_mhz * 1e6),
            7500000,
            0,
        )

        if lib.hackrf_start_rx_sweep(dev, _rx_callback, ffi.NULL) != 0:
            raise RuntimeError("hackrf_start_rx_sweep failed")

        try:
            stop_event = stop_event or Event()
            stop_event.wait()
        except KeyboardInterrupt:
            pass
    finally:
        for s, _ in _slaves:
            lib.hackrf_stop_rx(s)
            lib.hackrf_close(s)
        lib.hackrf_stop_rx_sweep(dev)
        lib.hackrf_close(dev)
        lib.hackrf_exit()
        lib.hs_cleanup()
