"""Простая обёртка для свипа HackRF и обработки данных.

Функция :func:`start_sweep` настраивает устройство, запускает свип
и вызывает переданный коллбэк для каждого готового свипа. Код минимален и
показывает, как обернуть C API через модуль :mod:`cffi`.
"""

from __future__ import annotations

import json
import os
import time
from threading import Event, Thread
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

# Список дополнительных плат для измерения RSSI
_slaves: list[ffi.CData] = []

# Поправка переводящая мощность FFT в дБм (подбирается экспериментально)
RSSI_OFFSET_DBM = -70.0


def _top3_mean(arr: np.ndarray) -> float:
    """Найти максимальное среднее по трём подряд идущим бинам."""
    if arr.size < 3:
        return float(arr.mean())
    window_sum = float(arr[0] + arr[1] + arr[2])
    max_mean = window_sum / 3.0
    for i in range(3, arr.size):
        window_sum += float(arr[i] - arr[i - 3])
        mean = window_sum / 3.0
        if mean > max_mean:
            max_mean = mean
    return max_mean


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
    """Принять буфер IQ, сделать FFT и вернуть среднее по трём пикам."""
    data = ffi.from_handle(transfer.rx_ctx)
    buf = ffi.buffer(transfer.buffer, transfer.valid_length)
    iq = (
        np.frombuffer(buf, dtype=np.int8).astype(np.float32).reshape(-1, 2)
    )
    # Убираем постоянную составляющую и нормируем амплитуду
    iq -= iq.mean(axis=0)
    iq /= 128.0
    # Окно Ханна
    win = np.hanning(len(iq))
    sig = (iq[:, 0] + 1j * iq[:, 1]) * win
    fft = np.fft.fft(sig)
    pwr = np.abs(fft) ** 2
    p_dbm = 10 * np.log10(pwr + 1e-12) + RSSI_OFFSET_DBM
    data["result"] = _top3_mean(p_dbm)
    data["event"].set()
    return 0


def measure_rssi(freq_hz: float) -> tuple[float, list[float]]:
    """Измерить RSSI на частоте ``freq_hz`` на всех slave-платах.

    Возвращает отметку времени и список значений RSSI.
    """
    timestamp = time.time()
    results: list[float] = [0.0] * len(_slaves)
    threads: list[Thread] = []
    ctxs = []

    def worker(dev: ffi.CData, ctx, idx: int) -> None:
        lib.hackrf_set_freq(dev, int(freq_hz))
        lib.hackrf_start_rx(dev, _rssi_callback, ctx)
        data = ffi.from_handle(ctx)
        data["event"].wait()
        lib.hackrf_stop_rx(dev)
        results[idx] = data["result"]

    for i, dev in enumerate(_slaves):
        event = Event()
        ctx = ffi.new_handle({"event": event, "result": 0.0})
        ctxs.append(ctx)
        t = Thread(target=worker, args=(dev, ctx, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return timestamp, results


def start_sweep(
    callback: Callable[[np.ndarray], None], *,
    config_path: str = "config.json",
    serial: Optional[str] = None,
) -> None:
    """Запустить свип и вызывать ``callback`` для каждого готового свипа."""

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
            _slaves.append(pp[0])

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
            while lib.hackrf_is_streaming(dev):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    finally:
        for s in _slaves:
            lib.hackrf_close(s)
        lib.hackrf_close(dev)
        lib.hackrf_exit()
        lib.hs_cleanup()
