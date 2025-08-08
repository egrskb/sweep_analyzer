"""Точки входа командной строки."""
from __future__ import annotations

from threading import Event
import numpy as np

from utils import config
from core.sdr import enumerate_devices
from gui.main_window import run


def pan_start() -> None:
    """Запустить графический интерфейс."""
    run()


def pan_info() -> None:
    """Вывести информацию о программе."""
    print("Анализатор спектра v0.1 — прототип на базе HackRF One")


def pan_sweep() -> None:
    """Вывести результаты свипа в консоль."""
    cfg = config.load_config()
    devices = enumerate_devices()
    if not devices:
        print("HackRF не найден")
        return
    dev = devices[0]
    stop = Event()

    def handle(power: np.ndarray) -> None:
        freqs = np.arange(cfg["freq_start"], cfg["freq_stop"], cfg["bin_size"])
        for f, p in zip(freqs, power):
            print(f"{f:.0f},{p:.2f}")
        print()

    try:
        dev.sweep(handle, stop_event=stop)
    except KeyboardInterrupt:
        stop.set()
        dev.close()
