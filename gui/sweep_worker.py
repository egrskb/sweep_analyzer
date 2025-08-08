"""Поток свипа HackRF."""
from __future__ import annotations

import time
from threading import Event
from typing import Optional

import numpy as np
from PyQt5 import QtCore

from core.sdr import SDRDevice


class SweepWorker(QtCore.QThread):
    """Фоновый поток, выполняющий свип через ``hackrf_sweep``.

    Parameters
    ----------
    device:
        Экземпляр открытого HackRF.
    cfg:
        Словарь настроек приложения.
    parent:
        Родительский объект Qt.
    """

    updated = QtCore.pyqtSignal(np.ndarray, np.ndarray, float, float)

    def __init__(self, device: SDRDevice, cfg: dict, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.device = device
        self.cfg = cfg
        self._stop_event: Event | None = None

    def run(self) -> None:  # pragma: no cover - GUI thread
        start_ts = time.perf_counter()
        last_ts = start_ts
        self._stop_event = Event()

        def handle(power: np.ndarray) -> None:
            nonlocal last_ts
            now = time.perf_counter()
            sweep_time = now - last_ts
            elapsed = now - start_ts
            last_ts = now
            freqs = np.arange(self.cfg["freq_start"], self.cfg["freq_stop"], self.cfg["bin_size"])
            if freqs.size != power.size:
                freqs = np.linspace(self.cfg["freq_start"], self.cfg["freq_stop"], power.size)
            self.updated.emit(freqs, power, sweep_time, elapsed)

        try:
            self.device.sweep(handle, stop_event=self._stop_event)
        except Exception:
            pass

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        self.wait(2000)
