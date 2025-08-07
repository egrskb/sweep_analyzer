"""Многоразовый виджет графика FFT на основе pyqtgraph."""
from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from scipy.signal import savgol_filter, find_peaks


class FFTPlot(QtWidgets.QWidget):
    """Виджет, отображающий одиночный спектральный след."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.curve = self.plot.plot(pen=pg.mkPen("y"))
        self.peak_curve = self.plot.plot(pen=pg.mkPen("r", width=2))
        self.plot.setLabel("left", "Мощность", units="дБ")
        self.plot.setLabel("bottom", "Частота", units="Гц")

        self._avg_buffer: deque[np.ndarray] = deque(maxlen=5)

    def update_spectrum(self, freqs: np.ndarray, power: np.ndarray) -> np.ndarray:
        """Обновить график новыми данными.

        Данные предварительно усредняются и сглаживаются фильтром
        Савицкого–Голея. Отдельный красный график отображает огибающую
        пиков спектра.
        """

        self._avg_buffer.append(power)
        avg = np.mean(np.vstack(self._avg_buffer), axis=0)

        win = min(11, max(3, (len(avg) // 2) * 2 + 1))
        smooth = savgol_filter(avg, win, 2)
        self.curve.setData(freqs, smooth)

        peaks, _ = find_peaks(smooth)
        if peaks.size > 1:
            envelope = np.interp(np.arange(len(smooth)), peaks, smooth[peaks])
            self.peak_curve.setData(freqs, envelope)
        else:
            self.peak_curve.setData(freqs[peaks], smooth[peaks])

        return smooth
