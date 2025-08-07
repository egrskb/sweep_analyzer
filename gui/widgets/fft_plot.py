"""Многоразовый виджет графика FFT на основе pyqtgraph."""
from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtWidgets
import numpy as np


class FFTPlot(QtWidgets.QWidget):
    """Виджет, отображающий одиночный спектральный след."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.curve = self.plot.plot(pen=pg.mkPen("y"))
        self.plot.setLabel("left", "Мощность", units="дБ")
        self.plot.setLabel("bottom", "Частота", units="Гц")

    def update_spectrum(self, freqs: np.ndarray, power: np.ndarray) -> None:
        """Обновить график новыми данными."""
        self.curve.setData(freqs, power)
