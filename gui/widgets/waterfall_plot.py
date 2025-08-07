"""Виджет водопада спектра."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets


class WaterfallPlot(QtWidgets.QWidget):
    """Отображает последовательные спектры в виде водопада."""

    def __init__(self, size: int = 200, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.size = size
        layout = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Время", units="с")
        self.plot.setLabel("bottom", "Частота", units="Гц")
        self.plot.invertY(True)
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        layout.addWidget(self.plot)
        self.data = np.zeros((self.size, 1024))

    def update_spectrum(self, freqs: np.ndarray, power: np.ndarray) -> None:
        """Сдвинуть изображение вверх и добавить новый спектр."""
        self.data = np.roll(self.data, -1, axis=0)
        if power.size != self.data.shape[1]:
            self.data = np.zeros((self.size, power.size))
        self.data[-1, :] = power
        self.img.setImage(self.data, autoLevels=False)
        self.plot.setXRange(freqs[0], freqs[-1], padding=0)

    def set_levels(self, vmin: float, vmax: float) -> None:
        """Установить уровни яркости."""
        self.img.setLevels([vmin, vmax])

    def reset_view(self) -> None:
        """Сбросить масштаб и очистить данные."""
        self.plot.enableAutoRange(True, True)
        self.data[:] = 0
        self.img.clear()
