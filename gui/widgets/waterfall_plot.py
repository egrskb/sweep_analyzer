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
        self.img = pg.ImageItem()
        self.view = pg.GraphicsLayoutWidget()
        self.vb = self.view.addViewBox()
        self.vb.setMenuEnabled(False)
        self.vb.setAspectLocked(False)
        self.vb.invertY(True)
        self.vb.addItem(self.img)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view)
        self.data = np.zeros((self.size, 1024))

    def update_spectrum(self, power: np.ndarray) -> None:
        """Сдвинуть изображение вверх и добавить новый спектр."""
        self.data = np.roll(self.data, -1, axis=0)
        if power.size != self.data.shape[1]:
            self.data = np.zeros((self.size, power.size))
        self.data[-1, :] = power
        self.img.setImage(self.data, autoLevels=True)
