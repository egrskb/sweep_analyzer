"""Виджет водопада спектра."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore


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
        cmap = pg.colormap.get("inferno")
        self.img.setLookupTable(cmap.getLookupTable())
        self.plot.addItem(self.img)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)
        layout.addWidget(self.plot)
        self.data = np.zeros((self.size, 1024))
        self._scaled = False

    def update_spectrum(self, freqs: np.ndarray, power: np.ndarray) -> None:
        """Сдвинуть изображение вверх и добавить новый спектр."""
        self.data = np.roll(self.data, -1, axis=0)
        if power.size != self.data.shape[1]:
            self.data = np.zeros((self.size, power.size))
        self.data[-1, :] = power
        if not self._scaled:
            xscale = (freqs[-1] - freqs[0]) / power.size
            self.img.scale(xscale, 1)
            self.img.setPos(freqs[0], -self.size)
            self.plot.setYRange(-self.size, 0)
            self._scaled = True
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
        self._scaled = False

    def _plot_clicked(self, evt) -> None:  # pragma: no cover - UI callback
        if evt.button() == QtCore.Qt.LeftButton:
            self.reset_view()
