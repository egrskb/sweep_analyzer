"""Reusable FFT plot widget using pyqtgraph."""
from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtWidgets
import numpy as np


class FFTPlot(QtWidgets.QWidget):
    """Widget displaying a single spectrum trace."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.curve = self.plot.plot(pen=pg.mkPen("y"))
        self.plot.setLabel("left", "Power", units="dB")
        self.plot.setLabel("bottom", "Frequency", units="Hz")

    def update_spectrum(self, freqs: np.ndarray, power: np.ndarray) -> None:
        """Update plot with new data."""
        self.curve.setData(freqs, power)
