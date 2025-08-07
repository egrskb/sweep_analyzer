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
        self.baseline_curve = self.plot.plot(pen=pg.mkPen("g"))
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#aaa"))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#aaa"))
        self.plot.addItem(self.v_line, ignoreBounds=True)
        self.plot.addItem(self.h_line, ignoreBounds=True)
        self.label = pg.TextItem(color="w")
        self.plot.addItem(self.label)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.setLabel("left", "Мощность", units="дБ")
        self.plot.setLabel("bottom", "Частота", units="Гц")

        self._avg_buffer: deque[np.ndarray] = deque(maxlen=5)
        self._baseline: np.ndarray | None = None

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

        self._baseline = (
            smooth if self._baseline is None else np.minimum(self._baseline, smooth)
        )
        self.baseline_curve.setData(freqs, self._baseline)

        return smooth

    def _mouse_moved(self, pos) -> None:  # pragma: no cover - UI callback
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot.plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()
        y = mouse_point.y()
        self.v_line.setPos(x)
        self.h_line.setPos(y)
        self.label.setText(f"{x/1e6:.2f} МГц\n{y:.1f} дБм")
        self.label.setPos(x, y)

    def set_levels(self, vmin: float, vmax: float) -> None:
        """Установить диапазон уровней по оси Y."""
        self.plot.setYRange(vmin, vmax, padding=0)

    def reset_view(self) -> None:
        """Очистить график и сбросить масштаб."""
        self.plot.enableAutoRange(True, True)
        self.curve.clear()
        self.peak_curve.clear()
        self.baseline_curve.clear()
        self._avg_buffer.clear()
        self._baseline = None
