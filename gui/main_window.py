"""Main application window."""
from __future__ import annotations

import sys
from typing import Optional

from PyQt5 import QtCore, QtWidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from pathlib import Path

from core.fft import FFTProcessor
from core.sdr import SDRDevice, MockSDR, enumerate_devices
from gui.widgets.fft_plot import FFTPlot
from gui.widgets.waterfall_plot import WaterfallPlot
from utils import config
from utils import logging as dlogging


class SweepWorker(QtCore.QThread):
    """Background thread performing sweep measurements."""

    updated = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, device: SDRDevice, fft: FFTProcessor, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.device = device
        self.fft = fft
        self._running = False

    def run(self) -> None:  # pragma: no cover - GUI thread
        self._running = True
        while self._running:
            iq = self.device.read_samples(self.fft.fft_size if hasattr(self.fft, 'fft_size') else 1024)
            freqs, power = self.fft.process(iq)
            self.updated.emit(freqs, power)
            self.msleep(1000)

    def stop(self) -> None:
        self._running = False


class MainWindow(QtWidgets.QMainWindow):
    """Main GUI for the spectrum analyzer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sweep Analyzer")
        self.cfg = config.load_config()

        self.fft_plot = FFTPlot()
        self.waterfall = WaterfallPlot()
        center = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(center)
        layout.addWidget(self.fft_plot)
        layout.addWidget(self.waterfall)
        self.setCentralWidget(center)

        self.toolbar = self.addToolBar("Main")
        self.start_action = self.toolbar.addAction("Start", self.start)
        self.stop_action = self.toolbar.addAction("Stop", self.stop)

        self.device: SDRDevice | None = None
        self.worker: SweepWorker | None = None

        self._build_menu()

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Export Spectrum", self.export_spectrum)
        file_menu.addAction("Export Waterfall", self.export_waterfall)
        file_menu.addAction("Save CSV", self.save_csv)

        device_menu = menubar.addMenu("Master SDR")
        self.device_group = QtWidgets.QActionGroup(self)
        self.device_group.setExclusive(True)
        self._device_menu = device_menu
        self.refresh_devices()

        view_menu = menubar.addMenu("View")
        cmap_menu = view_menu.addMenu("Color Map")
        for name in ["viridis", "plasma", "inferno", "magma"]:
            action = cmap_menu.addAction(name, lambda _, n=name: self.set_colormap(n))

        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)
        help_menu.addAction("Guide", self.show_guide)

    def refresh_devices(self) -> None:
        self._device_menu.clear()
        self.device_group = QtWidgets.QActionGroup(self)
        self.device_group.setExclusive(True)
        devices = enumerate_devices() or [MockSDR()]  # fallback for development
        for dev in devices:
            action = QtWidgets.QAction(dev.serial, self, checkable=True)
            action.triggered.connect(lambda checked, d=dev: self.set_device(d))
            self.device_group.addAction(action)
            self._device_menu.addAction(action)
        first = self.device_group.actions()[0]
        first.setChecked(True)
        self.set_device(devices[0])

    def set_device(self, dev: SDRDevice) -> None:
        if self.device is not None and self.worker:
            self.worker.stop()
        self.device = dev
        try:
            self.device.open()
        except Exception:
            pass

    def start(self) -> None:
        if not self.device:
            return
        fft = FFTProcessor(sample_rate=self.cfg["sample_rate"], avg_window=self.cfg["avg_window"])
        self.worker = SweepWorker(self.device, fft)
        self.worker.updated.connect(self.update_plots)
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None

    def update_plots(self, freqs: np.ndarray, power: np.ndarray) -> None:
        self.fft_plot.update_spectrum(freqs, power)
        self.waterfall.update_spectrum(power)

    def export_spectrum(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Spectrum", filter="PNG Files (*.png)")
        if path:
            exporter = ImageExporter(self.fft_plot.plot.plotItem)
            exporter.export(path)

    def export_waterfall(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Waterfall", filter="PNG Files (*.png)")
        if path:
            exporter = ImageExporter(self.waterfall.view.scene())
            exporter.export(path)

    def set_colormap(self, name: str) -> None:
        cmap = pg.colormap.get(name)
        self.waterfall.img.setLookupTable(cmap.getLookupTable())

    def save_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", filter="CSV Files (*.csv)")
        if path and getattr(self.fft_plot.curve, 'xData', None) is not None:
            freqs = self.fft_plot.curve.xData
            power = self.fft_plot.curve.yData
            dlogging.save_csv(Path(path), freqs, power)

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Sweep Analyzer",
            "<b>Sweep Analyzer</b><br>Version 0.1<br><a href='https://github.com'>Repository</a><br>Modular architecture demo.",
        )

    def show_guide(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Guide",
            "Use Start/Stop to control sweep. Choose Master SDR from menu. Export images from File menu.",
        )


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run()
