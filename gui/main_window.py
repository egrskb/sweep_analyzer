"""Главное окно приложения."""
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
    """Фоновый поток, выполняющий циклический съём спектра."""

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
    """Основной графический интерфейс анализатора спектра."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Анализатор спектра")
        self.cfg = config.load_config()

        self.fft_plot = FFTPlot()
        self.waterfall = WaterfallPlot()
        center = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(center)
        layout.addWidget(self.fft_plot)
        layout.addWidget(self.waterfall)
        self.setCentralWidget(center)

        self.toolbar = self.addToolBar("Главная")
        self.start_action = self.toolbar.addAction("Старт", self.start)
        self.stop_action = self.toolbar.addAction("Стоп", self.stop)

        self.device: SDRDevice | None = None
        self.worker: SweepWorker | None = None

        self._build_menu()

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")
        file_menu.addAction("Экспорт спектра", self.export_spectrum)
        file_menu.addAction("Экспорт водопада", self.export_waterfall)
        file_menu.addAction("Сохранить CSV", self.save_csv)

        device_menu = menubar.addMenu("Главный SDR")
        self.device_group = QtWidgets.QActionGroup(self)
        self.device_group.setExclusive(True)
        self._device_menu = device_menu
        self.refresh_devices()

        view_menu = menubar.addMenu("Вид")
        cmap_menu = view_menu.addMenu("Цветовая карта")
        for name in ["viridis", "plasma", "inferno", "magma"]:
            action = cmap_menu.addAction(name, lambda _, n=name: self.set_colormap(n))

        help_menu = menubar.addMenu("Справка")
        help_menu.addAction("О программе", self.show_about)
        help_menu.addAction("Руководство", self.show_guide)

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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт спектра", filter="Файлы PNG (*.png)")
        if path:
            exporter = ImageExporter(self.fft_plot.plot.plotItem)
            exporter.export(path)

    def export_waterfall(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт водопада", filter="Файлы PNG (*.png)")
        if path:
            exporter = ImageExporter(self.waterfall.view.scene())
            exporter.export(path)

    def set_colormap(self, name: str) -> None:
        cmap = pg.colormap.get(name)
        self.waterfall.img.setLookupTable(cmap.getLookupTable())

    def save_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить CSV", filter="Файлы CSV (*.csv)")
        if path and getattr(self.fft_plot.curve, 'xData', None) is not None:
            freqs = self.fft_plot.curve.xData
            power = self.fft_plot.curve.yData
            dlogging.save_csv(Path(path), freqs, power)

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "О программе",
            "<b>Анализатор спектра</b><br>Версия 0.1<br><a href='https://github.com'>Репозиторий</a><br>Демонстрация модульной архитектуры.",
        )

    def show_guide(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Руководство",
            "Используйте Старт/Стоп для управления съёмом. Выберите главный SDR в меню. Экспорт изображений доступен в меню Файл.",
        )


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run()
