"""Главное окно приложения."""
from __future__ import annotations

import sys
import time
from typing import Optional
from threading import Event

from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from pathlib import Path

from core.sdr import SDRDevice, enumerate_devices
from gui.widgets.fft_plot import FFTPlot
from gui.widgets.waterfall_plot import WaterfallPlot
from utils import config
from utils import logging as dlogging


class SweepWorker(QtCore.QThread):
    """Фоновый поток, выполняющий свип через ``hackrf_sweep``."""

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
            freqs = np.arange(
                self.cfg["freq_start"],
                self.cfg["freq_stop"],
                self.cfg["bin_size"],
            )
            if freqs.size != power.size:
                freqs = np.linspace(
                    self.cfg["freq_start"],
                    self.cfg["freq_stop"],
                    power.size,
                )
            self.updated.emit(freqs, power, sweep_time, elapsed)

        try:
            self.device.sweep(handle, stop_event=self._stop_event)
        except Exception:
            pass

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        self.wait(2000)


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
        self.waterfall.plot.setXLink(self.fft_plot.plot)
        self.setCentralWidget(center)
        self.fft_plot.set_levels(self.cfg["level_min"], self.cfg["level_max"])
        self.waterfall.set_levels(self.cfg["level_min"], self.cfg["level_max"])

        # настройки диапазона
        self.settings_dock = QtWidgets.QDockWidget("Диапазон", self)
        form = QtWidgets.QFormLayout()
        w = QtWidgets.QWidget()
        w.setLayout(form)
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0, 6e9)
        self.start_spin.setValue(self.cfg["freq_start"])
        self.start_spin.valueChanged.connect(lambda v: self._update_cfg("freq_start", v))
        self.stop_spin = QtWidgets.QDoubleSpinBox()
        self.stop_spin.setRange(0, 6e9)
        self.stop_spin.setValue(self.cfg["freq_stop"])
        self.stop_spin.valueChanged.connect(lambda v: self._update_cfg("freq_stop", v))
        self.bin_spin = QtWidgets.QDoubleSpinBox()
        self.bin_spin.setRange(1, 1e6)
        self.bin_spin.setValue(self.cfg["bin_size"])
        self.bin_spin.valueChanged.connect(lambda v: self._update_cfg("bin_size", v))
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(1e3, 1e9)
        self.step_spin.setValue(self.cfg["freq_step"])
        self.step_spin.valueChanged.connect(lambda v: self._update_cfg("freq_step", v))
        self.level_min = QtWidgets.QDoubleSpinBox()
        self.level_min.setRange(-200, 0)
        self.level_min.setValue(self.cfg["level_min"])
        self.level_min.valueChanged.connect(lambda v: self._update_cfg("level_min", v))
        self.level_max = QtWidgets.QDoubleSpinBox()
        self.level_max.setRange(-200, 0)
        self.level_max.setValue(self.cfg["level_max"])
        self.level_max.valueChanged.connect(lambda v: self._update_cfg("level_max", v))
        form.addRow("Старт, Гц", self.start_spin)
        form.addRow("Стоп, Гц", self.stop_spin)
        form.addRow("Bin, Гц", self.bin_spin)
        form.addRow("Шаг, Гц", self.step_spin)
        form.addRow("Ур. мин, дБ", self.level_min)
        form.addRow("Ур. макс, дБ", self.level_max)
        self.settings_dock.setWidget(w)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.settings_dock)

        self.toolbar = self.addToolBar("Главная")
        style = self.style()
        self.start_action = self.toolbar.addAction(
            style.standardIcon(QtWidgets.QStyle.SP_MediaPlay), "Старт", self.start
        )
        self.stop_action = self.toolbar.addAction(
            style.standardIcon(QtWidgets.QStyle.SP_MediaStop), "Стоп", self.stop
        )
        self.reset_action = self.toolbar.addAction(
            style.standardIcon(QtWidgets.QStyle.SP_BrowserReload), "Сброс", self.reset_view
        )
        self.stop_action.setEnabled(False)
        start_btn = self.toolbar.widgetForAction(self.start_action)
        if start_btn:
            start_btn.setStyleSheet("background-color:#4caf50;color:black;")
        stop_btn = self.toolbar.widgetForAction(self.stop_action)
        if stop_btn:
            stop_btn.setStyleSheet("background-color:#f44336;color:black;")

        self.status = self.statusBar()
        self.status.showMessage("Готов")

        self.device: SDRDevice | None = None
        self.worker: SweepWorker | None = None

        self._build_menu()

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")
        file_menu.addAction("Экспорт спектра", self.export_spectrum)
        file_menu.addAction("Экспорт водопада", self.export_waterfall)
        file_menu.addAction("Сохранить CSV", self.save_csv)

        plot_menu = menubar.addMenu("График")
        plot_menu.addAction("Сбросить масштаб", self.reset_view)

        settings_menu = menubar.addMenu("Настройки")
        device_menu = settings_menu.addMenu("Главный SDR")
        self.device_group = QtWidgets.QActionGroup(self)
        self.device_group.setExclusive(True)
        self._device_menu = device_menu
        self.refresh_devices()

        view_menu = menubar.addMenu("Вид")
        cmap_menu = view_menu.addMenu("Цветовая карта")
        for name in ["viridis", "plasma", "inferno", "magma"]:
            cmap_menu.addAction(name, lambda checked=False, n=name: self.set_colormap(n))

        help_menu = menubar.addMenu("Справка")
        help_menu.addAction("О программе", self.show_about)
        help_menu.addAction("Документация", self.show_docs)

    def refresh_devices(self) -> None:
        self._device_menu.clear()
        self.device_group = QtWidgets.QActionGroup(self)
        self.device_group.setExclusive(True)
        devices = enumerate_devices()
        if not devices:
            noact = QtWidgets.QAction("Нет доступных", self)
            noact.setEnabled(False)
            self._device_menu.addAction(noact)
            return
        for dev in devices:
            action = QtWidgets.QAction(dev.serial, self, checkable=True)
            action.triggered.connect(lambda checked, d=dev: self.set_device(d))
            self.device_group.addAction(action)
            self._device_menu.addAction(action)
        first = self.device_group.actions()[0]
        first.setChecked(True)
        self.set_device(devices[0])

    def set_device(self, dev: SDRDevice) -> None:
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.device = dev

    def start(self) -> None:
        """Запустить свип от текущего устройства."""
        if not self.device or self.worker:
            return
        self.start_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.worker = SweepWorker(self.device, self.cfg)
        self.worker.updated.connect(self.update_plots)
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.status.showMessage("Остановлено")

    def update_plots(
        self, freqs: np.ndarray, power: np.ndarray, sweep_time: float, elapsed: float
    ) -> None:
        self.fft_plot.set_levels(self.cfg["level_min"], self.cfg["level_max"])
        filtered = self.fft_plot.update_spectrum(freqs, power)
        self.waterfall.set_levels(self.cfg["level_min"], self.cfg["level_max"])
        self.waterfall.update_spectrum(freqs, filtered)
        self.status.showMessage(
            f"Свип: {sweep_time:.2f} с | Время работы: {elapsed:.1f} с"
        )

    def _update_cfg(self, key: str, value: float) -> None:
        self.cfg[key] = value
        config.save_config(self.cfg)
        if key in {"level_min", "level_max"}:
            self.fft_plot.set_levels(self.cfg["level_min"], self.cfg["level_max"])
            self.waterfall.set_levels(self.cfg["level_min"], self.cfg["level_max"])

    def export_spectrum(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт спектра", filter="Файлы PNG (*.png)")
        if path:
            exporter = ImageExporter(self.fft_plot.plot.plotItem)
            exporter.export(path)

    def export_waterfall(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт водопада", filter="Файлы PNG (*.png)")
        if path:
            exporter = ImageExporter(self.waterfall.plot.plotItem)
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
            "<b>Анализатор спектра</b><br>Версия 0.1<br>Прототип анализатора спектра.",
        )

    def show_docs(self) -> None:
        doc_path = Path(__file__).resolve().parent.parent / "docs/_build/html/index.html"
        if doc_path.exists():
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Документация")
            layout = QtWidgets.QVBoxLayout(dlg)
            browser = QtWidgets.QTextBrowser()
            browser.setSource(QtCore.QUrl.fromLocalFile(str(doc_path)))
            browser.setStyleSheet("background:white;color:black;")
            layout.addWidget(browser)
            dlg.resize(800, 600)
            dlg.exec_()
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Документация",
                "Документация не найдена. Сгенерируйте её с помощью Sphinx.",
            )

    def reset_view(self) -> None:
        """Сбросить масштаб графиков."""
        self.fft_plot.reset_view()
        self.waterfall.reset_view()


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#232629"))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#1e1e1e"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#2b2b2b"))
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#2b2b2b"))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    app.setPalette(palette)
    pg.setConfigOptions(background="#1e1e1e", foreground="w", antialias=True)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run()
