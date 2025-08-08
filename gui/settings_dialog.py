"""Диалог настроек диапазона и уровней."""
from __future__ import annotations

from PyQt5 import QtWidgets

from utils import config


class SettingsDialog(QtWidgets.QDialog):
    """Окно редактирования параметров.

    Parameters
    ----------
    cfg:
        Текущие настройки приложения.
    parent:
        Родительский виджет.
    """

    def __init__(self, cfg: dict, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметры")
        self.cfg = cfg.copy()

        form = QtWidgets.QFormLayout(self)
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0, 6e9)
        self.start_spin.setValue(self.cfg["freq_start"])
        form.addRow("Старт, Гц", self.start_spin)

        self.stop_spin = QtWidgets.QDoubleSpinBox()
        self.stop_spin.setRange(0, 6e9)
        self.stop_spin.setValue(self.cfg["freq_stop"])
        form.addRow("Стоп, Гц", self.stop_spin)

        self.bin_spin = QtWidgets.QDoubleSpinBox()
        self.bin_spin.setRange(1, 1e6)
        self.bin_spin.setValue(self.cfg["bin_size"])
        form.addRow("Bin, Гц", self.bin_spin)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(1e3, 1e9)
        self.step_spin.setValue(self.cfg["freq_step"])
        form.addRow("Шаг, Гц", self.step_spin)

        self.level_min = QtWidgets.QDoubleSpinBox()
        self.level_min.setRange(-200, 0)
        self.level_min.setValue(self.cfg["level_min"])
        form.addRow("Ур. мин, дБ", self.level_min)

        self.level_max = QtWidgets.QDoubleSpinBox()
        self.level_max.setRange(-200, 0)
        self.level_max.setValue(self.cfg["level_max"])
        form.addRow("Ур. макс, дБ", self.level_max)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def accept(self) -> None:
        self.cfg["freq_start"] = self.start_spin.value()
        self.cfg["freq_stop"] = self.stop_spin.value()
        self.cfg["bin_size"] = self.bin_spin.value()
        self.cfg["freq_step"] = self.step_spin.value()
        self.cfg["level_min"] = self.level_min.value()
        self.cfg["level_max"] = self.level_max.value()
        config.save_config(self.cfg)
        super().accept()
