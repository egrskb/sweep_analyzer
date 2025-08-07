"""Простой помощник по работе с конфигурацией в формате JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG = {
    "freq_start": 88e6,
    "freq_stop": 108e6,
    "bin_size": 1e3,
    "sample_rate": 2e6,
    "gain": 20,
    "ppm": 0,
    "avg_window": 1,
}

CONFIG_FILE = Path("config.json")


def load_config() -> Dict[str, Any]:
    """Загрузить конфигурацию из JSON-файла.

    Отсутствующие ключи дополняются значениями по умолчанию,
    чтобы избежать ошибок доступа по ключу.
    """
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            loaded = json.loads(CONFIG_FILE.read_text())
            cfg.update(loaded)
        except json.JSONDecodeError:
            pass
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Сохранить конфигурацию в JSON-файл."""
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
