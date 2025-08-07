"""Простой помощник по работе с конфигурацией в формате JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG = {
    "freq_start": 50e6,
    "freq_stop": 6e9,
    "freq_step": 5e6,
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
            if "freq_start_mhz" in loaded:
                loaded["freq_start"] = loaded["freq_start_mhz"] * 1e6
            if "freq_stop_mhz" in loaded:
                loaded["freq_stop"] = loaded["freq_stop_mhz"] * 1e6
            if "step_mhz" in loaded:
                loaded["freq_step"] = loaded["step_mhz"] * 1e6
            cfg.update(loaded)
        except json.JSONDecodeError:
            pass
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Сохранить конфигурацию в JSON-файл."""
    out = cfg.copy()
    out["freq_start_mhz"] = cfg["freq_start"] / 1e6
    out["freq_stop_mhz"] = cfg["freq_stop"] / 1e6
    out["step_mhz"] = cfg.get("freq_step", 0) / 1e6
    CONFIG_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
