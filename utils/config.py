"""Simple JSON-based configuration helper."""
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
    """Load configuration from JSON file."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict[str, Any]) -> None:
    """Save configuration to JSON file."""
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
