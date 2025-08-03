"""Простейший Python-пакет для свипа HackRF.

Перед использованием нужно собрать расширение CFFI
``hackrf_sweep._lib`` командой ``python build_hackrf_sweep.py``.
"""

from importlib import import_module


def _load_extension() -> None:
    """Импортировать собранное CFFI‑расширение."""

    try:  # pragma: no cover - exercised when module is missing
        import_module("hackrf_sweep._lib")
    except ModuleNotFoundError:  # pragma: no cover
        raise ImportError(
            "hackrf_sweep._lib не найден. Запустите 'python build_hackrf_sweep.py' "
            "перед использованием пакета."
        ) from None


_load_extension()

from .core import start_sweep, load_config, measure_rssi

__all__ = ["start_sweep", "load_config", "measure_rssi"]
