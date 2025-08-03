"""Minimal HackRF sweep Python package.

This package expects the CFFI extension :mod:`hackrf_sweep._lib` to be
available. Build it by running ``python build_hackrf_sweep.py`` before
importing :func:`start_sweep`.
"""

from importlib import import_module


def _load_extension() -> None:
    """Import the compiled CFFI extension if it exists."""

    try:  # pragma: no cover - exercised when module is missing
        import_module("hackrf_sweep._lib")
    except ModuleNotFoundError:  # pragma: no cover
        raise ImportError(
            "hackrf_sweep._lib not found. Run 'python build_hackrf_sweep.py' to "
            "compile the extension before using this package."
        ) from None


_load_extension()

from .core import start_sweep, load_config, measure_rssi

__all__ = ["start_sweep", "load_config", "measure_rssi"]
