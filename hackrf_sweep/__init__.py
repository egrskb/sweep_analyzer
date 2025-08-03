"""Minimal HackRF sweep Python package.

This package expects the CFFI extension :mod:`hackrf_sweep._lib` to be
available.  Build it by running ``python build_hackrf_sweep.py`` before
importing :func:`start_sweep`.
"""

try:  # pragma: no cover - exercised when module is missing
    from . import _lib  # noqa: F401  (imported for side effects)
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "hackrf_sweep._lib not found. Run 'python build_hackrf_sweep.py' to "
        "compile the extension before using this package."
    ) from exc

from .core import start_sweep

__all__ = ["start_sweep"]
