"""FFT processing utilities for spectrum analyzer."""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Deque, Optional, Tuple


class FFTProcessor:
    """Compute FFT and apply spectral filters.

    This class keeps internal state for averaging, min/max hold and
    persistence filters.  The public :meth:`process` method accepts IQ samples
    and returns frequency bins and power values in dB.

    Attributes:
        sample_rate: Sampling rate of the incoming IQ stream in Hz.
        fft_size: Number of points in FFT; determined from input length.
        avg_window: Number of frames used for running average.
    """

    def __init__(self, sample_rate: float, avg_window: int = 1) -> None:
        self.sample_rate = sample_rate
        self.avg_window = max(1, avg_window)
        self._avg_buffer: Deque[np.ndarray] = deque(maxlen=self.avg_window)
        self._min_hold: Optional[np.ndarray] = None
        self._max_hold: Optional[np.ndarray] = None
        self._persistence: Optional[np.ndarray] = None

    def process(self, iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate FFT and return frequencies and magnitudes in dB.

        Args:
            iq: Complex64 numpy array of IQ samples.

        Returns:
            Tuple of (frequencies, powers_db).
        """
        self.fft_size = len(iq)
        window = np.hanning(self.fft_size)
        spectrum = np.fft.fftshift(np.fft.fft(iq * window))
        power = 20 * np.log10(np.abs(spectrum) + 1e-12)
        freqs = np.linspace(-self.sample_rate / 2, self.sample_rate / 2, self.fft_size)

        # Average filter
        self._avg_buffer.append(power)
        avg_power = np.mean(np.vstack(self._avg_buffer), axis=0)

        # Min/Max hold
        self._min_hold = power if self._min_hold is None else np.minimum(self._min_hold, power)
        self._max_hold = power if self._max_hold is None else np.maximum(self._max_hold, power)

        # Persistence (decays old values)
        if self._persistence is None:
            self._persistence = power
        else:
            self._persistence = 0.9 * self._persistence + 0.1 * power

        return freqs, avg_power

    def current_min_max(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return min and max hold arrays."""
        return self._min_hold, self._max_hold

    def current_persistence(self) -> Optional[np.ndarray]:
        """Return persistence array."""
        return self._persistence
