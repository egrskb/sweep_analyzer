"""Утилиты FFT для анализатора спектра."""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Deque, Optional, Tuple


class FFTProcessor:
    """Вычисление FFT и применение спектральных фильтров.

    Класс хранит внутреннее состояние для усреднения, режимов
    минимального/максимального удержания и персистентности.
    Публичный метод :meth:`process` принимает IQ-сэмплы и
    возвращает частотные бинны и значения мощности в дБ.

    Атрибуты:
        sample_rate: частота дискретизации входного потока IQ в Гц.
        fft_size: количество точек FFT, определяется по длине входа.
        avg_window: размер окна для скользящего усреднения.
    """

    def __init__(self, sample_rate: float, avg_window: int = 1) -> None:
        self.sample_rate = sample_rate
        self.avg_window = max(1, avg_window)
        self._avg_buffer: Deque[np.ndarray] = deque(maxlen=self.avg_window)
        self._min_hold: Optional[np.ndarray] = None
        self._max_hold: Optional[np.ndarray] = None
        self._persistence: Optional[np.ndarray] = None

    def process(self, iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Вычислить FFT и вернуть частоты и мощности в дБ.

        Args:
            iq: массив numpy complex64 с IQ-сэмплами.

        Returns:
            Кортеж (частоты, мощности_дБ).
        """
        self.fft_size = len(iq)
        iq = iq - np.mean(iq)
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
        """Вернуть массивы минимального и максимального удержания."""
        return self._min_hold, self._max_hold

    def current_persistence(self) -> Optional[np.ndarray]:
        """Вернуть массив персистентности."""
        return self._persistence
