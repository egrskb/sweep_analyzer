import numpy as np
from core.fft import FFTProcessor


def test_fft_peak():
    sr = 1e3
    t = np.arange(1024) / sr
    freq = 100
    iq = np.exp(2j * np.pi * freq * t).astype(np.complex64)
    fft = FFTProcessor(sample_rate=sr)
    freqs, power = fft.process(iq)
    peak_freq = freqs[np.argmax(power)]
    assert abs(peak_freq - freq) < 1.0


def test_rssi_conversion():
    sr = 1e3
    t = np.arange(1024) / sr
    freq = 100
    # два сигнала разной амплитуды
    iq1 = np.exp(2j * np.pi * freq * t).astype(np.complex64)
    iq2 = (0.5 * np.exp(2j * np.pi * freq * t)).astype(np.complex64)
    fft = FFTProcessor(sample_rate=sr)
    _, power1 = fft.process(iq1)
    _, power2 = fft.process(iq2)
    peak1 = power1[np.argmax(power1)]
    peak2 = power2[np.argmax(power2)]
    # разница между амплитудами 1 и 0.5 должна быть около 6 дБ
    assert abs((peak1 - peak2) - 6.0) < 1.0
