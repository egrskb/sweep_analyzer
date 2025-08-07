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
