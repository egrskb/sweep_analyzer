from core.sdr import MockSDR, enumerate_devices


def test_mock_sweep():
    dev = MockSDR()
    received = []
    dev.sweep(lambda p: received.append(p))
    assert received and received[0].ndim == 1


def test_enumerate_devices_returns_list():
    devices = enumerate_devices()
    assert isinstance(devices, list)
