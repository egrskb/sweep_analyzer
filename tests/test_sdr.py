from core.sdr import MockSDR, enumerate_devices


def test_mock_read_samples():
    dev = MockSDR()
    samples = dev.read_samples(256)
    assert samples.shape[0] == 256


def test_enumerate_devices_returns_list():
    devices = enumerate_devices()
    assert isinstance(devices, list)
