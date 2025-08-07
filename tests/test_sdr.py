from core.sdr import MockSDR, enumerate_devices
from threading import Event


def test_mock_sweep():
    dev = MockSDR()
    received = []
    stop = Event()

    def handler(p):
        received.append(p)
        stop.set()

    dev.sweep(handler, stop_event=stop)
    assert received and received[0].ndim == 1


def test_enumerate_devices_returns_list():
    devices = enumerate_devices()
    assert isinstance(devices, list)
