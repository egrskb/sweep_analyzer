import numpy as np

from core import sdr


def test_sdrdevice_sweep(monkeypatch):
    captured = []

    def fake_start_sweep(cb, config_path=None, serial=None, stop_event=None):
        cb(np.array([1.0, 2.0], dtype=np.float32))

    monkeypatch.setattr(sdr, "start_sweep", fake_start_sweep)
    dev = sdr.SDRDevice("SER")
    dev.sweep(lambda p: captured.append(p))
    assert captured and captured[0].tolist() == [1.0, 2.0]


def test_sdrdevice_sweep_stop_event(monkeypatch):
    from threading import Event

    captured = []

    def fake_start_sweep(cb, config_path=None, serial=None, stop_event=None):
        i = 0
        while not stop_event.is_set():
            cb(np.array([float(i)], dtype=np.float32))
            i += 1
            if i == 2:
                stop_event.set()

    monkeypatch.setattr(sdr, "start_sweep", fake_start_sweep)
    dev = sdr.SDRDevice("SER")
    stop = Event()
    dev.sweep(lambda p: captured.append(p), stop_event=stop)
    assert len(captured) == 2


def test_enumerate_devices_returns_list():
    devices = sdr.enumerate_devices()
    assert isinstance(devices, list)

