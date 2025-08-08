import numpy as np

from core import sdr


def test_sdrdevice_sweep(monkeypatch):
    captured = []

    def simulate_start_sweep(cb, config_path=None, serial=None, stop_event=None):
        """Модель поведения ``hackrf_sweep``.

        Генерирует двумерный массив так же, как это делает настоящая
        библиотека: несколько "шагов" по частоте, каждый содержит набор
        значений мощности. Затем буфер разворачивается в одномерный вид,
        что повторяет реальный вывод C-расширения.
        """

        sweep = np.linspace(-80.0, -20.0, 256 * 2, dtype=np.float32).reshape(2, 256)
        cb(sweep)

    monkeypatch.setattr(sdr, "start_sweep", simulate_start_sweep)
    dev = sdr.SDRDevice("SER")
    dev.sweep(lambda p: captured.append(p))
    assert captured and captured[0].shape == (512,) and np.isclose(captured[0][0], -80.0)


def test_sdrdevice_sweep_stop_event(monkeypatch):
    from threading import Event

    captured = []

    def simulate_start_sweep(cb, config_path=None, serial=None, stop_event=None):
        i = 0
        while not stop_event.is_set():
            # Каждый свип содержит один шаг с 4 бинами для простоты
            sweep = np.full((1, 4), float(i), dtype=np.float32)
            cb(sweep)
            i += 1
            if i == 2:
                stop_event.set()

    monkeypatch.setattr(sdr, "start_sweep", simulate_start_sweep)
    dev = sdr.SDRDevice("SER")
    stop = Event()
    dev.sweep(lambda p: captured.append(p), stop_event=stop)
    assert len(captured) == 2 and all(arr.shape == (4,) for arr in captured)


def test_enumerate_devices_returns_list():
    devices = sdr.enumerate_devices()
    assert isinstance(devices, list)

