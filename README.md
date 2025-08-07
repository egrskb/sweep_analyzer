# HackRF Sweep Analyzer

A minimal Python package for the Panorama project that wraps a subset of the HackRF sweep API via [CFFI](https://cffi.readthedocs.io) and performs FFT processing with [FFTW](http://www.fftw.org/).  The package exposes a single function, `start_sweep`, which streams power readings from a HackRF and delivers each completed sweep to a user‑supplied callback.

## Features

- Declarative sweep configuration loaded from `config.json`
- FFT processing implemented in C using FFTW for performance
- Double‑buffered sweep storage to avoid copying in the Python callback
- Baseline capture and threshold comparison in the example scanner
- Optional RSSI readings from additional HackRF devices

## Requirements

- Python 3.8+
- [libhackrf](https://github.com/greatscottgadgets/hackrf) with headers
- libusb-1.0 development files
- FFTW3f and FFTW3f Threads libraries
- A HackRF device

On Debian/Ubuntu the native libraries can be installed with:

```bash
sudo apt install libhackrf-dev libusb-1.0-0-dev libfftw3f-dev
```

## Creating a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy cffi setuptools
```

## Building the extension

Compile the CFFI extension before using the package:

```bash
python build_hackrf_sweep.py
```

This produces the module `hackrf_sweep._lib` which is loaded automatically by the package.

## Configuration

Sweep parameters are read from `config.json` in the project root.  The default file contains:

```json
{
  "sample_rate": 20000000,
  "bandwidth": 15000000,
  "freq_start_mhz": 50.0,
  "freq_stop_mhz": 6000.0,
  "step_mhz": 5.0,
  "vga_gain": 20,
  "lna_gain": 16,
  "fft_threads": 1,
  "threshold_db": 10.0,
  "ignore_level_dbm": -100.0,
  "min_bins": 3,
  "stddev_max_db": 5.0
}
```

Changing values in this file lets you control sweep range, step size, gain, FFT threading, and detection thresholds without modifying code.

## Using the library

```python
from hackrf_sweep import start_sweep


def handle_sweep(sweep):
    # sweep is a 2D NumPy array with shape (steps, fft_size)
    print("received sweep", sweep.shape)

start_sweep(handle_sweep)
```

`start_sweep(callback, config_path="config.json", serial=None)` initialises HackRF, performs sweeps based on the configuration file, and calls `callback` with a NumPy array of power values for each completed sweep.

## Example scanner

The repository includes `drone_detection.py` which records a baseline sweep,
looks for ranges of three or more bins changing by ±10 dB and prints their
average RSSI. If extra HackRFs are connected their measurements are shown too:

```bash
python drone_detection.py
```

To perform all steps automatically, including dependency installation and
building the CFFI module, run the helper script:

```bash
./run_scanner.sh
```

The script lists connected HackRF devices, chooses the first as master, and
prints ranges like:

```
[+] Диапазон: 3200 - 3300 МГц | средний RSSI master: -35.5 dBm
```

If no anomalies are found, it prints `нет подозрительных активностей`. Press
`Ctrl+C` to stop.

## Запуск GUI

Для запуска графического интерфейса используйте вспомогательный скрипт:

```bash
./run_dev.bash
```

Он создаст виртуальное окружение, установит зависимости, соберёт расширение
`hackrf_sweep`, сгенерирует документацию Sphinx и запустит приложение.

## Development notes

After modifying the C sources or configuration, re-run `python build_hackrf_sweep.py` to rebuild the extension.  `start_sweep` handles initialisation and cleanup of the device, but you must have a HackRF attached for the example to run successfully.

