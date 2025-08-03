# HackRF Sweep Analyzer

A minimal Python package that wraps a subset of the HackRF sweep API via [CFFI](https://cffi.readthedocs.io) and performs FFT processing with [FFTW](http://www.fftw.org/).  The package exposes a single function, `start_sweep`, which streams power readings from a HackRF and delivers each completed sweep to a user‑supplied callback.

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
pip install numpy cffi
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
  "fft_threads": 1
}
```

Changing values in this file lets you control sweep range, step size, gain, and FFT threading without modifying code.

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

The repository includes `example.py` which records a baseline sweep, detects
bins rising more than 10 dB above it, and (if extra HackRFs are connected)
reports their RSSI at the same frequencies:

```bash
python example.py
```

The script lists connected HackRF devices, chooses the first as master, and prints lines such as:

```
Пик на частоте 100.00 МГц: 42.3 дБ
```

indicating the frequency and power of peaks relative to the baseline.  If
slaves are present, their RSSI readings are appended.  Interrupt with
`Ctrl+C` to stop.

## Development notes

After modifying the C sources or configuration, re-run `python build_hackrf_sweep.py` to rebuild the extension.  `start_sweep` handles initialisation and cleanup of the device, but you must have a HackRF attached for the example to run successfully.

