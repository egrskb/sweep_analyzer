#!/usr/bin/env bash
# Development helper script: create virtual env, install dependencies and run GUI
set -e

# create virtual environment if not exists
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

# install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -e . cffi sphinx

# build hackrf_sweep extension
python build_hackrf_sweep.py

# launch application
python -m gui.main_window
