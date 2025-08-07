#!/bin/bash
# Простой запуск сканера: создаёт окружение, ставит зависимости и запускает пример
set -e
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install numpy cffi
python build_hackrf_sweep.py
python drone_detection.py
