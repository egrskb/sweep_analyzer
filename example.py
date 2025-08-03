"""Простой пример сканера.

Мастер HackRF проходит весь диапазон, запоминает средний уровень шума
по первым пяти свипам и дальше ищет частоты, которые изменились
на 10 дБ и более. Если подряд найдено три и больше бина с таким
изменением, частота считается сигналом. Для неё измеряется RSSI на
дополнительных платах и выводится информация в консоль.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from hackrf_sweep import start_sweep, load_config, measure_rssi
from hackrf_sweep.core import FFT_SIZE, ffi, lib

# Загружаем настройки, чтобы знать частоты и шаги
CONFIG = load_config()
START_MHZ = CONFIG["freq_start_mhz"]
STEP_MHZ = CONFIG["step_mhz"]
BIN_WIDTH_MHZ = CONFIG["sample_rate"] / FFT_SIZE / 1e6

# Порог изменения RSSI в децибелах
THRESHOLD_DB = 10.0

# Уровень, ниже которого считаем шумом
IGNORE_LEVEL_DBM = -100.0

# Сколько свипов усреднять для baseline
BASELINE_SWEEPS = 5

# Длина истории для расчёта стандартного отклонения
HISTORY_LEN = 10

# Накопитель для baseline и число собранных свипов
BASELINE_ACCUM: np.ndarray | None = None
BASELINE_COUNT = 0
BASELINE: np.ndarray | None = None

# История последних свипов для статистики
HISTORY = np.zeros((HISTORY_LEN, 0, 0), dtype=np.float32)  # будет переопределено позже
HIST_IDX = 0
HIST_FILLED = False

# Текущие отслеживаемые частоты
TRACKED: Dict[float, Dict[str, Any]] = {}

# Время предыдущего свипа
_last_sweep_time = time.time()

def _approx_distance(rssi_dbm: float) -> float:
    """Грубая оценка расстояния по уровню сигнала."""
    ref_rssi = -40.0  # RSSI на расстоянии 1 м
    path_loss = 2.0   # потери в свободном пространстве
    return 10 ** ((ref_rssi - rssi_dbm) / (10 * path_loss))

def list_serials() -> List[Optional[str]]:
    """Получить серийные номера подключённых плат."""
    if lib.hackrf_init() != 0:
        raise RuntimeError("hackrf_init failed")
    try:
        lst = lib.hackrf_device_list()
        serials: List[Optional[str]] = []
        for i in range(lst.devicecount):
            sn = lst.serial_numbers[i]
            serials.append(ffi.string(sn).decode() if sn != ffi.NULL else None)
        lib.hackrf_device_list_free(lst)
        return serials
    finally:
        lib.hackrf_exit()

def _find_runs(mask: np.ndarray) -> np.ndarray:
    """Пометить участки из трёх и более подряд идущих True."""
    out = np.zeros_like(mask, dtype=bool)
    start = None
    for i, val in enumerate(mask):
        if val:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 3:
                out[start:i] = True
            start = None
    if start is not None and len(mask) - start >= 3:
        out[start:] = True
    return out

def process_sweep(sweep: np.ndarray) -> None:
    """Обработка каждого свипа мастера."""
    global BASELINE_ACCUM, BASELINE_COUNT, BASELINE, HISTORY, HIST_IDX, HIST_FILLED, _last_sweep_time

    now = time.time()
    cycle = now - _last_sweep_time
    _last_sweep_time = now

    # Накопление baseline из первых пяти свипов
    if BASELINE is None:
        if BASELINE_ACCUM is None:
            BASELINE_ACCUM = np.zeros_like(sweep)
            # переопределяем историю после первого свипа
            HISTORY = np.zeros((HISTORY_LEN, *sweep.shape), dtype=np.float32)
        BASELINE_ACCUM += sweep
        BASELINE_COUNT += 1
        if BASELINE_COUNT == BASELINE_SWEEPS:
            BASELINE = BASELINE_ACCUM / BASELINE_SWEEPS
            print("baseline определен")
        return

    # Пока не собрали 10 свипов, просто накапливаем историю
    if not HIST_FILLED:
        HISTORY[HIST_IDX] = sweep
        HIST_IDX += 1
        if HIST_IDX == HISTORY_LEN:
            HIST_FILLED = True
        print(f"[i] Время свипа: {cycle:.2f} с")
        return

    # Стандартное отклонение по прошлым свипам
    std = HISTORY.std(axis=0)

    diff = sweep - BASELINE
    for step_idx in range(diff.shape[0]):
        diff_row = diff[step_idx]
        base_row = BASELINE[step_idx]
        current_row = sweep[step_idx]
        std_row = std[step_idx]

        run_up = _find_runs(diff_row >= THRESHOLD_DB)
        run_down = _find_runs(diff_row <= -THRESHOLD_DB)

        for bin_idx in range(diff_row.shape[0]):
            freq_mhz = START_MHZ + step_idx * STEP_MHZ + bin_idx * BIN_WIDTH_MHZ
            base = base_row[bin_idx]
            current = current_row[bin_idx]
            delta = current - base

            if base < IGNORE_LEVEL_DBM and current < IGNORE_LEVEL_DBM:
                continue

            tracked = freq_mhz in TRACKED
            in_run = run_up[bin_idx] or run_down[bin_idx]

            if not (tracked or in_run):
                continue

            if not tracked:
                if std_row[bin_idx] > 2:
                    continue
                slave_vals = measure_rssi(freq_mhz * 1e6)
                TRACKED[freq_mhz] = {
                    "baseline": base,
                    "master": current,
                    "slaves": slave_vals,
                }
            else:
                slave_vals = measure_rssi(freq_mhz * 1e6)
                TRACKED[freq_mhz]["master"] = current
                TRACKED[freq_mhz]["slaves"] = slave_vals

            if abs(delta) < THRESHOLD_DB:
                del TRACKED[freq_mhz]
                continue

            verb = "вырос на" if delta > 0 else "упал на"
            sign = "[+]" if delta > 0 else "[-]"
            distance = _approx_distance(current)
            print(
                f"{sign} Частота: {freq_mhz/1000:.3f} GHz | RSSI: {verb} {abs(delta):.1f} dB "
                f"(было {base:.1f} dBm → стало {current:.1f} dBm) | "
                f"Расстояние: ~{distance:.1f} м | относительно SDR Master",
            )

    print(f"[i] Время свипа: {cycle:.2f} с")

    # Сохраняем свип в историю для последующей статистики
    HISTORY[HIST_IDX % HISTORY_LEN] = sweep
    HIST_IDX += 1

if __name__ == "__main__":
    serials = list_serials()
    if not serials:
        raise RuntimeError("HackRF device not found")

    master = serials[0]
    slaves = serials[1:]

    print(f"SDR master - {master}")
    if slaves:
        for s in slaves:
            print(f"SDR slave - {s}")
    else:
        print("SDR slave - нет")

    start_sweep(process_sweep, serial=master)
