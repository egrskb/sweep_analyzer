"""Пример поиска подозрительных диапазонов.

Мастер HackRF проходит весь диапазон, запоминает средний уровень шума
по первым пяти свипам и затем отслеживает участки из трёх и более соседних
бинов, где средний RSSI отличается на 10 дБ и более. Для каждого такого
диапазона выводится его средний уровень и измерения с двух дополнительных
плат (если они подключены).
"""

from __future__ import annotations

import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hackrf_sweep import start_sweep, load_config, measure_rssi
from hackrf_sweep.core import FFT_SIZE, ffi, lib

# -------------------------- параметры из конфигурации -------------------------
CONFIG = load_config()
START_MHZ = CONFIG["freq_start_mhz"]
STEP_MHZ = CONFIG["step_mhz"]
BIN_WIDTH_MHZ = CONFIG["sample_rate"] / FFT_SIZE / 1e6

THRESHOLD_DB = 10.0          # порог изменения
IGNORE_LEVEL_DBM = -100.0    # значения ниже считаем шумом
BASELINE_SWEEPS = 5          # сколько свипов усреднять
HISTORY_LEN = 10             # глубина истории для статистики

# ----------------------------- рабочие переменные ---------------------------
BASELINE_ACCUM: np.ndarray | None = None
BASELINE_COUNT = 0
BASELINE: np.ndarray | None = None

HISTORY = np.zeros((HISTORY_LEN, 0, 0), dtype=np.float32)
HIST_IDX = 0
HIST_FILLED = False

# Отслеживаемые диапазоны: ключ=(start_mhz, end_mhz)
TRACKED: Dict[Tuple[float, float], Dict[str, Any]] = {}

_last_sweep_time = time.time()

# ------------------------------- вспомогательные -----------------------------

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

def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Вернуть списки индексов подряд идущих True длиной ≥3."""
    segs: List[Tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= 3:
                segs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= 3:
        segs.append((start, len(mask)))
    return segs


def _update_tracked() -> None:
    """Каждую секунду обновлять RSSI для уже известных диапазонов."""
    while True:
        for (start, end), info in list(TRACKED.items()):
            freq_hz = ((start + end) / 2) * 1e6
            ts, slave_vals = measure_rssi(freq_hz)
            info["slaves"] = slave_vals
            info["timestamp"] = ts
            if slave_vals:
                slave_str = "".join(
                    f" | SDR slave {i+1}: {val:.1f} dBm" for i, val in enumerate(slave_vals)
                )
                print(f"[t] Диапазон: {start:.0f} - {end:.0f} МГц{slave_str}")
        time.sleep(1)

# ------------------------------ обработка свипа -----------------------------

def process_sweep(sweep: np.ndarray) -> None:
    """Обработка каждого свипа мастера."""
    global BASELINE_ACCUM, BASELINE_COUNT, BASELINE, HISTORY, HIST_IDX, HIST_FILLED, _last_sweep_time

    now = time.time()
    cycle = now - _last_sweep_time
    _last_sweep_time = now

    # ----------- накопление baseline из первых пяти свипов -----------
    if BASELINE is None:
        if BASELINE_ACCUM is None:
            BASELINE_ACCUM = np.zeros_like(sweep)
            HISTORY = np.zeros((HISTORY_LEN, *sweep.shape), dtype=np.float32)
        BASELINE_ACCUM += sweep
        BASELINE_COUNT += 1
        if BASELINE_COUNT == BASELINE_SWEEPS:
            BASELINE = BASELINE_ACCUM / BASELINE_SWEEPS
            print("baseline определен")
        return

    # --------------- заполнение истории для стандартного отклонения --------------
    if not HIST_FILLED:
        HISTORY[HIST_IDX] = sweep
        HIST_IDX += 1
        if HIST_IDX == HISTORY_LEN:
            HIST_FILLED = True
        print(f"[i] Время свипа: {cycle:.2f} с")
        return

    std = HISTORY.std(axis=0)
    diff = sweep - BASELINE

    alerts = False

    for step_idx in range(diff.shape[0]):
        diff_row = diff[step_idx]
        base_row = BASELINE[step_idx]
        current_row = sweep[step_idx]
        std_row = std[step_idx]

        up_segments = _segments(diff_row >= THRESHOLD_DB)
        down_segments = _segments(diff_row <= -THRESHOLD_DB)
        for segments, sign in ((up_segments, "+"), (down_segments, "-")):
            for start_idx, end_idx in segments:
                base_seg = base_row[start_idx:end_idx]
                curr_seg = current_row[start_idx:end_idx]
                if base_seg.mean() < IGNORE_LEVEL_DBM and curr_seg.mean() < IGNORE_LEVEL_DBM:
                    continue
                if std_row[start_idx:end_idx].mean() > 2:
                    continue
                mean_base = base_seg.mean()
                mean_curr = curr_seg.mean()
                key_start = START_MHZ + step_idx * STEP_MHZ + start_idx * BIN_WIDTH_MHZ
                key_end = START_MHZ + step_idx * STEP_MHZ + end_idx * BIN_WIDTH_MHZ
                key = (key_start, key_end)

                ts, slave_vals = measure_rssi(((key_start + key_end) / 2) * 1e6)
                tracked = TRACKED.get(key)
                if not tracked:
                    TRACKED[key] = {
                        "baseline": mean_base,
                        "master": mean_curr,
                        "slaves": slave_vals,
                        "timestamp": ts,
                    }
                else:
                    TRACKED[key]["master"] = mean_curr
                    TRACKED[key]["slaves"] = slave_vals
                    TRACKED[key]["timestamp"] = ts

                if abs(mean_curr - TRACKED[key]["baseline"]) < THRESHOLD_DB:
                    del TRACKED[key]
                    continue

                slave_str = "".join(
                    f" | SDR slave {i+1}: {val:.1f} dBm" for i, val in enumerate(slave_vals)
                )
                print(
                    f"[{sign}] Диапазон: {key_start:.0f} - {key_end:.0f} МГц | "
                    f"средний RSSI master: {mean_curr:.1f} dBm{slave_str}"
                )
                alerts = True

    print(f"[i] Время свипа: {cycle:.2f} с")
    if not alerts:
        print("нет подозрительных активностей")

    HISTORY[HIST_IDX % HISTORY_LEN] = sweep
    HIST_IDX += 1

# ----------------------------- запуск сканера -------------------------------

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
    updater = Thread(target=_update_tracked, daemon=True)
    updater.start()

    start_sweep(process_sweep, serial=master)
