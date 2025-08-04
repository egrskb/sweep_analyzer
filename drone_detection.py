"""Пример поиска подозрительных диапазонов.

Мастер HackRF проходит весь диапазон, запоминает средний уровень шума
по первым пяти свипам и затем отслеживает участки из нескольких соседних
бинов, где средний RSSI отличается от baseline на заданное в конфигурации
значение. Для каждого диапазона выводится его средний уровень и измерения
с двух дополнительных плат (если они подключены).
"""

from __future__ import annotations

import os
import time
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hackrf_sweep import start_sweep, load_config, measure_rssi
from hackrf_sweep.core import FFT_SIZE, ffi, lib

# -------------------------- параметры из конфигурации -------------------------
CONFIG = load_config()
START_MHZ = CONFIG["freq_start_mhz"]
STOP_MHZ = CONFIG["freq_stop_mhz"]
STEP_MHZ = CONFIG["step_mhz"]
BIN_WIDTH_MHZ = CONFIG["sample_rate"] / FFT_SIZE / 1e6

# Порог отклонения от baseline в дБ
THRESHOLD_DB = CONFIG.get("threshold_db", 10.0)
# Уровень, ниже которого считаем шумом
IGNORE_LEVEL_DBM = CONFIG.get("ignore_level_dbm", -100.0)
# Минимум подряд идущих бинов для уверенного сигнала
MIN_BINS = int(CONFIG.get("min_bins", 3))
# Максимально допустимое стандартное отклонение в дБ
STDDEV_MAX_DB = CONFIG.get("stddev_max_db", 5.0)
# Сколько первых свипов усреднять для baseline
BASELINE_SWEEPS = 5
# Глубина истории для статистики
HISTORY_LEN = 10

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

# Если baseline уже сохранён, загрузим его
if os.path.exists("baseline.npy"):
    BASELINE = np.load("baseline.npy")
    HISTORY = np.zeros((HISTORY_LEN, *BASELINE.shape), dtype=np.float32)
    BASELINE_COUNT = BASELINE_SWEEPS

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
    """Вернуть интервалы подряд идущих значений True."""
    segs: List[Tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def _top3_mean(arr: np.ndarray) -> float:
    """Вернуть максимальное среднее по трём подряд идущим бинам."""
    if arr.size < 3:
        return float(arr.mean())

    # Сумма первых трёх элементов
    window_sum = float(arr[0] + arr[1] + arr[2])
    max_mean = window_sum / 3.0

    # Двигаем окно и ищем максимальное среднее
    for i in range(3, arr.size):
        window_sum += float(arr[i] - arr[i - 3])
        mean = window_sum / 3.0
        if mean > max_mean:
            max_mean = mean

    return max_mean


def _classify(start: float, end: float) -> Optional[str]:
    """Простая классификация диапазона по частоте и ширине."""
    width = end - start
    center = (start + end) / 2
    if 5740 <= center <= 5820 and 15 <= width <= 25:
        return "FPV видео"
    if 2402 <= center <= 2483 and width <= 5:
        return "дрон-телеметрия или Wi-Fi"
    if abs(center - 433) <= 1 and width <= 2:
        return "управляющий канал 433 МГц"
    if abs(center - 868) <= 1 and width <= 2:
        return "управляющий канал 868 МГц"
    return None


def _refresh_range(key: Tuple[float, float], info: Dict[str, Any]) -> None:
    """Измерить уровень на диапазоне и обновить запись."""
    start, end = key
    freq_hz = ((start + end) / 2) * 1e6
    ts, slave_vals = measure_rssi(freq_hz)
    info["slaves"] = slave_vals
    info["timestamp"] = ts
    delta = info["master"] - info["baseline"]
    slave_str = "".join(
        f" | SDR slave {i+1}: {val:.1f} dBm" for i, val in enumerate(slave_vals)
    )
    print(
        f"[t] Диапазон: {start:.0f} - {end:.0f} МГц | прирост master {delta:+.1f} дБ{slave_str}"
    )


def _update_tracked() -> None:
    """Каждую секунду обновлять RSSI для уже известных диапазонов."""
    while True:
        for key, info in list(TRACKED.items()):
            _refresh_range(key, info)
        time.sleep(1)

# ------------------------------ обработка свипа -----------------------------

def process_sweep(sweep: np.ndarray) -> None:
    """Обработка каждого свипа мастера."""
    global BASELINE_ACCUM, BASELINE_COUNT, BASELINE, HISTORY, HIST_IDX, HIST_FILLED, _last_sweep_time

    now = time.time()
    cycle = now - _last_sweep_time
    _last_sweep_time = now

    # Если baseline загружен заранее, но размер истории ещё не задан
    global HISTORY
    if BASELINE is not None and HISTORY.shape[1] == 0:
        HISTORY = np.zeros((HISTORY_LEN, *sweep.shape), dtype=np.float32)

    # ----------- накопление baseline из первых пяти свипов -----------
    if BASELINE is None:
        if BASELINE_ACCUM is None:
            BASELINE_ACCUM = np.zeros_like(sweep)
            HISTORY = np.zeros((HISTORY_LEN, *sweep.shape), dtype=np.float32)
        BASELINE_ACCUM += sweep
        BASELINE_COUNT += 1
        if BASELINE_COUNT == BASELINE_SWEEPS:
            BASELINE = BASELINE_ACCUM / BASELINE_SWEEPS
            np.save("baseline.npy", BASELINE)
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

    np.save("curr_sweep.npy", sweep)
    print("[i] Считаем отклонения...")
    std = HISTORY.std(axis=0)
    diff = sweep - BASELINE
    mask_up = diff >= THRESHOLD_DB
    mask_down = diff <= -THRESHOLD_DB

    alerts = False

    rows = np.where(mask_up.any(axis=1) | mask_down.any(axis=1))[0]
    for step_idx in rows:
        base_row = BASELINE[step_idx]
        current_row = sweep[step_idx]
        std_row = std[step_idx]

        up_segments = _segments(mask_up[step_idx])
        down_segments = _segments(mask_down[step_idx])
        for segments, sign in ((up_segments, "+"), (down_segments, "-")):
            for start_idx, end_idx in segments:
                seg_len = end_idx - start_idx
                if seg_len < MIN_BINS and seg_len > 2:
                    continue
                base_seg = base_row[start_idx:end_idx]
                curr_seg = current_row[start_idx:end_idx]
                mean_base = _top3_mean(base_seg)
                mean_curr = _top3_mean(curr_seg)
                if mean_base < IGNORE_LEVEL_DBM and mean_curr < IGNORE_LEVEL_DBM:
                    continue
                if std_row[start_idx:end_idx].mean() > STDDEV_MAX_DB:
                    continue
                if abs(mean_curr - mean_base) < THRESHOLD_DB:
                    continue
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
                        "idx": (step_idx, start_idx, end_idx),
                        "count": 1,
                        "class": _classify(key_start, key_end),
                    }
                else:
                    tracked["master"] = mean_curr
                    tracked["slaves"] = slave_vals
                    tracked["timestamp"] = ts
                    tracked["idx"] = (step_idx, start_idx, end_idx)
                    tracked["count"] += 1

                delta = mean_curr - mean_base
                info = TRACKED[key]
                label = info.get("class")
                msg = (
                    f"[{sign}] Диапазон: {key_start:.0f} - {key_end:.0f} МГц | прирост {delta:+.1f} дБ"
                )
                if label:
                    msg += f" | {label}"
                if info["count"] > 3:
                    msg += " | устойчивый сигнал"
                slave_str = "".join(
                    f" | SDR slave {i+1}: {val:.1f} dBm" for i, val in enumerate(slave_vals)
                )
                msg += slave_str
                print(msg)
                alerts = True

    # Удаляем диапазоны, если уровень приблизился к baseline
    for key, info in list(TRACKED.items()):
        step_idx, start_idx, end_idx = info["idx"]
        seg = sweep[step_idx, start_idx:end_idx]
        mean_curr = _top3_mean(seg)
        if abs(mean_curr - info["baseline"]) < THRESHOLD_DB:
            del TRACKED[key]

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
    print(
        f"Диапазон сканирования: {START_MHZ:.0f} - {STOP_MHZ:.0f} МГц | шаг {STEP_MHZ:.0f} МГц"
    )
    updater = Thread(target=_update_tracked, daemon=True)
    updater.start()

    start_sweep(process_sweep, serial=master)
