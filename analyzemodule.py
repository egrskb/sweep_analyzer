import numpy as np

baseline = None

def process_sweep(sweep: np.ndarray):
    """Process a full sweep of power measurements."""
    global baseline
    # При первом свипе сохраняем baseline
    if baseline is None:
        baseline = sweep.copy()
        print("Baseline сохранён", baseline.shape)
        return

    # Разность текущего свипа и базового
    delta = sweep - baseline

    # Жёсткий порог 10 дБ
    mask = delta > 10.0  # изменение свыше 10 дБ

    if mask.any():
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            print(f"Аномалия на шаге {y}, бине {x}, Δ={delta[y, x]:.2f} дБ")

    # Опционально сглаживаем baseline
    baseline[:] = 0.99 * baseline + 0.01 * sweep
