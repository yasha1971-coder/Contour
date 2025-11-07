# xi_chart.py
# Аналитика и визуализация ACE v0.1: Ω′–Λ′–δ + Ξ
# Требования: Python 3.x, numpy, matplotlib, csv (встроенный модуль)

import csv
import math
import os
from collections import deque, defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---------- ПАРАМЕТРЫ АНАЛИТИКИ ----------
LOG_FILE = "ace_log.csv"      # лог, который пишет ace_kernel.py
OUT_PNG  = "xi_plots.png"     # общий рисунок (2 панели)
OUT_PNG2 = "regime_timeline.png"  # временная диаграмма режимов
OUT_TXT  = "xi_summary.txt"   # текстовый отчёт

ROLL_WIN = 60                 # окно "Ω′ ≈ const"
OMEGA_VAR_THRESH = 1e-2       # порог "почти константа"
LAMBDA_SLOPE_MIN = 1e-2       # минимальная «скорость дыхания» оси

# ---------- ЗАГРУЗКА ДАННЫХ ----------
if not os.path.exists(LOG_FILE):
    raise FileNotFoundError(
        f"Не найден {LOG_FILE}. Сначала запусти ace_kernel.py, "
        "чтобы он создал журнал шагов."
    )

rows = []
with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

if not rows:
    raise RuntimeError("Лог пуст. Вероятно, симуляция не записала данные.")

# Нормализуем имена полей (разные версии лога могут отличаться регистром/названием)
def pick(r, candidates, default=np.nan, cast=float):
    for c in candidates:
        if c in r:
            try:
                return cast(r[c])
            except Exception:
                return default
    # попробуем нечувствительно к регистру
    low = {k.lower(): k for k in r.keys()}
    for c in candidates:
        if c.lower() in low:
            try:
                return cast(r[low[c.lower()]])
            except Exception:
                return default
    return default

t       = np.arange(len(rows))
Omega   = np.array([pick(r, ["Omega_prime", "Omega", "omega"]) for r in rows], dtype=float)
Lambda  = np.array([pick(r, ["Lambda_prime", "Lambda", "lambda"]) for r in rows], dtype=float)
delta   = np.array([pick(r, ["delta", "Delta"]) for r in rows], dtype=float)
dphi    = np.array([pick(r, ["dphi", "d_phi", "dPhi"]) for r in rows], dtype=float)
xi      = np.array([pick(r, ["xi", "Xi"]) for r in rows], dtype=float)
tau     = np.array([pick(r, ["tau", "Tau"]) for r in rows], dtype=float)

reg_raw = [pick(r, ["regime", "state"], default=np.nan, cast=str) for r in rows]
reg = np.array(reg_raw, dtype=object)

# ---------- МЕТРИКИ ----------
# 1) Доля Drift
def share(mask):
    n = np.isfinite(mask).sum()
    if n == 0:
        return 0.0
    return float((mask == True).sum()) / len(mask)

reg_is_drift  = np.array([str(x).lower().startswith("drift")  for x in reg])
reg_is_lock   = np.array([str(x).lower().startswith("lock")   for x in reg])
reg_is_iter   = np.array([str(x).lower().startswith("iterate") for x in reg])

drift_share = share(reg_is_drift) * 100.0

# 2) Скользящая дисперсия Ω′ (var_win)
omega_var = np.full_like(Omega, np.nan, dtype=float)
dq = deque(maxlen=ROLL_WIN)
for i, val in enumerate(Omega):
    dq.append(val)
    if len(dq) == ROLL_WIN:
        omega_var[i] = float(np.var(np.fromiter(dq, float)))

# Берём медиану или среднее по валидным точкам
omega_var_win = float(np.nanmean(omega_var[np.isfinite(omega_var)]))

# 3) Среднее |dΛ′/dt|
dLambda = np.abs(np.diff(Lambda))
mean_abs_dLambda = float(np.nanmean(dLambda)) if len(dLambda) else float("nan")

# 4) Среднее Ξ (если поле есть)
xi_mean = float(np.nanmean(xi[np.isfinite(xi)])) if np.isfinite(xi).any() else float("nan")

# 5) Итоговый вердикт «ALIVE»
alive = (
    (drift_share >= 60.0) and
    (omega_var_win < OMEGA_VAR_THRESH) and
    (mean_abs_dLambda > LAMBDA_SLOPE_MIN) and
    (np.nan_to_num(delta).min(initial=1.0) > 0.0)
)

verdict = "ALIVE" if alive else "NOT ALIVE"

# ---------- ВИЗУАЛИЗАЦИЯ ----------
plt.figure(figsize=(12, 8))

# Панель 1: Ω′ и его скользящая дисперсия
ax1 = plt.subplot(2,1,1)
ax1.plot(t, Omega, label="Ω′ (circulation)")
ax1.set_ylabel("Ω′")
ax1.set_title("Ω′ и Λ′ (снизу); фон — режимы: Lock (голубой), Drift (зелёный), Iterate (оранжевый)")

# Подложка режимов (полупрозрачные прямоугольники)
def paint_regimes(axis):
    last = 0
    current = None
    colors = {"lock": (0.6, 0.8, 1.0, 0.25), "drift": (0.6, 1.0, 0.6, 0.25), "iterate": (1.0, 0.8, 0.4, 0.25)}
    for i, r in enumerate(reg):
        name = str(r).lower()
        bucket = "lock" if name.startswith("lock") else "drift" if name.startswith("drift") else "iterate" if name.startswith("iterate") else None
        if i == 0:
            current, last = bucket, 0
        elif bucket != current:
            if current in colors:
                axis.axvspan(last, i, color=colors[current], lw=0)
            current, last = bucket, i
    # закрываем последнюю полоску
    if current in colors:
        axis.axvspan(last, len(t)-1, color=colors[current], lw=0)

paint_regimes(ax1)

# Нанесём var по окну как вспомогательную линию (нормировку не делаем)
ax1_2 = ax1.twinx()
ax1_2.plot(t, omega_var, linestyle="--", alpha=0.6, label=f"var_win(Ω′), win={ROLL_WIN}")
ax1_2.set_ylabel("var_win(Ω′)")

# Легенды
lns1, labs1 = ax1.get_legend_handles_labels()
lns2, labs2 = ax1_2.get_legend_handles_labels()
ax1.legend(lns1+lns2, labs1+labs2, loc="upper right")

# Панель 2: Λ′ и |dΛ′/dt|
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax2.plot(t, Lambda, label="Λ′ (axis state)")
# |dΛ′/dt| как штриховая линия
pad = np.nan
abs_dL = np.concatenate([[pad], dLambda])
ax2.plot(t, abs_dL, linestyle="--", label="|dΛ′/dt|")
ax2.set_xlabel("step")
ax2.set_ylabel("Λ′ / |dΛ′/dt|")
paint_regimes(ax2)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=160)

# Отдельная «лента» режимов (time line)
plt.figure(figsize=(12, 2.2))
ax3 = plt.gca()
paint_regimes(ax3)
ax3.set_xlim(0, len(t)-1)
ax3.set_ylim(0, 1)
ax3.set_yticks([])
ax3.set_xlabel("step")
ax3.set_title("Regime timeline")
plt.tight_layout()
plt.savefig(OUT_PNG2, dpi=160)

# ---------- ОТЧЁТ ----------
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=== ACE v0.1 — Ξ-report ===\n")
    f.write(f"Verdict            : {verdict}\n")
    f.write(f"Drift share        : {drift_share:6.2f}%\n")
    f.write(f"var_win(Ω′)        : {omega_var_win:.6e}\n")
    f.write(f"mean |dΛ′/dt|      : {mean_abs_dLambda:.6f}\n")
    f.write(f"Ξ (mean)           : {xi_mean:.6f}\n")
    f.write(f"Alive rule         : drift≥60% & var_win(Ω′)<{OMEGA_VAR_THRESH} & |dΛ′/dt|>{LAMBDA_SLOPE_MIN} & min(δ)>0\n")
    f.write("\nFiles:\n")
    f.write(f"  {OUT_PNG}\n")
    f.write(f"  {OUT_PNG2}\n")

print("Готово.")
print(f"Вердикт: {verdict}")
print(f"Сохранено: {OUT_PNG}, {OUT_PNG2}, {OUT_TXT}")
