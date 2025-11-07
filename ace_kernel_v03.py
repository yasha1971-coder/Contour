# ACE v0.3 — "living drift"
# Цель: удерживать Ω′ ≈ const, но заставить Λ′ дышать (|dΛ′/dt| > 1e-2)
# Изменения: мягкая пульсация в Drift, ослабленный гистерезис, логирование связки Ω′↔Λ′

import csv, math, random, time
from collections import deque
import numpy as np

# ------------ параметры (под твой v0.2 baseline) ------------
EPSILON = 0.74        # ε — центр зоны
DELTA   = 0.10        # δ — ширина жизненной зоны (±)
ETA     = 0.012       # η — шаг/темп эволюции
NOISE   = 0.06        # фоновая турбулентность Δφ
WIN     = 60          # окно для var(Ω′) и корреляций
STEPS   = 6000
OMEGA_VAR_THRESH = 1e-2
LAMBDA_SLOPE_MIN = 1e-2

# лёгкий гистерезис для "липкого" Drift (ослаблен vs v0.2)
HYST_DRIFT = 0.01

# микро-пульсации в Drift (новое)
PULSE_OMEGA_AMP  = 0.003
PULSE_LAMBDA_AMP = 0.002
PULSE_OMEGA_W    = 0.010   # частота пульса Ω′
PULSE_LAMBDA_W   = 0.017   # частота пульса Λ′

# ------------ состояние ------------
rng = random.Random(42)
np_rng = np.random.default_rng(42)

t = 0
regime = "Drift"
Omega = 0.985           # Ω′ старт (почти-константа)
Lambda = 0.14           # Λ′ стартовый ритм
delta  = DELTA

omega_hist  = deque(maxlen=WIN)
lambda_hist = deque(maxlen=WIN)
regime_hist = []
dphi_hist   = []
coupling_hist = deque(maxlen=WIN)

# ------------ утилиты ------------
def dphi_signal(t):
    # основа — синус+шум, вблизи ε: это держит систему в работе
    base = 0.70 + 0.10*math.sin(0.71*t) + np_rng.normal(0.0, NOISE)
    return max(0.0, base)

def classify_regime(dphi, regime_prev):
    low  = EPSILON - delta
    high = EPSILON + delta
    # липкость Drift: ослабленный гистерезис
    if regime_prev == "Drift":
        low  -= HYST_DRIFT
        high += HYST_DRIFT
    if dphi < low:   return "Lock"
    if dphi > high:  return "Iter"
    return "Drift"

def omega_is_const():
    if len(omega_hist) < WIN: return False
    x = np.array(omega_hist, float)
    return x.var() < OMEGA_VAR_THRESH

def lambda_is_breathing():
    if len(lambda_hist) < 3: return True
    x = np.array(lambda_hist, float)
    slope = np.abs(np.diff(x)).mean()
    return slope > LAMBDA_SLOPE_MIN

# ------------ цикл ------------
rows = []
transitions = 0
last_regime = regime

for step in range(STEPS):
    t += 1
    dphi = dphi_signal(t)
    regime = classify_regime(dphi, regime)

    if regime != last_regime:
        transitions += 1
        last_regime = regime

    # базовые эволюции
    # — небольшая релаксация Ω′ к 1.0 (контур), + шум
    target = 1.0 - min(1.0, abs(dphi - EPSILON))  # ближе к ε → выше Ω′
    dOmega = ETA * (target - Omega) + np_rng.normal(0.0, 0.001)

    # — Λ′: собственное "дыхание" зависит от рода фазы
    if regime == "Lock":
        dLambda = -0.015*Lambda + np_rng.normal(0.0, 0.0015)
    elif regime == "Iter":
        dLambda =  0.010*np.sign(np_rng.normal()) + np_rng.normal(0.0, 0.003)
    else:  # Drift — ЖИЗНЕННОЕ ДЫХАНИЕ (новое)
        dOmega  += PULSE_OMEGA_AMP  * math.sin(PULSE_OMEGA_W  * t)
        dLambda  = PULSE_LAMBDA_AMP * math.sin(PULSE_LAMBDA_W * t)
        dLambda += 0.002*np_rng.normal()  # микрошум, чтобы не застывало

    # мягкая перекрёстная связка Ω′↔Λ′ (чтобы «слышали» друг друга)
    dOmega += 0.04 * math.tanh(Lambda - 0.14)
    dLambda += 0.03 * (Omega - 0.985)

    # применяем
    Omega = float(np.clip(Omega + dOmega, 0.8, 1.2))
    Lambda = float(np.clip(Lambda + dLambda, -1.0, 1.0))

    # лог
    omega_hist.append(Omega)
    lambda_hist.append(Lambda)
    dphi_hist.append(dphi)
    regime_hist.append(regime)

    # кросс-связка (корреляция производных)
    if len(omega_hist) >= 3 and len(lambda_hist) >= 3:
        dO = np.diff(np.array(omega_hist))
        dL = np.diff(np.array(lambda_hist))
        if dO.std() > 0 and dL.std() > 0:
            corr = float(np.corrcoef(dO, dL)[0,1])
            coupling_hist.append(corr)

    rows.append({
        "step": t,
        "Omega": Omega,
        "Lambda": Lambda,
        "delta": delta,
        "dphi": dphi,
        "regime": regime
    })

# ------------ сводка ------------
drift_share = 100.0 * sum(1 for r in regime_hist if r=="Drift") / len(regime_hist)
omega_var_win = float(np.array(omega_hist).var()) if len(omega_hist)>=WIN else float('nan')
mean_abs_dLambda = float(np.abs(np.diff(np.array(lambda_hist))).mean()) if len(lambda_hist)>=2 else float('nan')
coupling_corr = float(np.mean(coupling_hist)) if len(coupling_hist)>0 else float('nan')

alive = (drift_share >= 60.0) and omega_is_const() and lambda_is_breathing()

summary = {
    "summary": {
        "steps": len(regime_hist),
        "regime_share": {
            "Drift": drift_share,
            "Lock": 100.0 - drift_share if transitions==0 else 100.0*sum(1 for r in regime_hist if r=="Lock")/len(regime_hist),
            "Iter": 100.0*sum(1 for r in regime_hist if r=="Iter")/len(regime_hist)
        },
        "transitions": transitions,
        "longest_drift": max(len(list(g)) for k,g in
            __import__("itertools").groupby(regime_hist) if k=="Drift") if "Drift" in regime_hist else 0,
        "omega_var_win": omega_var_win,
        "mean_abs_dLambda": mean_abs_dLambda,
        "coupling_corr_dO_dL": coupling_corr
    },
    "alive": alive
}
print(summary)

# сохраняем лог для xi_chart
with open("ace_log_v03.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# консольный итог
print("=== Simulation complete! v0.3 ===")
print(f"Regime share (%): Drift={drift_share:.1f}")
print(f"avg var_win(Ω′): {omega_var_win:.6f}")
print(f"avg |dΛ′/dt|   : {mean_abs_dLambda:.6f}")
print(f"corr(dΩ′, dΛ′) : {coupling_corr:.4f}")
print(f"Verdict ALIVE  : {alive}")
