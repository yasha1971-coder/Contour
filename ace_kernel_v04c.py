# ACE v0.4c — Coherent Breathing (Drift attractor + adaptive coupling)
import numpy as np
import csv, time
from collections import deque

# ---------- ПАРАМЕТРЫ ----------
STEPS   = 6000
EPSILON = 0.74        # базовый «уровень поля»
DELTA   = 0.11        # ширина окрестности вокруг EPSILON
ETA     = 0.014       # шаг эволюции
DIM     = 16          # размер скрытого состояния

# связь Ω′↔Λ′
COUP_LA_TO_OM = 0.08  # Λ′ → Ω′
COUP_OM_TO_LA = 0.07  # Ω′ → Λ′
OMEGA_SET     = 1.0
LAMBDA_SET    = 0.14
ELASTIC       = 0.06  # дыхание Ω′ ±6%

# память/ритм
MEM_DECAY = 0.965
MEM_CAP   = 0.25

# окна
WIN_VAR   = 60
WIN_CORR  = 200

# безопасные границы шагов
CLIP_STEP = 0.05

# ---------- УТИЛИТЫ ----------
def nanfix(x, fallback=0.0, lo=None, hi=None):
    x = np.nan_to_num(x, nan=fallback, posinf=fallback, neginf=fallback)
    if lo is not None or hi is not None:
        lo = -np.inf if lo is None else lo
        hi =  np.inf if hi is None else hi
        x = np.minimum(np.maximum(x, lo), hi)
    return x

def rolling_corr(xs, ys):
    if len(xs) < WIN_CORR or len(ys) < WIN_CORR: return 0.0
    a = np.array(xs)[-WIN_CORR:]
    b = np.array(ys)[-WIN_CORR:]
    a = a - a.mean(); b = b - b.mean()
    da = a.std(); db = b.std()
    if da < 1e-12 or db < 1e-12: return 0.0
    return float(np.dot(a, b) / (len(a)*da*db))

def safe_var_win(arr, w=WIN_VAR):
    if len(arr) < w: return None
    seg = np.array(arr)[-w:]
    seg = seg - seg.mean()
    return float((seg**2).mean())

def regime_from_dphi(dphi, prev):
    low  = EPSILON - DELTA
    high = EPSILON + DELTA
    # Гистерезис
    hyst = 0.03
    if prev == "Drift":
        low  -= hyst
        high += hyst
    elif prev == "Lock":
        low  += 0.01
    elif prev == "Iterate":
        high -= 0.01
    if dphi < low:   return "Lock"
    if dphi > high:  return "Iterate"
    return "Drift"

def external_signal(t):
    # «обучающее» окно → мягкий стресс
    # первые 3000 тиков — синус + шум; дальше редкие спайки
    base = EPSILON + 0.1*np.sin(2*np.pi*t/180.0) + np.random.normal(0, 0.03)
    if t > 3000 and np.random.rand() < 0.01:
        base += np.random.choice([0.3, -0.25])  # редкие сильные события
    return base

# ---------- СИМУЛЯЦИЯ ----------
def run():
    X = np.random.randn(DIM)
    X = X/ max(1e-6, np.linalg.norm(X))

    OmegaPrime  = 1.0
    LambdaPrime = 0.14
    mem_omega   = 0.0
    mem_lambda  = 0.0

    regime = "Drift"
    last_regime = regime

    # истории
    h_domega, h_dlambda = [], []
    h_omega, h_lambda   = [], []
    h_regime            = []
    trans_count = 0
    longest_drift, cur_drift = 0, 0

    # CSV лог
    log_name = "ace_log_v04c.csv"
    f = open(log_name, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t","regime","dphi","Omega_prime","Lambda_prime",
                "dOmega","dLambda","corr_roll"])

    for t in range(STEPS):
        dphi = external_signal(t)
        regime = regime_from_dphi(dphi, last_regime)
        if regime != last_regime:
            trans_count += 1
        last_regime = regime

        # Базовые изменения скрытого состояния
        X = X + ETA * np.tanh(X) + np.random.normal(0, 0.01, size=DIM)
        X = X / max(1e-6, np.linalg.norm(X))

        # Базовая динамика Ω′, Λ′
        dOmega  = 0.0
        dLambda = 0.0

        # Klein-guard — мягко тянем к центру вблизи границ
        if abs(dphi - EPSILON) < 0.5*DELTA:
            mix = 0.6
            OmegaPrime = mix*OmegaPrime + (1-mix)*OMEGA_SET

        # ПАМЯТЬ (с ограничением)
        mem_omega  = nanfix(mem_omega  * MEM_DECAY + (OmegaPrime  - OMEGA_SET), 0.0)
        mem_lambda = nanfix(mem_lambda * MEM_DECAY + (LambdaPrime - LAMBDA_SET), 0.0)
        mem_omega  = float(np.clip(mem_omega,  -MEM_CAP, MEM_CAP))
        mem_lambda = float(np.clip(mem_lambda, -MEM_CAP, MEM_CAP))

        # АДАПТИВНАЯ СВЯЗЬ
        corr_roll = rolling_corr(h_domega, h_dlambda)
        gain = 1.0 + 0.5*max(0.0, corr_roll)

        dOmega  += gain * COUP_LA_TO_OM * (LambdaPrime - LAMBDA_SET)
        dLambda += gain * COUP_OM_TO_LA * (OmegaPrime  - OMEGA_SET)

        # ИНЕРЦИЯ на Drift — поддерживаем ритм
        if regime == "Drift":
            dOmega  += 0.10 * mem_omega
            dLambda += 0.12 * mem_lambda

        # немножко процессного шума
        dOmega  += np.random.normal(0, 0.002)
        dLambda += np.random.normal(0, 0.002)

        # клиппинг
        dOmega  = float(np.clip(dOmega,  -CLIP_STEP, CLIP_STEP))
        dLambda = float(np.clip(dLambda, -CLIP_STEP, CLIP_STEP))

        # обновление
        newOmega  = OmegaPrime  + dOmega
        newLambda = LambdaPrime + dLambda

        # эластичность Ω′ вокруг центра
        if abs(newOmega - OMEGA_SET) > ELASTIC:
            newOmega = OMEGA_SET + (newOmega - OMEGA_SET)*0.7

        # защита от NaN/Inf
        newOmega  = nanfix(newOmega,  OMEGA_SET, OMEGA_SET-1.0, OMEGA_SET+1.0)
        newLambda = nanfix(newLambda, LAMBDA_SET, LAMBDA_SET-1.0, LAMBDA_SET+1.0)

        # финал шага
        OmegaPrime, LambdaPrime = newOmega, newLambda

        h_omega.append(OmegaPrime); h_lambda.append(LambdaPrime)
        h_domega.append(dOmega);    h_dlambda.append(dLambda)
        h_regime.append(regime)

        # drift streaks
        if regime == "Drift":
            cur_drift += 1
            longest_drift = max(longest_drift, cur_drift)
        else:
            cur_drift = 0

        w.writerow([t, regime, dphi, OmegaPrime, LambdaPrime, dOmega, dLambda, corr_roll])

    f.close()

    # QC-итоги
    drift_share  = h_regime.count("Drift")   / STEPS
    lock_share   = h_regime.count("Lock")    / STEPS
    iter_share   = h_regime.count("Iterate") / STEPS
    var_omega    = safe_var_win(h_omega) or 0.0
    corr_last    = rolling_corr(h_domega, h_dlambda)

    summary = {
        "steps": STEPS,
        "regime_share": {"Drift": drift_share, "Lock": lock_share, "Iter": iter_share},
        "transitions": trans_count,
        "longest_drift": longest_drift,
        "omega_var_win": var_omega,
        "corr_last": corr_last,
        "alive": drift_share >= 0.35 and lock_share <= 0.45 and corr_last >= 0.25 and var_omega < 1e-3
    }
    # краткий вывод
    print("=== ACE v0.4c — run summary ===")
    for k,v in summary.items(): print(f"{k}: {v}")
    return log_name, summary

if __name__ == "__main__":
    run()
