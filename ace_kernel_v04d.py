# ACE v0.4d — Coherent Breathing (drift loosened, bidirectional coupling strengthened)
# Logs: ace_log_v04d.csv  |  Report dir: ace_v04d_report/

import numpy as np
import csv, os
from collections import deque

# ===== Tunables =====
STEPS          = 6000
EPSILON        = 0.74
DELTA          = 0.10
ETA            = 0.012

# Coupling (Λ'↔Ω')
COUP_LA_TO_OM  = 0.10   # Λ' -> Ω'
COUP_OM_TO_LA  = 0.09   # Ω' -> Λ'

# Elasticity of Ω' around identity (1.0): lower = мягче
ELASTIC        = 0.08   # (было 0.06)

# Hysteresis for regime stickiness
HYST           = 0.015  # (было 0.03)

# Noise (вернули «микродыхание» в Drift)
DRIFT_NOISE    = 0.004
ITER_NOISE     = 0.010
LOCK_NOISE     = 0.0015

# Soft memory
MEM_DECAY      = 0.92
MEM_GAIN_LA    = 0.08
MEM_GAIN_OM    = 0.06

# QC windows
WIN_VAR        = 120
WIN_CORR       = 240

# Perturbation probe (каждые N тиков «удар по системе»)
PROBE_PERIOD   = 1000
PROBE_MAG      = 0.05

# ====================

def classify_regime(dphi, prev_regime):
    low  = EPSILON - DELTA
    high = EPSILON + DELTA
    # hysteresis: если уже в Drift — чуть шире коридор
    if prev_regime == "Drift":
        low  -= HYST
        high += HYST
    if dphi < low:      return "Lock"
    if dphi > high:     return "Iterate"
    return "Drift"

def rolling_var(series, win):
    if len(series) < win: return np.nan
    x = np.array(series[-win:])
    return float(np.var(x))

def rolling_corr(a, b, win):
    if len(a) < win or len(b) < win: return np.nan
    x = np.array(a[-win:]); y = np.array(b[-win:])
    sx = np.std(x); sy = np.std(y)
    if sx==0 or sy==0: return 0.0
    return float(np.corrcoef(x, y)[0,1])

def clamp(x, lo, hi): return lo if x<lo else hi if x>hi else x

class ACEngine:
    def __init__(self):
        self.OmegaPrime   = 1.0
        self.LambdaPrime  = 0.14
        self.t            = 0

        # histories
        self.h_omega  = deque(maxlen=20000)
        self.h_lambda = deque(maxlen=20000)
        self.h_dphi   = deque(maxlen=20000)
        self.h_reg    = []

        # memories
        self.mem_lambda = 0.0
        self.mem_omega  = 0.0

        self.alive = False

    def step(self, dphi, prev_regime):
        # regime
        regime = classify_regime(dphi, prev_regime)

        # --- bidirectional coupling ---
        dOmega  = COUP_LA_TO_OM * (self.LambdaPrime - 0.14)
        dLambda = COUP_OM_TO_LA * (self.OmegaPrime  - 1.00)

        # soft memory (в Drift подпитываем дыхание микрошумом)
        if regime == "Drift":
            self.mem_lambda = self.mem_lambda * MEM_DECAY + (self.LambdaPrime - 0.14)
            self.mem_omega  = self.mem_omega  * MEM_DECAY + (self.OmegaPrime  - 1.00)

            dOmega  += MEM_GAIN_OM * self.mem_omega  + np.random.normal(0, DRIFT_NOISE)
            dLambda += MEM_GAIN_LA * self.mem_lambda + np.random.normal(0, DRIFT_NOISE)

        elif regime == "Iterate":
            # активная адаптация
            dOmega  += np.random.normal(0, ITER_NOISE)  + 0.02*np.sin(0.03*self.t)
            dLambda += np.random.normal(0, ITER_NOISE)  + 0.02*np.cos(0.025*self.t)
            self.mem_lambda *= MEM_DECAY
            self.mem_omega  *= MEM_DECAY

        else:  # Lock
            dOmega  += np.random.normal(0, LOCK_NOISE)
            dLambda += np.random.normal(0, LOCK_NOISE)
            self.mem_lambda *= MEM_DECAY
            self.mem_omega  *= MEM_DECAY

        # external metric drift proxy
        tau = ETA
        self.OmegaPrime  += tau * dOmega
        self.LambdaPrime += tau * dLambda

        # elastic pull (Ω' не «цементируем», а мягко возвращаем к 1.0)
        self.OmegaPrime = 1.0 + (self.OmegaPrime - 1.0) * (1.0 - ELASTIC)

        # bounded Λ' (чтобы не уносило)
        self.LambdaPrime = clamp(self.LambdaPrime, -1.0, 1.0)

        # probe impulse
        if self.t>0 and self.t % PROBE_PERIOD == 0:
            self.LambdaPrime += np.random.choice([+PROBE_MAG, -PROBE_MAG])
            self.OmegaPrime  += np.random.choice([+PROBE_MAG, -PROBE_MAG])

        # log
        self.h_omega.append(self.OmegaPrime)
        self.h_lambda.append(self.LambdaPrime)
        self.h_dphi.append(dphi)
        self.h_reg.append(regime)

        self.t += 1
        return regime

    def verdict(self):
        # доля режимов
        drift = self.h_reg.count("Drift")/len(self.h_reg)
        lock  = self.h_reg.count("Lock") /len(self.h_reg)
        itrt  = self.h_reg.count("Iterate")/len(self.h_reg)

        var_om = rolling_var(self.h_omega, WIN_VAR)
        # скорость «дыхания»
        dlam = np.diff(np.array(self.h_lambda)) if len(self.h_lambda)>1 else np.array([0.0])
        mean_abs_dlam = float(np.mean(np.abs(dlam))) if dlam.size>0 else 0.0

        corr = rolling_corr(self.h_omega, self.h_lambda, WIN_CORR)

        alive = (drift>=0.45 and drift<=0.75 and
                 (var_om is np.nan or var_om < 1e-3) and
                 mean_abs_dlam > 1e-3 and
                 (corr is not np.nan and corr >= 0.25))

        self.alive = alive
        return {
            "drift": drift, "lock": lock, "iterate": itrt,
            "var_omega_win": var_om,
            "mean_abs_dLambda": mean_abs_dlam,
            "corr_dO_dL": corr,
            "alive": alive
        }

def run_and_log(csv_path="ace_log_v04d.csv", seed=42):
    np.random.seed(seed)
    ace = ACEngine()
    regime = "Drift"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","OmegaPrime","LambdaPrime","dphi","regime"])
        for t in range(STEPS):
            # dphi: дыхательная смесь синуса + шума
            dphi = EPSILON + 0.22*np.sin(0.007*t) + np.random.normal(0, 0.06)
            regime = ace.step(dphi, regime)
            w.writerow([t, ace.OmegaPrime, ace.LambdaPrime, dphi, regime])
    return ace

if __name__ == "__main__":
    os.makedirs("ace_v04d_report", exist_ok=True)
    ace = run_and_log()
    v = ace.verdict()
    print("=== ACE v0.4d — run summary ===")
    print(f"steps: {STEPS}")
    print(f"regime_share: {{'Drift': {v['drift']:.3f}, 'Lock': {v['lock']:.3f}, 'Iter': {v['iterate']:.3f}}}")
    print(f"omega_var_win: {v['var_omega_win']}")
    print(f"corr_last: {v['corr_dO_dL']}")
    print(f"alive: {v['alive']}")
