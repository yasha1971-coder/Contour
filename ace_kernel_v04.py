# ACE v0.4 — Coherent Breathing
# Ω′↔Λ′ coupling + soft memory (exponential decay) + hysteresis + ξ monitor
# Outputs: ace_log_v04.csv + console summary

import numpy as np
import csv, time
from collections import deque

# -----------------------------
# Tunables (v0.4 defaults)
# -----------------------------
EPSILON          = 0.74   # field center
DELTA            = 0.10   # viability half-width
ETA              = 0.012  # transform step (tempo)
NOISE            = 0.06   # external perturbation intensity

COUP_OM_TO_LA    = 0.08   # Ω′ → Λ′
COUP_LA_TO_OM    = 0.06   # Λ′ → Ω′
MEM_INERTIA      = 0.15   # strength of rhythmic persistence
MEM_DECAY        = 0.97   # exponential decay of memory each tick (key!)
ELASTIC_OMEGA    = 0.04   # ± window where Ω′ can “breathe” around its center (≈1.0)

WIN              = 60     # rolling window for var(Ω′), ξ, corr
STEPS            = 8000   # ticks per run
DIM              = 16     # latent state dimension (kept simple for now)

# success criteria (v0.4)
DRIFT_TARGET     = 0.60
OMEGA_VAR_MAX    = 1e-2
LAMBDA_SLOPE_MIN = 1e-2
COUPLING_TARGET  = 0.25   # corr(dΩ′, dΛ′) threshold

# -----------------------------
# helpers
# -----------------------------
def safe_var(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2: return 0.0
    m = np.mean(x)
    return float(np.mean((x - m)**2))

def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if n < 3: return 0.0
    x, y = x[-n:], y[-n:]
    sx = np.std(x); sy = np.std(y)
    if sx == 0 or sy == 0: return 0.0
    return float(np.corrcoef(x, y)[0,1])

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

# -----------------------------
# Engine
# -----------------------------
class ACEngineV4:
    def __init__(self):
        self.epsilon = EPSILON
        self.delta   = DELTA
        self.eta     = ETA
        self.noise   = NOISE

        # state variables (center Ω′ ≈ 1.0)
        self.OmegaPrime  = 1.0
        self.LambdaPrime = 0.14

        # memory buffers
        self.mem_omega = 0.0
        self.mem_lambda = 0.0
        self.phase_memory = deque(maxlen=50)

        # histories
        self.h_omega  = deque(maxlen=WIN)
        self.h_lambda = deque(maxlen=WIN)
        self.h_domega = deque(maxlen=WIN)
        self.h_dlambda= deque(maxlen=WIN)

        # regime & hysteresis
        self.regime = "Drift"
        self.t = 0

        # metrics
        self.transitions = 0
        self.longest_drift = 0
        self._cur_drift_len = 0

    # classify with hysteresis (sticky drift)
    def classify_regime(self, dphi):
        low, high = self.epsilon - self.delta, self.epsilon + self.delta
        if self.regime == "Drift":
            low  -= 0.02
            high += 0.02
        if dphi < low:  return "Lock"
        if dphi > high: return "Iter"
        return "Drift"

    # elastic center for Ω′
    def elastic_omega(self, new):
        center = 1.0
        if abs(new - center) <= ELASTIC_OMEGA:
            return new
        # gentle restoration
        return center + (new - center) * 0.7

    def step(self, dphi):
        self.t += 1

        # regime
        new_regime = self.classify_regime(dphi)
        if new_regime != self.regime:
            self.transitions += 1
            self.regime = new_regime

        # external perturb
        env = self.noise * (np.random.randn() * 0.5 + 0.5*np.sin(0.017*self.t))

        # reciprocal coupling
        # memory (decay then add latest)
        self.mem_omega  = self.mem_omega  * MEM_DECAY + (self.OmegaPrime  - 1.0)
        self.mem_lambda = self.mem_lambda * MEM_DECAY + (self.LambdaPrime - 0.14)

        # base proposals
        dOmega  = 0.0
        dLambda = 0.0

        if self.regime == "Drift":
            # coherent breathing: small stochastic exploration + memory rhythm
            dOmega  += 0.005*np.sin(0.031*self.t) + 0.004*np.random.randn()
            dLambda += 0.010*np.sin(0.027*self.t + 0.6) + 0.006*np.random.randn()
            # memory nudges (proto-nervous system)
            dOmega  += MEM_INERTIA * self.mem_lambda
            dLambda += MEM_INERTIA * self.mem_omega
        elif self.regime == "Lock":
            # crystallize slightly but keep breathing
            dOmega  -= 0.006*(self.OmegaPrime - 1.0)
            dLambda *= 0.5
        else:  # Iterate
            # larger exploration to re-find drift
            dOmega  += 0.020*np.random.randn()
            dLambda += 0.020*np.random.randn()

        # reciprocal influence
        dOmega  += COUP_LA_TO_OM * (self.LambdaPrime - 0.14)
        dLambda += COUP_OM_TO_LA * (self.OmegaPrime  - 1.0)

        # environment + tempo
        dOmega  += env
        dLambda += env * 0.6
        dOmega  *= self.eta / 0.01
        dLambda *= self.eta / 0.01

        # apply
        newOmega  = self.elastic_omega(self.OmegaPrime + dOmega)
        newLambda = self.LambdaPrime + dLambda

        # record deltas for coupling metrics
        self.h_domega.append(newOmega - self.OmegaPrime)
        self.h_dlambda.append(newLambda - self.LambdaPrime)

        self.OmegaPrime, self.LambdaPrime = newOmega, newLambda
        self.h_omega.append(self.OmegaPrime)
        self.h_lambda.append(self.LambdaPrime)

        # drift streak
        if self.regime == "Drift":
            self._cur_drift_len += 1
            if self._cur_drift_len > self.longest_drift:
                self.longest_drift = self._cur_drift_len
        else:
            self._cur_drift_len = 0

        # ξ proxy: rolling corr of derivatives (non-negative)
        xi = max(0.0, corr(np.array(self.h_domega), np.array(self.h_dlambda)))

        return {
            "Omega_prime": self.OmegaPrime,
            "Lambda_prime": self.LambdaPrime,
            "regime": self.regime,
            "xi": xi
        }

# -----------------------------
# run + log
# -----------------------------
def main():
    eng = ACEngineV4()
    regimes = {"Drift":0,"Lock":0,"Iter":0}

    log_rows = []
    for t in range(STEPS):
        # drive field (ε-centered with noise)
        dphi = EPSILON + 0.08*np.sin(0.013*t) + 0.08*np.sin(0.021*t+1.2) + 0.18*np.random.randn()*0.02
        m = eng.step(dphi)
        regimes[m["regime"]] += 1
        log_rows.append([t, m["Omega_prime"], m["Lambda_prime"], m["regime"], m["xi"]])

    # write CSV
    with open("ace_log_v04.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["t","Omega_prime","Lambda_prime","regime","xi"])
        w.writerows(log_rows)

    # metrics
    drift_share = regimes["Drift"]/STEPS
    omega_var   = safe_var(list(eng.h_omega))
    mean_abs_dL = float(np.mean(np.abs(np.array(eng.h_dlambda)))) if len(eng.h_dlambda)>0 else 0.0
    coupling    = corr(np.array(eng.h_domega), np.array(eng.h_dlambda))

    alive = (drift_share >= DRIFT_TARGET and
             omega_var <= OMEGA_VAR_MAX and
             mean_abs_dL >= LAMBDA_SLOPE_MIN and
             coupling >= COUPLING_TARGET)

    # summary
    print("\n================ ACE v0.4 — run summary ================")
    print(f"Verdict ALIVE : {alive}")
    print("--------------------------------------------------------")
    print(f"Drift share      : {drift_share*100:5.1f}%")
    print(f"var_win(Ω′)      : {omega_var:.6e}")
    print(f"mean |dΛ′/dt|    : {mean_abs_dL:.6f}")
    print(f"corr(dΩ′, dΛ′)   : {coupling:.4f}")
    print(f"Transitions      : {eng.transitions}")
    print(f"Longest Drift    : {eng.longest_drift}")
    print("Log              : ace_log_v04.csv")
    print("========================================================\n")

if __name__ == "__main__":
    main()
