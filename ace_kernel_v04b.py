# ace_kernel_v04b.py
# Algorithm-Contour Engine — v0.4b (stabilized breathing + soft coupling)
# Goals:
#  - eliminate NaNs / blow-ups via clamping & normalization
#  - keep Ω′ “elastic” (not frozen), Λ′ “breathing”
#  - increase Drift persistence with hysteresis around thresholds
#  - introduce Ω′↔Λ′ soft bidirectional coupling + short-term memory
#  - log metrics suitable for Ξ-chart

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math
import random
import csv
import os
from typing import Deque, Dict, Any, Optional

# ---------- helpers ----------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def ema(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def safe_var(window):
    n = len(window)
    if n <= 1:
        return 0.0
    mu = sum(window) / n
    return sum((x - mu) ** 2 for x in window) / n

def safe_corr(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xm = sum(xs[-n:]) / n
    ym = sum(ys[-n:]) / n
    vx = sum((x - xm)**2 for x in xs[-n:]) / n
    vy = sum((y - ym)**2 for y in ys[-n:]) / n
    if vx <= 1e-16 or vy <= 1e-16:
        return 0.0
    cov = sum((xs[-n+i] - xm) * (ys[-n+i] - ym) for i in range(n)) / n
    return clamp(cov / math.sqrt(vx * vy), -1.0, 1.0)

# ---------- params ----------

@dataclass
class ACEParams:
    # regime thresholds
    eps: float = 0.74          # ε — baseline field threshold
    delta: float = 0.10        # δ — viability margin
    hysteresis: float = 0.02   # widen drift when already in drift

    # dynamics
    base_rhythm: float = 0.14  # target Λ′ scale
    eta: float = 0.012         # transformation/persistence
    noise_scale: float = 0.06  # external perturbation magnitude (Δφ)
    adv_spike_prob: float = 0.01
    adv_spike_scale: float = 1.8  # ≥ 1.5×ε spikes

    # coupling (soft, bounded)
    lambda_to_omega: float = 0.06   # Λ′ → Ω′ influence
    omega_to_lambda: float = 0.08   # Ω′ → Λ′ influence
    coupling_cap: float = 0.08      # cap absolute coupling injection per step

    # elasticity and memory
    omega_center: float = 1.0
    omega_elastic_window: float = 0.04  # ± window accepted as “natural”
    omega_restore: float = 0.7          # pullback factor if outside window
    mem_alpha_omega: float = 0.2        # EMA for Ω′
    mem_alpha_lambda: float = 0.25      # EMA for Λ′
    drift_inertia: float = 0.12         # reinforces recent breathing in Drift
    memory_window: int = 60

    # evaluation windows / limits
    var_window: int = 120
    corr_window: int = 200
    alive_var_thresh: float = 1e-2
    alive_lambda_slope_min: float = 1e-2

    # guards
    klein_guard: bool = True
    tau_min: float = 0.012
    seed: Optional[int] = None

class ACEngine:
    """
    ACE v0.4b — stabilized single-contour engine

    Regimes:
      Lock   : dphi < eps - delta
      Drift  : eps - delta <= dphi <= eps + delta  (with hysteresis if already in Drift)
      Iterate: dphi > eps + delta

    Life (operational):
      drift_share ≥ ~0.6
      var_win(Ω′) < 1e-2
      mean|ΔΛ′|/dt > 1e-2
      corr(dΩ′, dΛ′) > ~0.25  (target for coherent breathing)
    """

    def __init__(self, params: ACEParams = ACEParams()):
        self.p = params
        if self.p.seed is not None:
            random.seed(self.p.seed)

        # state
        self.t = 0
        self.regime = "Drift"
        self.Omega = 1.0           # Ω′ circulation
        self.Lambda = self.p.base_rhythm  # Λ′ axial rhythm
        self.delta = self.p.delta  # adaptive δ (can drift slightly)
        self.tau = 0.04            # transformation parameter (placeholder)

        # smoothed
        self.Omega_s = self.Omega
        self.Lambda_s = self.Lambda

        # histories
        self.h_omega: Deque[float] = deque(maxlen=max(self.p.var_window, self.p.corr_window))
        self.h_lambda: Deque[float] = deque(maxlen=max(self.p.var_window, self.p.corr_window))
        self.h_domega: Deque[float] = deque(maxlen=self.p.corr_window)
        self.h_dlambda: Deque[float] = deque(maxlen=self.p.corr_window)
        self.h_regime: Deque[str] = deque(maxlen=2000)

        # counters
        self.transitions = 0
        self.last_regime = self.regime
        self.locked_steps = 0
        self.drift_steps = 0
        self.iter_steps = 0

        # logging
        self.csv_path = "ace_log_v04b.csv"
        self._prepare_csv()

    # --------- CSV ---------

    def _prepare_csv(self):
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "t", "Omega", "Lambda", "delta", "dphi", "regime",
                "Omega_s", "Lambda_s", "tau",
                "var_omega_win", "corr_domega_dlambda", "alive"
            ])

    def _append_csv(self, row: Dict[str, Any]):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                row["t"], row["Omega"], row["Lambda"], row["delta"], row["dphi"], row["regime"],
                row["Omega_s"], row["Lambda_s"], row["tau"],
                row["var_omega_win"], row["corr_domega_dlambda"], row["alive"]
            ])

    # --------- signals & regimes ---------

    def _dphi(self) -> float:
        # baseline breathing + noise + occasional adversarial spike
        base = 0.5 * (1 + math.sin(2 * math.pi * (self.t / 100.0)))
        noise = random.gauss(0.0, self.p.noise_scale)
        dphi = max(0.0, base + noise)

        if random.random() < self.p.adv_spike_prob:
            dphi = max(dphi, self.p.adv_spike_scale * self.p.eps)

        return dphi

    def _classify_regime(self, dphi: float) -> str:
        eps = self.p.eps
        low = eps - self.delta
        high = eps + self.delta

        # hysteresis: if already in Drift, require stronger push to exit
        if self.regime == "Drift":
            low -= self.p.hysteresis
            high += self.p.hysteresis

        if dphi < low:
            return "Lock"
        elif dphi > high:
            return "Iterate"
        else:
            return "Drift"

    # --------- dynamics ---------

    def _update_lambda(self, regime: str):
        # Λ′ breathing — regime-dependent frequency & amplitude
        # bounded sinusoid + small exploration
        base = self.p.base_rhythm
        if regime == "Lock":
            amp = 0.6
            freq = 2 * math.pi * (self.t / 55.0)
        elif regime == "Drift":
            amp = 1.0
            freq = 2 * math.pi * (self.t / 38.0)
        else:  # Iterate
            amp = 1.3
            freq = 2 * math.pi * (self.t / 26.0)

        target = amp * base * math.sin(freq)
        # small exploration noise
        target += random.gauss(0.0, 0.02 * base)

        # soft memory / inertia in Drift (reinforce natural rhythm)
        if regime == "Drift":
            target += self.p.drift_inertia * (self.Lambda_s - self.Lambda)

        # Ω′ → Λ′ coupling (soft, bounded)
        omega_deviation = self.Omega_s - self.p.omega_center
        coup = self.p.omega_to_lambda * omega_deviation
        coup = clamp(coup, -self.p.coupling_cap, self.p.coupling_cap)
        target += coup

        # update & clamp
        self.Lambda = clamp(target, -5.0, 5.0)

    def _update_omega(self, regime: str, dphi: float):
        # Ω′ elastic constancy: encourage near-const but allow breathing
        # higher target when dphi near eps; lower when far
        proximity = 1.0 - min(1.0, abs(dphi - self.p.eps))
        target = self.p.omega_center - 0.02 + 0.04 * proximity  # ~ [0.98, 1.02]

        # Λ′ → Ω′ coupling
        lambda_push = self.p.lambda_to_omega * math.tanh(self.Lambda_s)
        lambda_push = clamp(lambda_push, -self.p.coupling_cap, self.p.coupling_cap)
        target += lambda_push

        # small endogenous fluctuation (prevents absolute freeze)
        target += random.gauss(0.0, 0.003)

        # elasticity window (soft pullback)
        dev = target - self.p.omega_center
        if abs(dev) > self.p.omega_elastic_window:
            target = self.p.omega_center + dev * self.p.omega_restore

        # update & clamp to safe envelope
        self.Omega = clamp(target, 0.85, 1.15)

    # --------- life check ---------

    def _omega_is_constant(self) -> bool:
        if len(self.h_omega) < self.p.var_window:
            return False
        varw = safe_var(list(self.h_omega)[-self.p.var_window:])
        return varw < self.p.alive_var_thresh

    def _lambda_vital(self) -> bool:
        if len(self.h_lambda) < 3:
            return True
        last = list(self.h_lambda)[-3:]
        slope = abs(last[-1] - last[0])
        return slope > self.p.alive_lambda_slope_min

    def _corr_dynamics(self) -> float:
        if len(self.h_domega) < self.p.corr_window or len(self.h_dlambda) < self.p.corr_window:
            return 0.0
        return safe_corr(list(self.h_domega), list(self.h_dlambda))

    def is_alive(self) -> bool:
        return self._omega_is_constant() and self._lambda_vital()

    # --------- step ---------

    def step(self):
        self.t += 1

        # transform param (placeholder; guards against too small τ)
        self.tau = max(self.tau + random.gauss(0.0, 0.001), self.p.tau_min)

        # field input
        dphi = self._dphi()

        # classify + Klein guard widening near boundary
        new_regime = self._classify_regime(dphi)
        if self.p.klein_guard and abs(dphi - self.p.eps) < (self.delta * 0.5):
            new_regime = "Drift"

        # count transitions
        if new_regime != self.regime:
            self.transitions += 1
            self.regime = new_regime

        # regime counters
        if self.regime == "Lock":
            self.locked_steps += 1
        elif self.regime == "Drift":
            self.drift_steps += 1
        else:
            self.iter_steps += 1

        # update dynamics
        self._update_lambda(self.regime)
        self._update_omega(self.regime, dphi)

        # smooth states
        self.Omega_s = ema(self.Omega_s, self.Omega, self.p.mem_alpha_omega)
        self.Lambda_s = ema(self.Lambda_s, self.Lambda, self.p.mem_alpha_lambda)

        # history (for var & corr)
        if self.h_omega:
            self.h_domega.append(self.Omega_s - self.h_omega[-1])
        if self.h_lambda:
            self.h_dlambda.append(self.Lambda_s - self.h_lambda[-1])
        self.h_omega.append(self.Omega_s)
        self.h_lambda.append(self.Lambda_s)
        self.h_regime.append(self.regime)

        # adaptive delta micro-tuning (keep δ > 0, bounded)
        self.delta = clamp(self.delta + 0.01 * abs(dphi - self.p.eps) - 0.005, 0.06, 0.18)

        # metrics
        var_win = safe_var(list(self.h_omega)[-self.p.var_window:]) if len(self.h_omega) >= self.p.var_window else 1.0
        corr = self._corr_dynamics()
        alive = self.is_alive()

        # log
        row = {
            "t": self.t,
            "Omega": round(self.Omega, 6),
            "Lambda": round(self.Lambda, 6),
            "delta": round(self.delta, 6),
            "dphi": round(dphi, 6),
            "regime": self.regime,
            "Omega_s": round(self.Omega_s, 6),
            "Lambda_s": round(self.Lambda_s, 6),
            "tau": round(self.tau, 6),
            "var_omega_win": round(var_win, 10),
            "corr_domega_dlambda": round(corr, 6),
            "alive": alive
        }
        self._append_csv(row)
        return row

# --------- runnable demo ---------

if __name__ == "__main__":
    eng = ACEngine(ACEParams(seed=42))
    T = 6000  # ticks
    for _ in range(T):
        eng.step()

    drift_share = eng.drift_steps / max(1, (eng.locked_steps + eng.drift_steps + eng.iter_steps))
    print("ACE v0.4b run complete")
    print(f"Transitions: {eng.transitions}")
    print(f"Regime share — Drift: {drift_share:.3f}, Lock: {eng.locked_steps/T:.3f}, Iterate: {eng.iter_steps/T:.3f}")
    print(f"Alive? {eng.is_alive()}")
    print(f"Log: {os.path.abspath(eng.csv_path)}")
