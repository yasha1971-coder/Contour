# ace_kernel_v04e.py
# ACE v0.4e (fix) — minimal kernel with Λ′-gain + anti-stall bump
# Saves: ace_v04e_report/data.csv
# Returns: dict(history=..., meta=...)

from __future__ import annotations
import os, csv, math, json
from collections import deque
import numpy as np

# ---------- helpers ----------
def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def rolling_variance(win: deque) -> float:
    """Numerically stable rolling variance of the Omega window."""
    if not win:
        return 0.0
    arr = np.fromiter(win, dtype=float)
    m = arr.mean()
    return float(np.mean((arr - m) ** 2))

# ---------- core engine ----------
class ACEngineV04e:
    """
    Ω′ (omega) — invariant flux proxy
    Λ′ (lam)   — rhythm continuity proxy
    dΩ′, dΛ′   — per-step deltas (post-hysteresis)
    """
    def __init__(self, params: dict, steps: int = 6000, seed: int = 42):
        self.p = params.copy()
        self.steps = int(steps)
        self.rng = np.random.default_rng(seed)

        # core params (with safe defaults)
        self.c_ol = float(self.p.get("COUP_OM_TO_LA", 0.10))
        self.c_lo = float(self.p.get("COUP_LA_TO_OM", 0.12))
        self.mem   = float(self.p.get("MEM_DECAY", 0.80))
        self.hyst  = float(self.p.get("HYST", 0.010))
        self.noise = float(self.p.get("NOISE", 0.006))

        # v04e-fix additions
        self.var_win = int(self.p.get("VAR_WINDOW", 220))
        self.bump    = float(self.p.get("ANTI_STALL_BUMP", 0.012))
        self.l_gain  = float(self.p.get("L_GAIN", 1.35))
        self.drift_hyst = float(self.p.get("DRIFT_HYST", 0.022))  # для классификации (мягко)

        # state
        self.omega = 1.0 + 0.01 * self.rng.standard_normal()
        self.lam   = 0.98 + 0.01 * self.rng.standard_normal()

        self.win_omega = deque(maxlen=self.var_win)

        # logging
        self.t_hist, self.o_hist, self.l_hist = [], [], []
        self.do_hist, self.dl_hist = [], []
        self.regime_hist = []  # 0 drift, 1 lock, 2 iterate
        self.bump_count = 0
        self.last_bump_t = -1

    # --- one step ---
    def step(self, t: int):
        o, l = self.omega, self.lam

        # coupling dynamics + noise
        dO_raw = self.c_lo * (l - o) + self.noise * self.rng.standard_normal()
        dL_raw = self.c_ol * (o - l) + self.noise * self.rng.standard_normal()

        # hysteresis (deadband)
        dO = 0.0 if abs(dO_raw) < self.hyst else dO_raw
        dL = 0.0 if abs(dL_raw) < self.hyst else dL_raw

        # Λ′ gain (v04e-fix): усилить дыхание ритма, не ломая Ω′
        dL *= self.l_gain

        # memory decay (инерция поля)
        o = (1.0 - self.mem) * (o + dO) + self.mem * o
        l = (1.0 - self.mem) * (l + dL) + self.mem * l

        # анти-столл: если Λ′ почти стоит, а Ω′ застыл — мягкий пинок
        self.win_omega.append(o)
        var_o = rolling_variance(self.win_omega)
        if abs(dL) < 1e-5 and var_o < 1e-8:
            kick = self.bump if self.rng.random() > 0.5 else -self.bump
            l += kick
            dL = (1.0 - self.mem) * kick
            self.bump_count += 1
            self.last_bump_t = t

        # commit
        self.omega, self.lam = o, l

        # regime (мягкая классификация по расхождению и активности)
        eps = abs(o - l)
        if eps < self.hyst and abs(dL) < self.drift_hyst:
            regime = 1  # lock
        elif eps > 3.0 * self.hyst or abs(dL) > self.drift_hyst:
            regime = 0  # drift
        else:
            regime = 2  # iterate (редко, но оставим)

        # log
        self.t_hist.append(t)
        self.o_hist.append(o)
        self.l_hist.append(l)
        self.do_hist.append(dO)
        self.dl_hist.append(dL)
        self.regime_hist.append(regime)

    # --- run ---
    def run(self):
        for t in range(self.steps):
            self.step(t)

        return {
            "history": {
                "t": np.array(self.t_hist),
                "omega": np.array(self.o_hist),
                "lambda": np.array(self.l_hist),
                "domega": np.array(self.do_hist),
                "dlambda": np.array(self.dl_hist),
                "regime": np.array(self.regime_hist, dtype=int),
            },
            "meta": {
                "bumps": int(self.bump_count),
                "last_bump_t": int(self.last_bump_t),
                "var_window": self.var_win,
            }
        }

# ---------- public API used by main.py ----------
def run_kernel(params: dict, steps: int = 6000, seed: int = 42, out_dir: str = "ace_v04e_report"):
    """
    Runs ACE v0.4e-fix and writes CSV to ace_v04e_report/data.csv.
    Returns (history, meta) for the caller (main.py).
    """
    eng = ACEngineV04e(params, steps=steps, seed=seed)
    result = eng.run()

    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "omega", "lambda", "domega", "dlambda", "regime"])
        for i in range(len(result["history"]["t"])):
            w.writerow([
                int(result["history"]["t"][i]),
                float(result["history"]["omega"][i]),
                float(result["history"]["lambda"][i]),
                float(result["history"]["domega"][i]),
                float(result["history"]["dlambda"][i]),
                int(result["history"]["regime"][i]),
            ])

    return result
