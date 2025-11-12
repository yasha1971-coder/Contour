# ACE Engine v0.4e-fix — kernel (metrics fixed, hooks added)
# Author: ACE project

from __future__ import annotations
import os, json, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

# -----------------------
# Utilities
# -----------------------

def soft_clip(x: float, limit: float = 5.0) -> float:
    # smooth limiter to prevent numerical blowups without hard cut
    return float(np.tanh(x / limit) * limit)

# -----------------------
# Params
# -----------------------

@dataclass
class ACEParams:
    COUP_LA_TO_OM: float = 0.18
    COUP_OM_TO_LA: float = 0.20
    MEM_DECAY: float = 0.38
    HYST: float = 0.0065
    NOISE: float = 0.012
    VAR_WINDOW: int = 300
    DRIFT_HYST: float = 0.022
    ANTI_STALL_BUMP: float = 0.012
    L_GAIN: float = 1.3

    # hooks
    USE_NUDGES: bool = False
    USE_ENV_ADAPTER: bool = False
    NUDGE_GAIN: float = 0.0
    ENV_GAIN: float = 0.0
    ENTROPY_WINDOW: int = 50

    @staticmethod
    def from_dict(d: Dict) -> "ACEParams":
        p = ACEParams()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

# -----------------------
# Engine
# -----------------------

@dataclass
class ACEState:
    t: int = 0
    omega_p: float = 0.0   # Ω′
    lambda_p: float = 0.0  # Λ′
    hist: Dict[str, List[float]] = field(default_factory=lambda: {
        "omega_p": [], "lambda_p": []
    })

class ACEEngine:
    def __init__(self, params: ACEParams, rng_seed: int = 1337):
        self.p = params
        self.s = ACEState()
        self.rng = np.random.default_rng(rng_seed)

        # hooks (user can assign callables)
        self.hook_pre_step = None    # def f(engine)->None
        self.hook_post_step = None   # def f(engine)->None
        self.nudge_fn = None         # def f(step)->float
        self.env_adapter_fn = None   # def f(engine)->float

        # mini-регулятор v0.5: авто-подстройка NOISE/MEM_DECAY к целевому окну
        self.var_target_lo, self.var_target_hi = 1e-6, 1e-3

    # --------- core dynamics ---------

    def step(self, dt: float = 1.0):
        p, s, r = self.p, self.s, self.rng

        if self.hook_pre_step:
            self.hook_pre_step(self)

        # базовая стохастика / внешние возмущения
        noise = p.NOISE * r.standard_normal()

        # легкие nudges (подталкивания)
        if p.USE_NUDGES and self.nudge_fn is not None:
            noise += p.NUDGE_GAIN * float(self.nudge_fn(s.t))

        # адаптер внешней среды (например, энтропия входного потока)
        if p.USE_ENV_ADAPTER and self.env_adapter_fn is not None:
            noise += p.ENV_GAIN * float(self.env_adapter_fn(self))

        # нелинейная связь (двунаправленная)
        domega = -p.HYST * s.omega_p + p.COUP_LA_TO_OM * np.tanh(s.lambda_p) + noise
        dlambda = -p.MEM_DECAY * s.lambda_p + p.COUP_OM_TO_LA * np.tanh(s.omega_p)
        dlambda *= p.L_GAIN

        # интеграция
        s.omega_p += dt * domega
        s.lambda_p += dt * dlambda

        # мягкий клиппинг для устойчивости
        s.omega_p = soft_clip(s.omega_p, 5.0)
        s.lambda_p = soft_clip(s.lambda_p, 5.0)

        # журнал
        s.t += 1
        s.hist["omega_p"].append(s.omega_p)
        s.hist["lambda_p"].append(s.lambda_p)

        if self.hook_post_step:
            self.hook_post_step(self)

    # --------- metrics (fixed) ---------

    @staticmethod
    def _window(arr: List[float], n: int) -> np.ndarray:
        if len(arr) == 0:
            return np.asarray([], dtype=float)
        return np.asarray(arr[-n:], dtype=float)

    def compute_metrics(self, dt: float = 1.0) -> Dict[str, float]:
        n = int(self.p.VAR_WINDOW)
        op = self._window(self.s.hist["omega_p"], n)
        lp = self._window(self.s.hist["lambda_p"], n)

        if op.size < 4 or lp.size < 4:
            # недостаточно данных
            return {
                "var_win": np.nan,
                "mean_abs_dlambda_dt": np.nan,
                "corr_domega_dlambda": 0.0,
                "drift_share": 0.0,
                "lock_share": 0.0,
                "iterate_share": 100.0,
                "regime_transitions_per_1k": 0
            }

        var_win = float(np.var(op - op.mean(), ddof=0))
        domega_dt = np.diff(op, prepend=op[0]) / float(dt)
        dlambda_dt = np.diff(lp, prepend=lp[0]) / float(dt)
        mean_abs_dlambda_dt = float(np.mean(np.abs(dlambda_dt)))

        if np.std(domega_dt) > 0 and np.std(dlambda_dt) > 0:
            corr = float(np.corrcoef(domega_dt, dlambda_dt)[0, 1])
        else:
            corr = 0.0

        LOCK_THR = 1e-4
        DRIFT_THR = 1e-2

        abs_d = np.abs(dlambda_dt)
        lock_mask = abs_d < LOCK_THR
        drift_mask = abs_d >= DRIFT_THR
        iterate_mask = ~lock_mask & ~drift_mask

        npts = abs_d.size
        lock_share = 100.0 * lock_mask.sum() / npts
        drift_share = 100.0 * drift_mask.sum() / npts
        iterate_share = 100.0 * iterate_mask.sum() / npts

        regime_idx = np.where(lock_mask, 0, np.where(iterate_mask, 1, 2))
        transitions = int(np.sum(regime_idx[1:] != regime_idx[:-1]))
        transitions_per_1k = int(round(1000.0 * transitions / max(1, npts - 1)))

        return {
            "var_win": var_win,
            "mean_abs_dlambda_dt": mean_abs_dlambda_dt,
            "corr_domega_dlambda": corr,
            "drift_share": drift_share,
            "lock_share": lock_share,
            "iterate_share": iterate_share,
            "regime_transitions_per_1k": transitions_per_1k
        }

    # --------- alive test ---------

    @staticmethod
    def alive_verdict(metrics: Dict[str, float]) -> Dict[str, object]:
        v = metrics.get("var_win")
        m = metrics.get("mean_abs_dlambda_dt")
        d = metrics.get("drift_share", 0.0)

        ok_var = (v is not None) and (not math.isnan(v)) and (1e-6 <= v <= 1e-3)
        ok_vel = (m is not None) and (not math.isnan(m)) and (m >= 1e-2)
        ok_drift = d >= 20.0

        verdict = "ALIVE" if (ok_var and ok_vel and ok_drift) else "NOT ALIVE"
        return {
            "ok_var": ok_var, "ok_vel": ok_vel, "ok_drift": ok_drift,
            "verdict": verdict
        }

    # --------- self-goal loop v0.5 ---------

    def self_goal_adjust(self, metrics: Dict[str, float], k_noise=0.25, k_decay=0.15):
        # поддержка дыхания: вар-окно в целевом диапазоне
        v = metrics.get("var_win")
        if v is None or math.isnan(v):
            return
        if v < self.var_target_lo:
            # слишком «заморожено» → добавим шума, уменьшим затухание памяти
            self.p.NOISE = max(0.001, self.p.NOISE * (1.0 + k_noise))
            self.p.MEM_DECAY = max(0.05, self.p.MEM_DECAY * (1.0 - k_decay))
        elif v > self.var_target_hi:
            # слишком хаотично → уменьшим шум, усилим затухание
            self.p.NOISE = max(0.001, self.p.NOISE * (1.0 - k_noise))
            self.p.MEM_DECAY = min(0.95, self.p.MEM_DECAY * (1.0 + k_decay))

    # --------- run ---------

    def run(self, steps: int = 6000, dt: float = 1.0, use_regulator: bool = True) -> Dict[str, object]:
        for _ in range(int(steps)):
            self.step(dt=dt)
            # мини-регулятор
            if use_regulator and self.s.t > self.p.VAR_WINDOW:
                metrics = self.compute_metrics(dt=dt)
                self.self_goal_adjust(metrics)

        metrics = self.compute_metrics(dt=dt)
        verdict = self.alive_verdict(metrics)
        return {
            "metrics": metrics,
            "verdict": verdict,
            "params": self.p.__dict__
        }

# -------------- helpers --------------

def load_params_json(path: Optional[str]) -> ACEParams:
    if not path:
        return ACEParams()
    with open(path, "r") as f:
        return ACEParams.from_dict(json.load(f))
