# meta_core_v05a.py
# ACE Meta-Core v0.5a: self-goal loop for NOISE/MEM_DECAY with psi-control
# Requires: numpy, matplotlib (optional for plots)

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
import time
import numpy as np

# ---------- helpers

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ema(prev: float, x: float, alpha: float) -> float:
    return alpha * x + (1 - alpha) * prev

# ---------- config & state

@dataclass
class MetaTargets:
    # target windows for "living" corridor
    var_lo: float = 1e-6
    var_hi: float = 1e-3
    v_min: float = 1e-2               # |dΛ′/dt|
    drift_min: float = 0.20           # ≥ 20%
    psi_transitions_per_1k: float = 300.0  # desired regime transitions per 1k steps

@dataclass
class MetaGains:
    # PI-like controller gains
    k_p_var: float = 0.50
    k_i_var: float = 0.02

    k_p_vel: float = 0.40
    k_i_vel: float = 0.02

    k_p_psi: float = 0.25
    k_i_psi: float = 0.01

    # soft coupling from NOISE to velocity and MEM_DECAY to variance
    # (signs matter: ↑noise -> ↑velocity & ↑variance; ↑mem_decay -> ↓variance, иногда ↓velocity)
    noise_step: float = 0.0020
    mem_step: float  = 0.0200

@dataclass
class MetaBounds:
    noise_min: float = 0.004
    noise_max: float = 0.050
    mem_min: float   = 0.10
    mem_max: float   = 0.65

@dataclass
class MetaPulse:
    # external perturbations channel
    enable: bool = True
    period: int = 800                 # steps between pulse starts
    duration: int = 120               # steps in pulse
    amplitude: float = 0.35           # relative increase of NOISE during pulse (multiplier)
    warmup: int = 600                 # do not pulse before this step

    def factor(self, t: int) -> float:
        if not self.enable or t < self.warmup:
            return 1.0
        phase = t % self.period
        if phase < self.duration:
            # smooth bell-shaped envelope inside the pulse
            x = (phase / max(1, self.duration))
            envelope = 0.5 - 0.5 * math.cos(2 * math.pi * x)  # 0..1..0
            return 1.0 + self.amplitude * envelope
        return 1.0

@dataclass
class MetaLogs:
    t: list[int] = field(default_factory=list)
    var_win: list[float] = field(default_factory=list)
    v_mean: list[float] = field(default_factory=list)
    drift_share: list[float] = field(default_factory=list)
    transitions_per_1k: list[float] = field(default_factory=list)
    psi_err: list[float] = field(default_factory=list)
    noise: list[float] = field(default_factory=list)
    mem_decay: list[float] = field(default_factory=list)

# ---------- main meta-core

class MetaCore:
    """
    Self-goal loop:
      - holds variance window inside [var_lo, var_hi]
      - keeps mean |dΛ′/dt| >= v_min
      - keeps drift_share >= drift_min
      - aims psi = desired regime transitions per 1k steps
    Produces gentle updates to NOISE & MEM_DECAY each control tick.
    """
    def __init__(
        self,
        out_dir: str | Path = "ace_v04e_report",
        targets: MetaTargets = MetaTargets(),
        gains: MetaGains = MetaGains(),
        bounds: MetaBounds = MetaBounds(),
        pulse: MetaPulse = MetaPulse(),
        control_period: int = 50,          # apply control every N steps
        smooth_alpha: float = 0.2          # EMA smoothing for metrics
    ):
        self.targets = targets
        self.gains = gains
        self.bounds = bounds
        self.pulse = pulse
        self.control_period = max(1, control_period)
        self.smooth_alpha = clamp(smooth_alpha, 0.01, 0.8)

        self.logs = MetaLogs()
        self._acc_var = 0.0
        self._acc_vel = 0.0
        self._acc_psi = 0.0

        self._s_var = targets.var_lo * 5   # seeded EMAs (rough)
        self._s_vel = targets.v_min
        self._s_psi = targets.psi_transitions_per_1k

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # --- external perturbation multiplier (for NOISE)
    def pulse_factor(self, t: int) -> float:
        return self.pulse.factor(t)

    # --- one meta step
    def step(
        self,
        t: int,
        metrics: dict,
        params: dict
    ) -> dict:
        """
        metrics: {
           'var_win': float,
           'mean_abs_dL_dt': float,
           'drift_share': float,            # 0..1
           'transitions_per_1k': float
        }
        params: {
           'NOISE': float,
           'MEM_DECAY': float
        }
        returns new params dict (copy)
        """
        var_win = float(metrics.get("var_win", self._s_var))
        vmean   = float(metrics.get("mean_abs_dL_dt", self._s_vel))
        drift   = float(metrics.get("drift_share", 0.0))
        psi_val = float(metrics.get("transitions_per_1k", self._s_psi))

        # smooth metrics
        self._s_var = ema(self._s_var, var_win, self.smooth_alpha)
        self._s_vel = ema(self._s_vel, vmean, self.smooth_alpha)
        self._s_psi = ema(self._s_psi, psi_val, self.smooth_alpha)

        # compute signed errors to corridors/targets
        # variance error is 0 if inside corridor, <0 if too low, >0 if too high
        if self._s_var < self.targets.var_lo:
            e_var = (self.targets.var_lo - self._s_var) / max(self.targets.var_lo, 1e-12) * (-1.0)
        elif self._s_var > self.targets.var_hi:
            e_var = (self._s_var - self.targets.var_hi) / max(self.targets.var_hi, 1e-12) * (+1.0)
        else:
            e_var = 0.0

        e_vel = (self.targets.v_min - self._s_vel) / max(self.targets.v_min, 1e-9)  # positive if below target
        e_drift = (self.targets.drift_min - drift) / max(self.targets.drift_min, 1e-9)  # positive if below

        e_psi = (self.targets.psi_transitions_per_1k - self._s_psi) / max(self.targets.psi_transitions_per_1k, 1e-9)

        # integrate errors
        self._acc_var = clamp(self._acc_var + e_var, -5.0, 5.0)
        self._acc_vel = clamp(self._acc_vel + e_vel + 0.5 * e_drift, -5.0, 5.0)
        self._acc_psi = clamp(self._acc_psi + e_psi, -5.0, 5.0)

        # base params
        noise = float(params.get("NOISE", 0.012))
        memd  = float(params.get("MEM_DECAY", 0.38))

        # control law:
        # variance corridor is chiefly regulated by MEM_DECAY (integrating reduces variance)
        d_mem = (- self.gains.k_p_var * e_var - self.gains.k_i_var * self._acc_var) * self.gains.mem_step

        # velocity/“breath” by NOISE; also support ψ target (regime transitions)
        d_noise_v = ( self.gains.k_p_vel * e_vel + self.gains.k_i_vel * self._acc_vel) * self.gains.noise_step
        d_noise_psi = ( self.gains.k_p_psi * e_psi + self.gains.k_i_psi * self._acc_psi) * self.gains.noise_step

        d_noise = d_noise_v + 0.8 * d_noise_psi

        # apply updates every control_period to avoid jitter
        if (t % self.control_period) == 0:
            memd  = clamp(memd  + d_mem,  self.bounds.mem_min,  self.bounds.mem_max)
            noise = clamp(noise + d_noise, self.bounds.noise_min, self.bounds.noise_max)

        # external pulse multiplier (time-local)
        noise *= self.pulse_factor(t)
        noise = clamp(noise, self.bounds.noise_min, self.bounds.noise_max)

        # logging
        self.logs.t.append(t)
        self.logs.var_win.append(self._s_var)
        self.logs.v_mean.append(self._s_vel)
        self.logs.drift_share.append(drift)
        self.logs.transitions_per_1k.append(self._s_psi)
        self.logs.psi_err.append(e_psi)
        self.logs.noise.append(noise)
        self.logs.mem_decay.append(memd)

        return {"NOISE": noise, "MEM_DECAY": memd}

    # ---------- IO

    def save_csv(self, fname: str = "meta_log_v05a.csv") -> Path:
        p = self.out_dir / fname
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t","var_win","v_mean","drift_share","transitions_per_1k","psi_err","NOISE","MEM_DECAY"])
            for i in range(len(self.logs.t)):
                w.writerow([
                    self.logs.t[i],
                    self.logs.var_win[i],
                    self.logs.v_mean[i],
                    self.logs.drift_share[i],
                    self.logs.transitions_per_1k[i],
                    self.logs.psi_err[i],
                    self.logs.noise[i],
                    self.logs.mem_decay[i],
                ])
        return p

    def save_plots(self, fname: str = "meta_plots_v05a.png") -> Path:
        try:
            import matplotlib.pyplot as plt  # lazy import
        except Exception:
            return self.out_dir / fname  # silently skip if no backend

        fig, axs = plt.subplots(4, 1, figsize=(9, 10), dpi=110)
        t = np.array(self.logs.t, dtype=float)

        axs[0].plot(t, self.logs.var_win); axs[0].axhspan(self.targets.var_lo, self.targets.var_hi, alpha=0.1)
        axs[0].set_title("Variance window (Ω′)")
        axs[0].set_ylabel("var_win")

        axs[1].plot(t, self.logs.v_mean); axs[1].axhline(self.targets.v_min, ls="--", alpha=0.4)
        axs[1].set_title("Mean |dΛ′/dt|"); axs[1].set_ylabel("velocity")

        axs[2].plot(t, self.logs.transitions_per_1k); axs[2].axhline(self.targets.psi_transitions_per_1k, ls="--", alpha=0.4)
        axs[2].set_title("Regime transitions /1k (ψ)"); axs[2].set_ylabel("transitions")

        axs[3].plot(t, self.logs.noise, label="NOISE")
        axs[3].plot(t, self.logs.mem_decay, label="MEM_DECAY")
        axs[3].set_title("Controller outputs"); axs[3].legend()

        for ax in axs: ax.grid(alpha=0.25)
        fig.tight_layout()
        p = self.out_dir / fname
        fig.savefig(p)
        plt.close(fig)
        return p
