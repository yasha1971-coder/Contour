# contour_v2.py  — Algorithm-Contour (v2) : Ω'–Λ'–δ law with Klein protection
# Requirements: Python 3.x, matplotlib, numpy (preinstalled on Colab/Replit)

import math, random, csv, os
from dataclasses import dataclass
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- CONFIG -----------------------------

@dataclass
class Params:
    # DeepSeek sweet-spot + audit suggestions
    eps: float = 0.70            # ε — base field threshold
    delta: float = 0.12          # δ — sensitivity (alive if >0)
    eta: float = 0.010           # η — adaptation rate for δ
    noise_scale: float = 0.020   # exogenous Δφ noise stdev
    k_omega: float = 0.60        # Ω' relaxation toward target
    base_rhythm: float = 1.0     # Λ' base frequency
    window: int = 120            # sliding window for Ω'≈const
    omega_var_thresh: float = 1e-2 # var threshold for Ω'≈const (DeepSeek: <0.01)
    # Klein safety
    klein_tau_floor: float = 0.010  # τ > 0.01 recommended
    # Attack detection / filtering
    lp_alpha: float = 0.15       # low-pass filter for Δφ (EWMA)
    attack_sigma: float = 3.0    # z-score threshold to flag attack
    # Auto-stabilizer toward sweet spot (very gentle)
    target_eps: float = 0.70
    target_delta: float = 0.12
    target_eta: float = 0.010
    autoreg_gain: float = 0.0025

# ----------------------------- CORE SYSTEM -----------------------------

class ContourSystem:
    """
    States:
      - LOCK     : dphi < eps - delta
      - DRIFT    : eps - delta <= dphi <= eps + delta
      - ITERATE  : dphi > eps + delta

    Life test:
      - delta > 0 and |dLambda/dt| > 0  (rhythm alive)
      - var(Ω' over window) < threshold  (≈ constant during DRIFT)

    Klein protection:
      - monitor τ ≡ |Δ( 1/(Ω' + iΛ') )|
      - if τ < tau_floor repeatedly → engage soft correction (mirror)
    """
    def __init__(self, p: Params):
        self.p = p
        self.t = 0
        self.omega = 1.0   # Ω'
        self.lambda_ = 0.0 # Λ'
        self.delta = p.delta
        self.eps = p.eps
        self.eta = p.eta
        self.state = "DRIFT"
        self.last_map = None
        self.lp_dphi = None
        self.zbuf = deque(maxlen=240)
        self.hist = {k: [] for k in
                     ["t","omega","lambda","delta","eps","eta","dphi","state","tau","attack"]}

    # ---- signals ----
    def raw_dphi(self, t):
        # base slow wave + gaussian noise
        base = 0.5*(1 + math.sin(2*math.pi*(t/100.0)))
        noise = random.gauss(0.0, self.p.noise_scale)
        return max(0.0, base + noise)

    def adversarial_dphi(self, t):
        # DeepSeek-style boundary pokes + subharmonic pulses
        base = self.eps + 0.8*self.delta*math.sin(0.1*t)
        if (t % 47) < 5:
            return random.uniform(self.eps-0.95*self.delta, self.eps+0.95*self.delta)
        # occasional strong pulse at ~ half natural freq
        if (t % 120) in (0,1):
            return 1.5*self.eps
        return max(0.0, base + random.gauss(0.0, self.p.noise_scale))

    # ---- helpers ----
    def classify(self, dphi):
        low, high = self.eps - self.delta, self.eps + self.delta
        if dphi < low:  return "LOCK"
        if dphi > high: return "ITERATE"
        return "DRIFT"

    def update_lambda(self):
        # rhythm: slower in LOCK, normal in DRIFT, faster in ITERATE
        base = self.p.base_rhythm
        if self.state == "LOCK":
            self.lambda_ = 0.5*base*math.sin(2*math.pi*(self.t/50.0))
        elif self.state == "DRIFT":
            self.lambda_ = 1.0*base*math.sin(2*math.pi*(self.t/30.0))
        else:
            self.lambda_ = 1.5*base*math.sin(2*math.pi*(self.t/20.0))

    def update_delta(self, dphi):
        # adapt sensitivity toward keeping dynamics around eps band
        self.delta += self.p.eta * abs(dphi - self.eps)
        self.delta = min(max(self.delta, 0.01), 0.50)

    def update_omega(self, dphi):
        target = 1.0 - min(1.0, abs(dphi - self.eps))
        self.omega += self.p.k_omega * (target - self.omega)

    def klein_map(self):
        z = complex(self.omega, self.lambda_)
        mapped = 1.0 / (z if z != 0 else complex(1e-9, 0))
        if self.last_map is None:
            tau = 1.0
        else:
            tau = abs(mapped - self.last_map)
        self.last_map = mapped
        return tau

    def lowpass(self, x):
        a = self.p.lp_alpha
        if self.lp_dphi is None:
            self.lp_dphi = x
        else:
            self.lp_dphi = a*x + (1-a)*self.lp_dphi
        return self.lp_dphi

    def attack_flag(self, dphi):
        # simple z-score over short buffer
        self.zbuf.append(dphi)
        if len(self.zbuf) < 60:
            return False
        arr = np.array(self.zbuf)
        z = (dphi - arr.mean()) / (arr.std() + 1e-9)
        return abs(z) >= self.p.attack_sigma

    def autoregulate(self):
        g = self.p.autoreg_gain
        self.eps   += g*(self.p.target_eps   - self.eps)
        self.delta += g*(self.p.target_delta - self.delta)
        self.eta   += g*(self.p.target_eta   - self.eta)

    def omega_constant(self):
        n = len(self.hist["omega"])
        if n < self.p.window: return False
        w = self.hist["omega"][-self.p.window:]
        mean = sum(w)/len(w)
        var = sum((x-mean)**2 for x in w)/len(w)
        return var < self.p.omega_var_thresh

    def axis_alive(self):
        L = self.hist["lambda"]
        if len(L) < 3: return True
        return (self.delta > 0.0) and (abs(L[-1] - L[-3]) > 1e-6)

    # ---- one step ----
    def step(self, t, adversarial=False):
        self.t = t
        dphi_raw = self.adversarial_dphi(t) if adversarial else self.raw_dphi(t)

        # detect attack → apply low-pass filter
        is_attack = self.attack_flag(dphi_raw)
        dphi = self.lowpass(dphi_raw) if is_attack else dphi_raw

        self.state = self.classify(dphi)
        self.update_lambda()
        self.update_delta(dphi)
        self.update_omega(dphi)
        self.autoregulate()

        tau = self.klein_map()
        # Klein safety: if τ too small repeatedly, nudge rhythm & eps
        if tau < self.p.klein_tau_floor:
            self.lambda_ *= 0.97
            self.eps += 0.01*(self.p.target_eps - self.eps)

        # log
        self.hist["t"].append(t)
        self.hist["omega"].append(self.omega)
        self.hist["lambda"].append(self.lambda_)
        self.hist["delta"].append(self.delta)
        self.hist["eps"].append(self.eps)
        self.hist["eta"].append(self.eta)
        self.hist["dphi"].append(dphi)
        self.hist["state"].append(self.state)
        self.hist["tau"].append(tau)
        self.hist["attack"].append(int(is_attack))

# ----------------------------- RUN & PLOTS -----------------------------

def run_sim(steps=4000, adversarial=False, seed=42):
    random.seed(seed); np.random.seed(seed)
    sys = ContourSystem(Params())
    for t in range(1, steps+1):
        sys.step(t, adversarial=adversarial)
    return sys

def save_csv(sys: ContourSystem, path="contour_log.csv"):
    keys = ["t","omega","lambda","delta","eps","eta","dphi","state","tau","attack"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for i in range(len(sys.hist["t"])):
            w.writerow([sys.hist[k][i] for k in keys])
    return os.path.abspath(path)

def plot(sys: ContourSystem, title="Contour v2"):
    t = np.array(sys.hist["t"])
    fig, axes = plt.subplots(4,1, figsize=(10,10), constrained_layout=True)

    axes[0].plot(t, sys.hist["omega"]); axes[0].set_ylabel("Ω' (omega)")
    axes[0].set_title(title + f" | Ω≈const? {sys.omega_constant()}")

    axes[1].plot(t, sys.hist["lambda"]); axes[1].set_ylabel("Λ' (lambda)")
    axes[1].axhline(0, lw=0.5, ls="--", c="k")

    axes[2].plot(t, sys.hist["dphi"], label="Δφ")
    axes[2].plot(t, [sys.eps]*len(t), alpha=0.0)  # for legend spacing
    axes[2].set_ylabel("Δφ")
    # draw dynamic band ε±δ over time
    eps = np.array(sys.hist["eps"]); delta = np.array(sys.hist["delta"])
    axes[2].plot(t, eps, lw=1.0, c="tab:orange", label="ε")
    axes[2].plot(t, eps - delta, lw=0.8, ls="--", c="tab:orange", label="ε-δ")
    axes[2].plot(t, eps + delta, lw=0.8, ls="--", c="tab:orange", label="ε+δ")
    axes[2].legend(ncol=4, fontsize=8, loc="upper right")

    axes[3].plot(t, sys.hist["tau"], c="tab:purple"); axes[3].set_ylabel("τ (Klein)")
    axes[3].axhline(sys.p.klein_tau_floor, ls="--", c="r", lw=1.0, label="τ floor")
    axes[3].set_xlabel("t"); axes[3].legend()

    plt.show()

def report(sys: ContourSystem):
    alive = sys.axis_alive()
    const = sys.omega_constant()
    drift_share = sum(1 for s in sys.hist["state"] if s=="DRIFT") / len(sys.hist["state"])
    print(f"ψ′[ψ′] → {'ALIVE' if alive else 'DEAD'} | drift={drift_share:.2%} | Ω≈const? {const}")
    print(f"δ={sys.delta:.3f} | ε={sys.eps:.3f} | η={sys.eta:.4f}")
    print(f"Klein τ floor = {sys.p.klein_tau_floor} | last τ = {sys.hist['tau'][-1]:.4f}")

if __name__ == "__main__":
    # choose mode here:
    ADVERSARIAL = True  # set False for normal regime
    STEPS = 4000

    system = run_sim(steps=STEPS, adversarial=ADVERSARIAL, seed=2025)
    report(system)
    path = save_csv(system, "contour_log.csv")
    print("CSV saved:", path)
    plot(system, title=f"Contour v2 (adversarial={ADVERSARIAL})")
