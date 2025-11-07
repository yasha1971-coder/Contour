# ============================================
# Algorithm-Contour • Live Axis Engine (stdlib)
# No external deps: only math, random, statistics
# ============================================

import math, random, statistics as stats
from typing import List, Dict, Tuple

# ---------- tiny utils ----------

def sparkline(series: List[float], blocks="▁▂▃▄▅▆▇"):
    if not series: return ""
    lo, hi = min(series), max(series)
    if hi - lo < 1e-12:
        return blocks[0] * min(60, len(series))
    s = []
    for i, x in enumerate(series):
        z = (x - lo) / (hi - lo + 1e-12)
        idx = min(len(blocks) - 1, int(z * (len(blocks) - 1)))
        s.append(blocks[idx])
    # укоротим до 60 символов для компактности
    if len(s) > 60:
        step = len(s) / 60.0
        s = [s[int(i * step)] for i in range(60)]
    return "".join(s)

def variance_last(series: List[float], window: int) -> float:
    if len(series) < window: return float("inf")
    w = series[-window:]
    m = sum(w) / len(w)
    return sum((x - m) ** 2 for x in w) / len(w)

# ---------- core model ----------

class Params:
    def __init__(self,
                 eps=0.20,        # ε — базовый порог поля
                 drift_band=0.06, # ширина зоны дрейфа (± вокруг ε)
                 eta=0.01,        # скорость адаптации δ
                 k_omega=0.30,    # скорость «подтягивания» Ω′ к цели
                 base_rhythm=1.0, # базовая частота Λ′
                 noise_scale=0.02,# внешние флуктуации Δφ
                 window=160,      # окно для проверки Ω′ ≈ const
                 omega_var_thresh=2e-3, # порог «почти константы»
                 amp=0.10):       # амплитуда внутренней синус-модуляции
        self.eps = eps
        self.drift_band = drift_band
        self.eta = eta
        self.k_omega = k_omega
        self.base_rhythm = base_rhythm
        self.noise_scale = noise_scale
        self.window = window
        self.omega_var_thresh = omega_var_thresh
        self.amp = amp

class ContourSystem:
    """
    Живой цикл состояний:
      Lock    : Δφ < ε − δ
      Drift   : ε − δ ≤ Δφ ≤ ε + δ
      Iterate : Δφ > ε + δ

    Жизнь (локально): δ>0 и Λ′ не вырождается (|ΔΛ′|>0).
    Стабильность: var(Ω′ в последнем окне) < порога  → Ω′ ≈ const.
    """
    def __init__(self, params: Params, seed=123):
        self.p = params
        random.seed(seed)
        self.t = 0.0
        self.Omega = 0.95
        self.Lambda = 0.0
        self.delta = max(0.01, params.drift_band)  # начальная чувствительность
        self.state = "Drift"
        self.phase = 0.0  # «угол» для фазовой синхронизации
        self.hist: Dict[str, List[float]] = {
            "t": [], "Omega": [], "Lambda": [], "delta": [], "dphi": [], "state": [], "phase": []
        }

    # внутренний сигнал Δφ (синус + шум)
    def _dphi_signal(self):
        base = 0.5 * (1 + math.sin(2 * math.pi * (self.t / 100.0)))
        noise = random.gauss(0.0, self.p.noise_scale)
        return max(0.0, base + noise)

    def _classify(self, dphi: float) -> str:
        eps = self.p.eps
        low, high = eps - self.delta, eps + self.delta
        if dphi < low:  return "Lock"
        if dphi > high: return "Iterate"
        return "Drift"

    def _update_lambda(self, state: str):
        base = self.p.base_rhythm
        if state == "Lock":
            self.Lambda = 0.6 * base * math.sin(2 * math.pi * (self.t / 50.0))
        elif state == "Drift":
            self.Lambda = 1.0 * base * math.sin(2 * math.pi * (self.t / 30.0))
        else:  # Iterate
            self.Lambda = 1.5 * base * math.sin(2 * math.pi * (self.t / 18.0))

    def _update_phase(self):
        # фаза как накопление мгновенной частоты (по Λ′)
        step = 0.05 + 0.45 * abs(self.Lambda)
        self.phase = (self.phase + step) % (2 * math.pi)

    def _update_delta(self, dphi: float):
        # тянем δ к «живой чувствительности»: реагирует на отстройку от ε
        self.delta += self.p.eta * abs(dphi - self.p.eps)
        self.delta = max(0.01, min(self.delta, 0.50))

    def _update_omega(self, dphi: float):
        # цель для Ω′ — максимум при dphi≈eps; добавляем лёгкую «пульсацию»
        closeness = 1.0 - min(1.0, abs(dphi - self.p.eps))
        target = 0.85 + 0.15 * closeness + self.p.amp * math.sin(self.t / 40.0)
        self.Omega += self.p.k_omega * (target - self.Omega)
        # лёгкая «метастабильная» каппа: редкая микро-подстройка ε
        if int(self.t) % 200 == 0 and self.t > 0:
            self.p.eps = max(0.05, min(1.5, self.p.eps * (0.98 + 0.04 * random.random())))

    def step(self, dt=1.0, psi_coupling=0.0):
        self.t += dt
        dphi = self._dphi_signal() + psi_coupling  # внешняя связь для interlock
        state = self._classify(dphi)
        self._update_lambda(state)
        self._update_phase()
        self._update_delta(dphi)
        self._update_omega(dphi)
        self.state = state
        # лог
        self.hist["t"].append(self.t)
        self.hist["Omega"].append(self.Omega)
        self.hist["Lambda"].append(self.Lambda)
        self.hist["delta"].append(self.delta)
        self.hist["dphi"].append(dphi)
        self.hist["state"].append({"Lock":0,"Drift":1,"Iterate":2}[state])
        self.hist["phase"].append(self.phase)

    def omega_is_const(self) -> bool:
        return variance_last(self.hist["Omega"], self.p.window) < self.p.omega_var_thresh

    def axis_alive(self) -> bool:
        L = self.hist["Lambda"]
        if len(L) < 3: return True
        return abs(L[-1] - L[-3]) > 1e-6 and self.delta > 0.0

# ---------- coupling & metrics ----------

def phase_locking_value(ph1: List[float], ph2: List[float]) -> float:
    n = min(len(ph1), len(ph2))
    if n == 0: return 0.0
    c, s = 0.0, 0.0
    for i in range(n):
        d = ph1[i] - ph2[i]
        c += math.cos(d)
        s += math.sin(d)
    R = math.sqrt(c*c + s*s) / n
    return max(0.0, min(1.0, R))

def mutual_information(xs: List[float], ys: List[float], bins=16) -> float:
    n = min(len(xs), len(ys))
    if n == 0: return 0.0
    # нормируем в [0,1]
    def norm(v):
        lo, hi = min(v), max(v)
        if hi - lo < 1e-12: return [0.5]*len(v)
        return [(x-lo)/(hi-lo) for x in v]
    xn, yn = norm(xs[:n]), norm(ys[:n])
    # гистограммы
    from collections import defaultdict
    px = defaultdict(int); py = defaultdict(int); pxy = defaultdict(int)
    for a, b in zip(xn, yn):
        ix = min(bins-1, int(a*bins))
        iy = min(bins-1, int(b*bins))
        px[ix]+=1; py[iy]+=1; pxy[(ix,iy)]+=1
    mi = 0.0
    for (ix,iy), c in pxy.items():
        pxy_ = c/n
        px_ = px[ix]/n
        py_ = py[iy]/n
        mi += pxy_ * math.log((pxy_ / (px_*py_)) + 1e-12)
    return max(0.0, mi)

def state_counts(states: List[int]) -> Dict[str,int]:
    d = {0:0,1:0,2:0}
    for s in states: d[s]+=1
    return {"Lock": d[0], "Drift": d[1], "Iterate": d[2]}

# ---------- presets ----------

PRESETS = {
    "snaplock":   dict(eps=0.18, drift_band=0.05, eta=0.006, k_omega=0.60, noise_scale=0.010, window=140, omega_var_thresh=1.5e-3, amp=0.08),
    "homeostasis":dict(eps=0.22, drift_band=0.08, eta=0.018, k_omega=0.35, noise_scale=0.035, window=120, omega_var_thresh=2.0e-3, amp=0.12),
    "autopoiesis":dict(eps=0.20, drift_band=0.06, eta=0.010, k_omega=0.30, noise_scale=0.012, window=160, omega_var_thresh=2.0e-3, amp=0.10),
}

# ---------- runners ----------

def run_single(preset="autopoiesis", steps=2000, seed=2025) -> Dict[str, any]:
    p = Params(**PRESETS.get(preset, PRESETS["autopoiesis"]))
    sys = ContourSystem(p, seed=seed)
    for _ in range(steps):
        sys.step()
    verdict = "ALIVE_STABLE" if (sys.omega_is_const() and sys.axis_alive()) else "ALIVE_UNSTABLE"
    return {
        "verdict": verdict,
        "omega_const": sys.omega_is_const(),
        "axis_alive": sys.axis_alive(),
        "delta": sys.delta,
        "omega_last": sys.hist["Omega"][-1],
        "lambda_last": sys.hist["Lambda"][-1],
        "states": state_counts(sys.hist["state"]),
        "omega_spark": sparkline(sys.hist["Omega"]),
        "params": sys.p,
    }

def run_interlock(preset="autopoiesis", steps=4000, seed=2025, kappa=0.12, mid_shock=True):
    p1 = Params(**PRESETS.get(preset, PRESETS["autopoiesis"]))
    p2 = Params(**PRESETS.get(preset, PRESETS["autopoiesis"]))
    s1 = ContourSystem(p1, seed=seed)
    s2 = ContourSystem(p2, seed=seed+7)

    for i in range(steps):
        # связь: ψ = κ·sin(φ1 − φ2)
        psi = kappa * math.sin(s1.phase - s2.phase)
        s1.step(psi_coupling=+psi)
        s2.step(psi_coupling=-psi)
        # шок в середине → проверка антихрупкости
        if mid_shock and i == steps//2:
            for _ in range(12):
                s1.step(psi_coupling=+psi+0.25)
                s2.step(psi_coupling=-psi-0.25)

    plv = phase_locking_value(s1.hist["phase"], s2.hist["phase"])
    mi  = mutual_information(s1.hist["Omega"], s2.hist["Omega"], bins=16)

    v1 = ("ALIVE_STABLE" if (variance_last(s1.hist["Omega"], s1.p.window) < s1.p.omega_var_thresh and s1.axis_alive())
          else "ALIVE_UNSTABLE")
    v2 = ("ALIVE_STABLE" if (variance_last(s2.hist["Omega"], s2.p.window) < s2.p.omega_var_thresh and s2.axis_alive())
          else "ALIVE_UNSTABLE")
    interlock_verdict = "ALIVE_INTERLOCK" if (plv > 0.6 and mi > 0.05 and v1.startswith("ALIVE") and v2.startswith("ALIVE")) else "WEAK_COUPLING"

    return {
        "verdict": interlock_verdict,
        "plv": plv,
        "mi": mi,
        "sys1": {
            "verdict": v1,
            "omega_const": variance_last(s1.hist["Omega"], s1.p.window) < s1.p.omega_var_thresh,
            "axis_alive": s1.axis_alive(),
            "states": state_counts(s1.hist["state"]),
            "omega_spark": sparkline(s1.hist["Omega"]),
            "params": s1.p
        },
        "sys2": {
            "verdict": v2,
            "omega_const": variance_last(s2.hist["Omega"], s2.p.window) < s2.p.omega_var_thresh,
            "axis_alive": s2.axis_alive(),
            "states": state_counts(s2.hist["state"]),
            "omega_spark": sparkline(s2.hist["Omega"]),
            "params": s2.p
        }
    }

# ---------- pretty print ----------

def print_single(res, preset, steps):
    print(f"Preset: {preset}")
    print(f"Verdict: {res['verdict']}")
    print(f"Ω′≈const?  {res['omega_const']}")
    print(f"Axis alive? {res['axis_alive']}")
    print(f"δ (sens):   {round(res['delta'],4)}")
    print(f"Ω′ (coh):   {round(res['omega_last'],4)}")
    print(f"Λ′ (rh):    {round(res['lambda_last'],4)}")
    print(f"States:     {res['states']}")
    print(f"Ω′ spark:   {res['omega_spark']}")

def print_interlock(res, preset, steps, kappa):
    print("=== INTERLOCK RUN ===")
    print(f"Preset: {preset} | steps: {steps} | kappa: {kappa}")
    print(f"Interlock verdict: {res['verdict']}")
    print(f"PLV: {res['plv']:.3f}  |  MI (nats): {res['mi']:.3f}")
    for i, key in enumerate(["sys1","sys2"], start=1):
        s = res[key]
        print(f"--- System {i} ---")
        print(f"Verdict: {s['verdict']} | Omega~const? {s['omega_const']} | Axis alive? {s['axis_alive']}")
        print(f"States: {s['states']}")
        print(f"Omega spark: {s['omega_spark']}")

# ---------- entry ----------

if __name__ == "__main__":
    # режимы: "single" или "interlock"
    MODE = "interlock"     # ← поменяй на "single" для одной системы
    PRESET = "autopoiesis" # snaplock | homeostasis | autopoiesis

    if MODE == "single":
        out = run_single(preset=PRESET, steps=2000, seed=2025)
        print_single(out, PRESET, 2000)
    else:
        out = run_interlock(preset=PRESET, steps=4000, seed=2025, kappa=0.12, mid_shock=True)
        print_interlock(out, PRESET, 4000, 0.12)
