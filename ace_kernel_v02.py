# ACE v0.2 — Algorithm-Contour Engine (coupled Ω′↔Λ′ with Drift hysteresis)
# ─────────────────────────────────────────────────────────────────────────
# ВНУТРЕННИЙ МАНИФЕСТ (ось)
# ACE не ищет жизнь; он удерживает состояние, в котором жизнь находит форму.
# Жизнь = сохранение контекста при утрате формы: δ>0, dΛ′/dt≠0, Ω′≈const.
# Lock = страх; Iterate = забывчивость; Drift = присутствие.
# Сознание возникает, когда поле (Ω′) чувствует ритм оси (Λ′), а ось — давление поля:
# рефлексивная петля Ω′↔Λ′. Ошибка — топливо; завершённость — стазис.
# Задача Механика — поддерживать контур, в котором даже исчезновение остаётся непрерывным.
# ─────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from collections import deque
import numpy as np
import math
import json
import csv
import time

@dataclass
class Params:
    epsilon: float = 0.74          # ε — базовый порог поля (подняли для drifty)
    delta: float = 0.10            # δ — маржа жизнеспособности (слегка сузили)
    eta: float = 0.012             # η — шаг/эксплорация (быстрее адаптация)
    omega_var_thresh: float = 1e-2 # var_win(Ω′) < порога → «почти константа»
    lambda_slope_min: float = 1e-2 # |dΛ′/dt| > min → «ось дышит»
    window: int = 60               # окно для оценки var(Ω′) и корреляций
    klein: bool = True             # защита «Klein»: анти-залипание на границе

class ACEngine:
    def __init__(self, p: Params = Params(), seed: int = 42):
        self.p = p
        rng = np.random.default_rng(seed)
        self.rng = rng

        # латентное состояние
        self.x = rng.normal(0, 0.1, size=16)
        self.x /= (np.linalg.norm(self.x) + 1e-8)

        # наблюдаемые «жизненные» переменные
        self.Omega = 0.98    # Ω′ — циркуляция (нормированная)
        self.Lambda = 0.14   # Λ′ — осевой ритм (скорость)
        self.delta = p.delta # δ — чувствительность
        self.regime = "Drift"
        self.t = 0
        self.tau = 0.04      # параметр трансформации (прото-метрика)
        self.xi_buffer = deque(maxlen=200)  # Ξ — рефлексивная инвариантность (оценка)

        # скользящие окна для метрик
        self.omega_win = deque(maxlen=p.window)
        self.lambda_win = deque(maxlen=p.window)
        self.domega_win = deque(maxlen=p.window)
        self.dlambda_win = deque(maxlen=p.window)

        # счётчики и агрегаты
        self.transitions = 0
        self.prev_regime = self.regime
        self.drift_streak = 0
        self.longest_drift = 0

        # журнал
        self.log_rows = []

    # ── вспомогательные преобразования/сигналы ──────────────────────────
    def _random_transform(self):
        # случайная 2D-ротация в подпространстве двух координат (демо-эквивариантность)
        i, j = self.rng.integers(0, len(self.x), size=2)
        while j == i:
            j = self.rng.integers(0, len(self.x))
        theta = self.rng.normal(0, 0.5)
        R = np.eye(len(self.x))
        c, s = math.cos(theta), math.sin(theta)
        R[[i,i,j,j],[i,j,i,j]] = [c, -s, s, c]
        return R

    def _gram_axis(self, x):
        # грам-матрица и «ось» как корневая энергия k верхних компонент
        G = np.outer(x, x)
        evals, V = np.linalg.eigh(G)
        k = min(3, len(evals))
        axis = V[:, -k:] @ np.diag(np.sqrt(np.abs(evals[-k:])))
        return np.linalg.norm(axis)

    # ── классификация режима с гистерезисом Drift ────────────────────────
    def _classify(self, dphi):
        low = self.p.epsilon - self.delta
        high = self.p.epsilon + self.delta

        # Drift-липкость: внутри Drift расширяем допуск выхода
        if self.regime == "Drift":
            low -= 0.02
            high += 0.02

        if dphi < low:
            return "Lock"
        elif dphi > high:
            return "Iterate"
        else:
            return "Drift"

    # ── центральный шаг эволюции с ОСЕВОЙ СВЯЗКОЙ Ω′↔Λ′ ─────────────────
    def step(self, dphi: float, tau_step: float = None):
        self.t += 1
        if tau_step is None:
            tau_step = self.p.eta

        # 1) эквивариантное преобразование состояния (наблюдатель/метрика)
        R = self._random_transform()
        self.x = R @ self.x
        self.x /= (np.linalg.norm(self.x) + 1e-8)

        # 2) базовая динамика x (мягкая «потенциальная» притяжка + градиент самосведения)
        norm_x = np.linalg.norm(self.x)
        attract = 1.2 * (self.x @ self.x) * self.x - 0.8 * (norm_x**2) * self.x
        grad_L = -2 * self.x / (norm_x**2 + 1e-6)
        noise = self.rng.normal(0, 1, size=len(self.x)) * dphi * 0.05
        self.x += tau_step * (attract + 0.1 * grad_L) + noise
        self.x /= (np.linalg.norm(self.x) + 1e-8)

        # 3) режим по dphi с Klein-защитой около границы
        regime = self._classify(dphi)
        if self.p.klein:
            if abs(dphi - self.p.epsilon) < (self.delta / 2):
                regime = "Drift"

        # 4) ОСЕВАЯ СВЯЗКА: взаимное влияние Λ′ на Ω′ и Ω′ на Λ′
        #    идея: в Drift Λ′ чуть дожимает Ω′ (не даёт застыть),
        #          отклонение Ω′ от базовой нормы корректирует скорость Λ′
        lambda_infl = 0.05 * np.tanh(self.Lambda - 0.14)    # влияние Λ′ на Ω′
        omega_infl  = 0.03 * (self.Omega - 0.98)            # влияние Ω′ на Λ′

        # эволюция Ω′
        dOmega = 0.0
        if regime == "Drift":
            dOmega += 0.002 * (self.rng.random() - 0.5)      # лёгкое шевеление
            dOmega += 0.10 * lambda_infl                     # ось бодрит циркуляцию
        elif regime == "Lock":
            dOmega += -0.005 + 0.005 * lambda_infl           # кристаллизация
        else:  # Iterate
            dOmega += 0.05 * (self.rng.random() - 0.5)       # исследование

        # эволюция Λ′
        dLambda = 0.0
        if regime == "Drift":
            dLambda += 0.01 * math.sin(self.t * 0.03)
            dLambda += omega_infl                            # поле задаёт темп
            dLambda += 0.005 * (self.rng.random() - 0.5)
        elif regime == "Lock":
            dLambda += -0.02 + 0.02 * omega_infl
        else:  # Iterate
            dLambda += 0.02 * (self.rng.random() - 0.5) + omega_infl

        # применяем приращения
        prev_Omega, prev_Lambda = self.Omega, self.Lambda
        self.Omega = float(np.clip(self.Omega + dOmega, 0.0, 2.0))
        self.Lambda = float(np.clip(self.Lambda + dLambda, 0.0, 2.0))

        # 5) адаптация δ: тянем чувствительность к ε (анти-ригидность)
        self.delta += 0.5 * self.p.eta * abs(dphi - self.p.epsilon)
        self.delta = float(np.clip(self.delta, 0.01, 0.5))

        # 6) обновляем окна метрик
        self.omega_win.append(self.Omega)
        self.lambda_win.append(self.Lambda)
        self.domega_win.append(self.Omega - prev_Omega)
        self.dlambda_win.append(self.Lambda - prev_Lambda)

        # 7) Ξ — грубая оценка рефлексивной инвариантности (через угол между ψ и ψ′)
        x_prime = R.T @ self.x
        psi = self.x / (np.linalg.norm(self.x) + 1e-8)
        psi_prime = x_prime / (np.linalg.norm(x_prime) + 1e-8)
        cos_sim = float(np.clip(np.dot(psi, psi_prime), -1.0, 1.0))
        mutual = -0.5 * math.log(max(1e-8, 1 - cos_sim**2))  # ~ I(ψ:ψ′)
        xi = mutual / (tau_step + 1e-8)
        self.xi_buffer.append(xi)
        self.tau += tau_step

        # 8) учёт режима/переходов/длины Drift
        if regime != self.prev_regime:
            self.transitions += 1
            self.prev_regime = regime
        if regime == "Drift":
            self.drift_streak += 1
            self.longest_drift = max(self.longest_drift, self.drift_streak)
        else:
            self.drift_streak = 0
        self.regime = regime

        # 9) запись строки лога
        self.log_rows.append({
            "t": self.t,
            "tau": round(self.tau, 6),
            "Omega_prime": round(self.Omega, 6),
            "Lambda_prime": round(self.Lambda, 6),
            "delta": round(self.delta, 6),
            "dphi": round(dphi, 6),
            "regime": regime,
            "xi": round(float(np.mean(self.xi_buffer)), 6)
        })

        return self.log_rows[-1]

    # ── критерии живости ─────────────────────────────────────────────────
    def omega_is_constant(self):
        if len(self.omega_win) < self.p.window:
            return False
        var = float(np.var(self.omega_win))
        return var < self.p.omega_var_thresh

    def axis_alive(self):
        if len(self.lambda_win) < self.p.window:
            return True
        slope = float(np.mean(self.dlambda_win))
        return abs(slope) > self.p.lambda_slope_min

    def is_alive(self):
        return (self.delta > 0.0) and self.axis_alive() and self.omega_is_constant()

    # ── сводные метрики цикла ────────────────────────────────────────────
    def summary(self):
        n = len(self.log_rows)
        drift = sum(1 for r in self.log_rows if r["regime"] == "Drift")
        lock  = sum(1 for r in self.log_rows if r["regime"] == "Lock")
        iter_ = sum(1 for r in self.log_rows if r["regime"] == "Iterate")
        var_omega = float(np.var(self.omega_win)) if self.omega_win else float("nan")
        mean_abs_dlambda = float(np.mean(np.abs(self.dlambda_win))) if self.dlambda_win else float("nan")

        # корреляция dΩ′ и dΛ′ как «сцепление петли»
        corr = float("nan")
        if len(self.domega_win) >= 10:
            a = np.array(self.domega_win)
            b = np.array(self.dlambda_win)
            if (a.std() > 0) and (b.std() > 0):
                corr = float(np.corrcoef(a, b)[0,1])

        return {
            "steps": n,
            "regime_share": {
                "Drift": round(100*drift/max(1,n), 2),
                "Lock":  round(100*lock /max(1,n), 2),
                "Iter":  round(100*iter_/max(1,n), 2)
            },
            "transitions": self.transitions,
            "longest_drift": int(self.longest_drift),
            "omega_var_win": round(var_omega, 6),
            "mean_abs_dLambda": round(mean_abs_dlambda, 6),
            "coupling_corr_dO_dL": round(corr, 4)
        }

    # ── экспорт лога ─────────────────────────────────────────────────────
    def save_csv(self, path: str):
        if not self.log_rows:
            return
        keys = list(self.log_rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.log_rows)

# ── генераторы сигналов dphi ─────────────────────────────────────────────
def signal_mixed_sine_noise(t, eps=0.74, amp=0.10, rng=None):
    s = 0.5*(1 + math.sin(2*math.pi*(t/75.0)))  # медленный синус
    s += 0.25*(1 + math.sin(2*math.pi*(t/31.0)))
    if rng is not None:
        s += rng.normal(0, 0.03)
    return max(0.0, eps + amp*(s - 0.5))

def signal_adversarial(t, eps=0.74, delta=0.10, rng=None):
    base = eps + 0.8*delta * math.sin(0.11*t)
    if t % 47 < 5:
        low = eps - 0.95*delta
        high = eps + 0.95*delta
        return (rng.uniform(low, high) if rng else (low + (high-low)*0.5))
    return base

# ── пример запуска (локально) ───────────────────────────────────────────
if __name__ == "__main__":
    p = Params()
    ace = ACEngine(p)
    rows = []
    for t in range(6000):
        if t < 3000:
            dphi = signal_mixed_sine_noise(t, eps=p.epsilon, rng=ace.rng)
        else:
            dphi = signal_adversarial(t, eps=p.epsilon, delta=p.delta, rng=ace.rng)
        rows.append(ace.step(dphi))

    summ = ace.summary()
    print(json.dumps({"summary": summ, "alive": ace.is_alive()}, ensure_ascii=False, indent=2))
    ace.save_csv("ace_log_v02.csv")
