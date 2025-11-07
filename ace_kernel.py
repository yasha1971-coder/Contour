# ace_kernel.py
# Minimal ACE v0.1 runtime — Ω′–Λ′–δ loop with QC and logging
# Requirements: Python 3.x, numpy

import numpy as np
import csv
from collections import deque
from math import sin, pi
import time

# -----------------------------
# Tunable parameters (defaults)
# -----------------------------
EPSILON = 0.68          # ε — базовый порог поля
DELTA   = 0.15          # δ — ширина жизненной зоны вокруг ε (±)
ETA     = 0.008         # η — шаг преобразования/эволюции
NOISE   = 0.05          # интенсивность внешнего шума в dphi
OMEGA_VAR_THRESH   = 1e-2   # порог "Ω′ ≈ const" по скользящей дисперсии
LAMBDA_SLOPE_MIN   = 1e-2   # минимальная "скорость дыхания" оси
WIN                = 60     # окно для вар. Ω′ и оценки дыхания
STEPS              = 4000   # число шагов симуляции
DIM                = 16     # размерность латентного состояния
KLEIN_GUARD        = True   # защита от застревания на границе (дрейф-замыкание)

# -----------------------------
# Utility
# -----------------------------
def random_orthogonal(n, scale=1.0):
    """Случайная ортогональная матрица через QR; scale ослабляет 'шаг' трансформации."""
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    # слегка "подмешаем" единицу, чтобы управлялась интенсивность
    return (1.0 - scale) * np.eye(n) + scale * Q

def mutual_info_proxy(u, v, eps=1e-8):
    """Простейший прокси взаимной информации по косинусной близости."""
    u = u / (np.linalg.norm(u) + eps)
    v = v / (np.linalg.norm(v) + eps)
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    # -0.5 * log(1 - c^2) — гладкий рост к бесконечности при совпадении направлений
    return float(-0.5 * np.log(1.0 - c * c + eps))

# -----------------------------
# ACEngine
# -----------------------------
class ACEngine:
    def __init__(self,
                 epsilon=EPSILON, delta=DELTA, eta=ETA,
                 omega_var_thresh=OMEGA_VAR_THRESH,
                 lambda_slope_min=LAMBDA_SLOPE_MIN,
                 klein_guard=KLEIN_GUARD,
                 dim=DIM):
        self.epsilon = float(epsilon)
        self.delta   = float(delta)
        self.eta     = float(eta)
        self.omega_var_thresh = float(omega_var_thresh)
        self.lambda_slope_min = float(lambda_slope_min)
        self.klein_guard = bool(klein_guard)

        # state
        self.x = np.random.randn(dim) * 0.1
        self.x /= (np.linalg.norm(self.x) + 1e-8)

        self.omega_win  = deque(maxlen=WIN)
        self.lambda_win = deque(maxlen=WIN)
        self.xi_win     = deque(maxlen=100)

        self.tau = 0.04  # параметр "преобразования"
        self.last_lambda_value = 0.0

    def classify_regime(self, dphi):
        low  = self.epsilon - self.delta
        high = self.epsilon + self.delta
        if dphi < low:
            return "Lock"
        elif dphi > high:
            return "Iterate"
        else:
            return "Drift"

    def klein_transition(self, regime, dphi):
        if not self.klein_guard:
            return regime
        # если слишком близко к границе — принудительно в Drift (анти-стоп-вилы)
        boundary = abs(dphi - self.epsilon)
        if boundary < (self.delta * 0.5):
            return "Drift"
        return regime

    def step(self, dphi, tau_step):
        # 1) случайная ортогональная трансформация состояния (смена рамки/наблюдателя)
        R = random_orthogonal(len(self.x), scale=min(0.5, tau_step*10.0))
        self.x = R @ self.x

        # 2) внутренние динамики (эквивариантная эволюция + "самоинфо"-градиент)
        norm_x = np.linalg.norm(self.x) + 1e-8
        cubic   = 1.2 * (self.x @ self.x) * self.x
        quartic = 0.8 * (norm_x ** 2) * self.x
        grad_L  = -2.0 * self.x / (norm_x ** 2)  # d(-log det(xx^T + epsI))/dx ~ -2x/||x||^2
        # обновление состояния
        self.x += tau_step * ( (R.T @ (cubic - quartic)) + 0.10 * grad_L ) \
                  + np.random.randn(len(self.x)) * dphi * 0.05
        self.x /= (np.linalg.norm(self.x) + 1e-8)

        # 3) Ω′ как отношение структуры к инфо-градиенту (прокси)
        omega_prime = (np.linalg.norm(self.x) ** 2) / (np.linalg.norm(grad_L) + 1e-8)
        self.omega_win.append(float(omega_prime))

        # 4) Λ′ — «дыхание оси»: возьмём скорость изменения реконструкции (простой прокси)
        #   здесь используем норму приращения состояния как ритм (быстро, стабильно, без SVD)
        lambda_prime = float(np.linalg.norm(self.x - R.T @ self.x))  # frame-aware "движение"
        self.lambda_win.append(lambda_prime)

        # 5) классификация режима + Klein guard
        regime = self.classify_regime(dphi)
        regime = self.klein_transition(regime, dphi)

        # 6) Ξ — рефлексивная инвариантность (прокси взаим.инф.) с обратным преобразованием
        x_prime = R.T @ self.x
        xi = mutual_info_proxy(self.x, x_prime) / (tau_step + 1e-8)
        self.xi_win.append(xi)

        self.tau += tau_step

        return omega_prime, lambda_prime, xi, regime

    # QC: Ω′ ≈ const
    def omega_is_constant(self):
        if len(self.omega_win) < WIN:
            return False
        return float(np.var(self.omega_win)) < self.omega_var_thresh

    # QC: |dΛ′/dt| > 0
    def axis_is_breathing(self):
        if len(self.lambda_win) < WIN:
            return True
        lam = np.array(self.lambda_win, dtype=float)
        slopes = np.diff(lam)
        return abs(np.mean(slopes)) > self.lambda_slope_min

    def is_alive(self):
        return (self.delta > 0.0) and self.axis_is_breathing() and self.omega_is_constant()

# -----------------------------
# Main simulation
# -----------------------------
def main():
    ace = ACEngine()
    drift_steps = 0

    # лог в CSV
    with open("ace_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step","Omega_prime","Lambda_prime","delta","dphi","regime","xi","tau","alive"])

        for t in range(STEPS):
            # внешнее возмущение dphi: база около ε + синус + шум
            base = ace.epsilon + 0.30 * sin(0.71 * (t * pi / 30.0))
            dphi = float(base + np.random.randn() * NOISE)
            dphi = max(0.0, dphi)

            # редкий "спайк-атака"
            if t in (1200, 3000):
                dphi *= 1.8

            omega_p, lambda_p, xi, regime = ace.step(dphi=dphi, tau_step=ETA)
            alive = ace.is_alive()
            if regime == "Drift":
                drift_steps += 1

            writer.writerow([t, f"{omega_p:.6f}", f"{lambda_p:.6f}",
                             f"{ace.delta:.6f}", f"{dphi:.6f}", regime,
                             f"{xi:.6f}", f"{ace.tau:.6f}", str(alive)])

    # сводка
    drift_ratio = drift_steps / STEPS
    omega_var   = float(np.var(list(ace.omega_win))) if len(ace.omega_win) > 1 else float('nan')
    # оценка дыхания как средняя |dΛ′/dt|
    if len(ace.lambda_win) > 1:
        lam = np.array(ace.lambda_win, dtype=float)
        dlam = np.abs(np.diff(lam))
        lambda_speed = float(np.mean(dlam))
    else:
        lambda_speed = float('nan')

    verdict = "ALIVE" if ace.is_alive() else "NOT ALIVE"

    print("\n=== ACE v0.1 — run summary ===")
    print(f"Verdict         : {verdict}")
    print(f"Drift share     : {drift_ratio*100:.2f}%")
    print(f"var_win(Ω′)     : {omega_var:.6e}")
    print(f"mean |dΛ′/dt|   : {lambda_speed:.6f}")
    print(f"Log             : ace_log.csv")
    print("==============================\n")

if __name__ == "__main__":
    main()
