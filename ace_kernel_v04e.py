import json, os
import numpy as np

class ACEngineV04eFix:
    def __init__(self, params, steps=6000, seed=42):
        self.p = params
        self.steps = steps
        self.rng = np.random.default_rng(seed)

        # основные состояния
        self.Omega_prime = 1.0
        self.Lambda_prime = 0.15

        # память и окна
        self.mem_omega = 0.0
        self.mem_lambda = 0.0
        self.omega_history = []
        self.lambda_history = []
        self.domega_history = []
        self.dlambda_history = []
        self.regime_history = []
        self.last_bump_t = -10**9
        self.t = 0

        # счётчики
        self.bump_count = 0
        self.regime = "Lock"
        self.regime_switches = 0

    # --- утилиты ---
    def _rolling_var_omega(self):
        win = int(self.p["VAR_WINDOW"])
        floor = float(self.p["OMEGA_VAR_FLOOR"])
        if len(self.omega_history) < win:
            return floor
        v = np.var(self.omega_history[-win:])
        return max(v, floor)

    def _classify_regime(self, dphi):
        low, high, h = self.p["DRIFT_LOW"], self.p["DRIFT_HIGH"], self.p["DRIFT_HYST"]
        lo, hi = low, high
        if self.regime == "Drift":
            lo -= h
            hi += h
        new_regime = "Drift"
        if dphi < lo: new_regime = "Lock"
        elif dphi > hi: new_regime = "Iterate"
        if new_regime != self.regime:
            self.regime_switches += 1
            self.regime = new_regime
        return new_regime

    # --- один шаг динамики ---
    def step(self):
        p = self.p
        # взаимная связь (слабая, но постоянная)
        if len(self.lambda_history) >= 5:
            lam_ref = np.mean(self.lambda_history[-5:])
        else:
            lam_ref = self.Lambda_prime
        if len(self.omega_history) >= 5:
            omg_ref = np.mean(self.omega_history[-5:])
        else:
            omg_ref = self.Omega_prime

        self.Omega_prime += p["COUP_LA_TO_OM"] * (self.Lambda_prime - lam_ref) * 1e-2
        self.Lambda_prime += p["COUP_OM_TO_LA"] * (self.Omega_prime - omg_ref) * 1e-2

        # шумовой пол
        eps = p["NOISE"]
        self.Omega_prime += eps * self.rng.normal() * 1e-2
        self.Lambda_prime += eps * self.rng.normal() * 1e-2

        # лёгкий гистерезис на Омеге: тянем к памяти
        self.mem_omega = p["MEM_DECAY"] * self.mem_omega + (1 - p["MEM_DECAY"]) * self.Omega_prime
        self.mem_lambda = p["MEM_DECAY"] * self.mem_lambda + (1 - p["MEM_DECAY"]) * self.Lambda_prime
        self.Omega_prime += (self.mem_lambda - self.mem_omega) * 1e-3  # мягкая подстройка

        # производные
        if self.omega_history:
            dOm = self.Omega_prime - self.omega_history[-1]
            dLa = self.Lambda_prime - self.lambda_history[-1]
        else:
            dOm = dLa = 0.0

        self.omega_history.append(self.Omega_prime)
        self.lambda_history.append(self.Lambda_prime)
        self.domega_history.append(dOm)
        self.dlambda_history.append(dLa)

        # анти-стопор
        var_win = self._rolling_var_omega()
        if (var_win <= p["ANTI_STALL_EPS"]) and ((self.t - self.last_bump_t) > p["ANTI_STALL_COOLDOWN"]):
            bump = p["ANTI_STALL_BUMP"] * (1.0 + 0.25 * self.rng.normal())
            self.Omega_prime += np.sign(self.rng.normal()) * bump
            self.bump_count += 1
            self.last_bump_t = self.t
            # пересчитать производную после пинка
            if len(self.omega_history) >= 1:
                dOm = self.Omega_prime - self.omega_history[-1]
                self.domega_history[-1] = dOm

        # классификация режима
        dphi = abs(self.Omega_prime - self.Lambda_prime)
        regime = self._classify_regime(dphi)
        self.regime_history.append(regime)

        self.t += 1

    def run(self):
        for _ in range(self.steps):
            self.step()

        # метрики
        arr_reg = np.array(self.regime_history)
        drift_share = np.mean(arr_reg == "Drift")
        lock_share  = np.mean(arr_reg == "Lock")
        iter_share  = np.mean(arr_reg == "Iterate")

        var_win = self._rolling_var_omega()
        dl = np.array(self.dlambda_history)
        mean_abs_dlambda = float(np.mean(np.abs(dl)))

        # безопасная корреляция
        dO = np.array(self.domega_history)
        dL = np.array(self.dlambda_history)
        if (np.std(dO) > 0) and (np.std(dL) > 0):
            corr = float(np.corrcoef(dO, dL)[0,1])
        else:
            corr = 0.0

        metrics = {
            "drift_share": float(drift_share),
            "lock_share": float(lock_share),
            "iter_share": float(iter_share),
            "var_win": float(var_win),
            "mean_abs_dlambda": float(mean_abs_dlambda),
            "corr": float(corr),
            "regime_switches": int(self.regime_switches),
            "bumps": int(self.bump_count),
            "last_bump_t": int(self.last_bump_t)
        }
        return metrics

def load_params(path="best_params.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
