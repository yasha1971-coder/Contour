# ace_kernel_v04e.py
# ACE v0.4e-fix — ядро с поддержкой расширенных параметров
# Публичный интерфейс: ACEEngineV04e(params).run(n_steps)-> dict(metrics, traces)

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass

# ---------- утилиты -----------------------------------------------------------

def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Корреляция в [-1, 1] с защитой от NaN/константности."""
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if math.isnan(c) or math.isinf(c):
        return 0.0
    return float(np.clip(c, -1.0, 1.0))

def rolling_var(x: np.ndarray, win: int) -> float:
    """Вариация на последнем окне (ddof=0) с отсечкой снизу."""
    if x.size == 0:
        return 0.0
    win = int(max(2, min(win, x.size)))
    w = x[-win:]
    v = float(np.var(w))
    return max(v, 1e-12)

# ---------- параметры ---------------------------------------------------------

@dataclass
class ACEParams:
    COUP_LA_TO_OM: float = 0.19
    COUP_OM_TO_LA: float = 0.18
    MEM_DECAY:     float = 0.35
    HYST:          float = 0.006
    NOISE:         float = 0.014

    # новые поля (должны подтягиваться из JSON)
    VAR_WINDOW:       int   = 300
    DRIFT_HYST:       float = 0.022
    ANTI_STALL_BUMP:  float = 0.012
    L_GAIN:           float = 1.35

# ---------- движок ------------------------------------------------------------

class ACEEngineV04e:
    """
    Минимальный самодостаточный движок v0.4e:
    - Две главные величины: Ω′ (omega_p) и Λ′ (lambda_p_raw).
    - Лямбда проходит через усиление L_GAIN при оценке скорости.
    - Режимы: LOCK / DRIFT / ITERATE (по порогам на приращениях).
    - Анти-застойный «пинок» использует ANTI_STALL_BUMP.
    """

    def __init__(self, params: ACEParams, rng_seed: int = 42):
        self.p = params
        self.rng = np.random.default_rng(rng_seed)

        # состояния
        self.omega_p = 0.0
        self.lambda_p = 0.0

        # треки
        self.t_om = []
        self.t_la = []
        self.regimes = []  # 'L','D','I'

        # служебное
        self._stall_counter = 0

    def _step_dynamics(self):
        """Один шаг динамики Ω′ и Λ′."""
        p = self.p

        # Взаимосвязи (простая линейная связка с памятью и шумом)
        d_om = p.COUP_LA_TO_OM * self.lambda_p - p.MEM_DECAY * self.omega_p
        d_la = p.COUP_OM_TO_LA * self.omega_p - p.MEM_DECAY * self.lambda_p

        # шум для «дыхания»
        d_om += self.rng.normal(0.0, p.NOISE)
        d_la += self.rng.normal(0.0, p.NOISE)

        # обновление
        self.omega_p += d_om
        self.lambda_p += d_la

        return d_om, d_la

    def _classify_regime(self, d_om: float, d_la: float) -> str:
        """
        Классификация режима по «скорости» и гистерезису:
        - |dΩ′| и |dΛ′| малы → LOCK
        - средние → DRIFT
        - большие → ITERATE
        Границы симметричные, DRIFT_HYST расширяет середину.
        """
        p = self.p
        a = abs(d_om)
        b = abs(d_la)
        # базовые пороги
        lock_th = p.HYST
        iter_th = lock_th + p.DRIFT_HYST  # всё, что выше — iterate
        m = max(a, b)
        if m < lock_th:
            return 'L'
        if m > iter_th:
            return 'I'
        return 'D'

    def _anti_stall(self, d_om: float, d_la: float):
        """
        Если долго «тишина», пихнём систему маленьким импульсом.
        """
        p = self.p
        quiet = (abs(d_om) < p.HYST and abs(d_la) < p.HYST)
        self._stall_counter = self._stall_counter + 1 if quiet else 0
        fired = False
        if self._stall_counter >= 200:  # ~ адаптивная частота
            bump = p.ANTI_STALL_BUMP
            self.omega_p += bump * self.rng.choice([-1.0, 1.0])
            self.lambda_p += bump * self.rng.choice([-1.0, 1.0])
            self._stall_counter = 0
            fired = True
        return fired

    def run(self, n_steps: int = 6000) -> dict:
        bumps_fired = 0
        last_bump_t = -1

        for t in range(n_steps):
            d_om, d_la = self._step_dynamics()
            fired = self._anti_stall(d_om, d_la)
            if fired:
                bumps_fired += 1
                last_bump_t = t

            self.t_om.append(self.omega_p)
            self.t_la.append(self.lambda_p)
            self.regimes.append(self._classify_regime(d_om, d_la))

        om = np.asarray(self.t_om, dtype=float)
        la = np.asarray(self.t_la, dtype=float)

        # приращения (dt=1); скорость λ учитывает L_GAIN
        d_om = np.diff(om)
        d_la_raw = np.diff(la)
        d_la = self.p.L_GAIN * d_la_raw

        mean_abs_dla = float(np.mean(np.abs(d_la))) if d_la.size else 0.0
        corr = safe_corrcoef(d_om, d_la) if d_om.size and d_la.size else 0.0
        varw = rolling_var(om, self.p.VAR_WINDOW)

        # доли режимов
        if self.regimes:
            r = np.asarray(self.regimes)
            drift_share = float(np.mean(r == 'D') * 100.0)
            lock_share  = float(np.mean(r == 'L') * 100.0)
            iterate_share = float(np.mean(r == 'I') * 100.0)
        else:
            drift_share = lock_share = iterate_share = 0.0

        # переходы режимов
        transitions = int(np.sum((np.array(self.regimes[1:]) != np.array(self.regimes[:-1])))) if len(self.regimes) > 1 else 0
        transitions_per_1k = int(round(transitions * 1000.0 / max(1, n_steps)))

        metrics = {
            "drift_share": drift_share,
            "lock_share": lock_share,
            "iterate_share": iterate_share,
            "var_win": varw,
            "mean_abs_dla": mean_abs_dla,
            "corr": corr,
            "regime_transitions_per_1k": transitions_per_1k,
            "anti_stall_bumps": bumps_fired,
            "last_bump_t": last_bump_t,
        }

        traces = {
            "omega_p": om,
            "lambda_p": la,
            "d_om": d_om,
            "d_la": d_la,
            "regimes": self.regimes,
        }

        return {"metrics": metrics, "traces": traces}
