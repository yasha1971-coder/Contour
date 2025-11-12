# ace_kernel_v04e.py
# ACE v0.4e — минимальное ядро с двумя хуками и nudges до интеграции шага

from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from typing import Callable, Dict, Optional, Any, Tuple
import numpy as np
import math
import os

# ---------------------------
# Параметры и состояние
# ---------------------------

@dataclass
class ACEParams:
    COUP_LA_TO_OM: float = 0.18   # λ → ω
    COUP_OM_TO_LA: float = 0.20   # ω → λ
    MEM_DECAY: float    = 0.38
    HYST: float         = 0.0065
    NOISE: float        = 0.012
    L_GAIN: float       = 1.3
    VAR_WINDOW: int     = 300

    # границы безопасности
    NOISE_MIN: float = 0.006
    NOISE_MAX: float = 0.020
    MEM_MIN:   float = 0.30
    MEM_MAX:   float = 0.45

@dataclass
class ACEState:
    t: int = 0
    omega: float = 0.0   # Ω′
    lam: float = 0.0     # Λ′
    last_omega: float = 0.0
    last_lam: float = 0.0

# ---------------------------
# Вспомогательные функции
# ---------------------------

def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    try:
        c = np.corrcoef(a, b)[0, 1]
        if math.isnan(c):
            return 0.0
        return float(c)
    except Exception:
        return 0.0

# ---------------------------
# Ядро ACE
# ---------------------------

class ACEEngineV04e:
    """
    Мини-ядро ACE со сценарием self-goal v0.5:
    - два хука:
        * on_pre_integrate(engine, state, metrics, nudges) -> Optional[dict]
        * on_post_step(engine, state, metrics) -> None
    - nudges (коррекции NOISE/MEM_DECAY) рассчитываются и ПРИМЕНЯЮТСЯ
      ДО интеграции шага (до обновления Ω′/Λ′).
    """

    def __init__(
        self,
        params: Optional[ACEParams] = None,
        seed: int = 42,
        hooks: Optional[Dict[str, Callable[..., Any]]] = None,
    ):
        self.params = params or ACEParams()
        self.state = ACEState()
        self.rng = np.random.default_rng(seed)

        # истории для метрик
        w = max(10, int(self.params.VAR_WINDOW))
        self.omega_hist = deque(maxlen=w)
        self.lam_hist = deque(maxlen=w)
        self.domega_hist = deque(maxlen=w)
        self.dlam_hist = deque(maxlen=w)

        # хуки (no-op по умолчанию)
        self.hooks: Dict[str, Callable[..., Any]] = {
            "on_pre_integrate": lambda *args, **kwargs: None,
            "on_post_step": lambda *args, **kwargs: None,
        }
        if hooks:
            self.hooks.update(hooks)

        # доли режимов (грубая оценка)
        self.regime_counts = {"drift": 0, "lock": 0, "iterate": 0}

    # ---------- публичный API ----------
    def set_hook(self, name: str, fn: Callable[..., Any]) -> None:
        if name not in self.hooks:
            raise KeyError(f"Unknown hook: {name}")
        self.hooks[name] = fn

    def get_params(self) -> dict:
        return asdict(self.params)

    def step(self) -> Dict[str, float]:
        """
        Один шаг симуляции:
          1) измеряем метрики
          2) вычисляем nudges (self-goal v0.5)
          3) ВЫЗЫВАЕМ on_pre_integrate и применяем nudges к параметрам
          4) интегрируем динамику Ω′/Λ′
          5) ВЫЗЫВАЕМ on_post_step
        """
        metrics = self._measure_metrics()

        # (2) — self-goal loop v0.5 → nudges
        nudges = self._compute_nudges(metrics)

        # (3) — ХУК ПЕРЕД ИНТЕГРАЦИЕЙ: дать внешнему коду скорректировать nudges/состояние
        try:
            hook_delta = self.hooks["on_pre_integrate"](self, self.state, metrics, dict(nudges))
            if isinstance(hook_delta, dict):
                nudges.update(hook_delta)
        except Exception as e:
            # Хук не должен ломать симуляцию
            print(f"[hook:on_pre_integrate] error: {e}")

        # применяем nudges к параметрам ДО интеграции шага
        self._apply_nudges(nudges)

        # (4) — интеграция шага (минимальная, стабильная)
        self._integrate()

        # обновляем истории
        self._push_histories()

        # финальные метрики после шага (по желанию можно вернуть pre или post)
        post_metrics = self._measure_metrics()

        # (5) — ХУК ПОСЛЕ ШАГА
        try:
            self.hooks["on_post_step"](self, self.state, dict(post_metrics))
        except Exception as e:
            print(f"[hook:on_post_step] error: {e}")

        return post_metrics

    def run(self, steps: int) -> Dict[str, float]:
        m = {}
        for _ in range(steps):
            m = self.step()
        return m

    # ---------- внутренняя механика ----------
    def _integrate(self) -> None:
        """Простая устойчивая динамика Ω′/Λ′ за один шаг."""
        p = self.params
        s = self.state

        # шум
        eta = float(self.rng.normal(0.0, 1.0))
        xi = float(self.rng.normal(0.0, 1.0))
        noise_term = p.NOISE * eta

        # dΩ′ и dΛ′ (минимальная, но не тривиальная связь)
        d_omega = -p.HYST * s.omega + p.COUP_LA_TO_OM * s.lam + noise_term
        d_lam = -p.MEM_DECAY * s.lam + p.L_GAIN * (s.omega + 0.25 * xi)

        s.last_omega = s.omega
        s.last_lam = s.lam

        # шаг интеграции (dt=1)
        s.omega += d_omega
        s.lam += d_lam
        s.t += 1

        # запись приращений
        self.domega_hist.append(d_omega)
        self.dlam_hist.append(d_lam)

        # обновление режима (грубая классификация)
        if abs(d_omega) + abs(d_lam) < 1e-4:
            self.regime_counts["lock"] += 1
        elif np.sign(d_omega) == np.sign(d_lam):
            self.regime_counts["drift"] += 1
        else:
            self.regime_counts["iterate"] += 1

    def _push_histories(self) -> None:
        self.omega_hist.append(self.state.omega)
        self.lam_hist.append(self.state.lam)

    def _measure_metrics(self) -> Dict[str, float]:
        w = max(3, len(self.omega_hist))
        if w < 3:
            # прогрев
            return {
                "var_win_omega": 0.0,
                "mean_abs_dlambda": 0.0,
                "corr_domega_dlambda": 0.0,
                "drift_share": 0.0,
                "lock_share": 0.0,
                "iterate_share": 0.0,
                "regime_transitions_per_1k": 0.0,
            }

        omega_arr = np.array(self.omega_hist, dtype=float)
        var_win = float(np.var(omega_arr, ddof=0))

        dω = np.array(self.domega_hist, dtype=float)
        dλ = np.array(self.dlam_hist, dtype=float)
        mean_abs_dlambda = float(np.mean(np.abs(dλ))) if len(dλ) else 0.0
        corr = safe_corr(dω[-w:], dλ[-w:])

        total = sum(self.regime_counts.values()) or 1
        drift_share = 100.0 * self.regime_counts["drift"] / total
        lock_share = 100.0 * self.regime_counts["lock"] / total
        iterate_share = 100.0 * self.regime_counts["iterate"] / total

        reg_transitions_per_1k = float(total) * (1000.0 / max(1, self.state.t))

        return {
            "var_win_omega": var_win,
            "mean_abs_dlambda": mean_abs_dlambda,
            "corr_domega_dlambda": corr,
            "drift_share": drift_share,
            "lock_share": lock_share,
            "iterate_share": iterate_share,
            "regime_transitions_per_1k": reg_transitions_per_1k,
        }

    # --- self-goal loop v0.5: вычисление nudges ---
    def _compute_nudges(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Возвращает предложенные изменения параметров ДО интеграции шага.
        Цель: держать var_win(Ω′) в окне [1e-6 .. 1e-3] и mean|dΛ′/dt| ≥ 1e-2.
        """
        var_min, var_max = 1e-6, 1e-3
        target_mean = 1e-2

        var_win = metrics["var_win_omega"]
        v_err_low = max(0.0, var_min - var_win)
        v_err_high = max(0.0, var_win - var_max)

        mean_abs_dlambda = metrics["mean_abs_dlambda"]
        need_more_breath = mean_abs_dlambda < target_mean

        nudges = {"dNOISE": 0.0, "dMEM": 0.0, "reason": []}

        # Если дисперсия слишком высокая → чуть зажать памятью
        if v_err_high > 0:
            k = 0.10  # мягко
            nudges["dMEM"] += k * (v_err_high / var_max)
            nudges["reason"].append("var_high→MEM+")
            # и немного уменьшить шум
            nudges["dNOISE"] -= 0.10 * (v_err_high / var_max)

        # Если дисперсия слишком низкая → немного отпустить память и дать шуму
        if v_err_low > 0:
            k = 0.08
            nudges["dMEM"] -= k * (v_err_low / var_min)
            nudges["dNOISE"] += 0.08 * (v_err_low / var_min)
            nudges["reason"].append("var_low→MEM- NOISE+")

        # Если «дыхание» слабое → дополнительно стимулируем шум
        if need_more_breath:
            nudges["dNOISE"] += 0.003
            nudges["reason"].append("breath_low→NOISE+")

        # Лёгкая нормализация: не позволяем чрезмерных шагов
        nudges["dNOISE"] = clip(nudges["dNOISE"], -0.004, +0.004)
        nudges["dMEM"] = clip(nudges["dMEM"], -0.04, +0.04)

        return nudges

    def _apply_nudges(self, nudges: Dict[str, float]) -> None:
        p = self.params
        if "dNOISE" in nudges:
            p.NOISE = clip(p.NOISE + float(nudges["dNOISE"]), p.NOISE_MIN, p.NOISE_MAX)
        if "dMEM" in nudges:
            p.MEM_DECAY = clip(p.MEM_DECAY + float(nudges["dMEM"]), p.MEM_MIN, p.MEM_MAX)

# ---------------------------
# Быстрый стенд
# ---------------------------

def quick_run(steps: int = 4000, seed: int = 42, hooks: Optional[Dict[str, Callable[..., Any]]] = None) -> Tuple[ACEEngineV04e, Dict[str, float]]:
    eng = ACEEngineV04e(ACEParams(), seed=seed, hooks=hooks)
    # прогрев истории (нулевые значения)
    for _ in range(eng.params.VAR_WINDOW // 2):
        eng._push_histories()
        eng.domega_hist.append(0.0)
        eng.dlam_hist.append(0.0)
    metrics = eng.run(steps)
    return eng, metrics

if __name__ == "__main__":
    # пример пользовательского хука:
    def pre_integrate(engine: ACEEngineV04e, state: ACEState, metrics: Dict[str, float], nudges: Dict[str, float]):
        # пример: мягкий приоритет «живого дыхания» — если mean|dΛ′/dt| низок, усиливаем шум сильнее
        if metrics.get("mean_abs_dlambda", 0.0) < 1e-2:
            nudges["dNOISE"] = clip(nudges.get("dNOISE", 0.0) + 0.001, -0.004, 0.004)
        return nudges  # можно вернуть изменённые nudges

    def post_step(engine: ACEEngineV04e, state: ACEState, metrics: Dict[str, float]):
        # лог каждые 1000 шагов (как пример использования)
        if state.t % 1000 == 0:
            print(f"[t={state.t}] var={metrics['var_win_omega']:.3e} "
                  f"mean|dΛ′/dt|={metrics['mean_abs_dlambda']:.5f} "
                  f"NOISE={engine.params.NOISE:.4f} MEM={engine.params.MEM_DECAY:.3f}")

    engine, m = quick_run(hooks={"on_pre_integrate": pre_integrate, "on_post_step": post_step})
    print("Final:", m)
