# ace_kernel_v04e.py
# ACE Engine v0.4e-fix — безопасное ядро с хуками, клиппингом и анти-NaN
# Совместимо с прежними отчётами: summary.txt, data.csv
# Автор: ACE (Autonomous Contour Engine)

from __future__ import annotations
import json, math, csv, os, time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

# ---------- Константы безопасности ----------
EPS = 1e-12
CAP = 1e9
OMEGA_CAP = 1e6
LAMBDA_CAP = 1e6

# ---------- Утилиты безопасности ----------
def sanitize(x: float) -> float:
    return float(np.nan_to_num(x, nan=0.0, posinf=CAP, neginf=-CAP))

def safe_var(x: np.ndarray) -> float:
    if x.size < 2: return 0.0
    if np.allclose(x, x[0]): return 0.0
    v = float(np.var(x))
    if not np.isfinite(v): return 0.0
    return v

def safe_mean_abs(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    m = float(np.mean(np.abs(x)))
    if not np.isfinite(m): return 0.0
    return m

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 2: return 0.0
    a = a[-n:]; b = b[-n:]
    sa, sb = np.std(a), np.std(b)
    if sa < EPS or sb < EPS: return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c): return 0.0
    return c

def clip_params(p: Dict[str, float]) -> Dict[str, float]:
    """Жёсткий клиппинг допустимых диапазонов."""
    p = dict(p)
    p["NOISE"] = float(np.clip(p.get("NOISE", 0.012), 0.008, 0.020))
    p["MEM_DECAY"] = float(np.clip(p.get("MEM_DECAY", 0.38), 0.20, 0.45))
    p["HYST"] = float(np.clip(p.get("HYST", 0.0065), 0.004, 0.010))
    p["L_GAIN"] = float(np.clip(p.get("L_GAIN", 1.30), 1.10, 1.50))
    p["COUP_LA_TO_OM"] = float(np.clip(p.get("COUP_LA_TO_OM", 0.20), 0.15, 0.30))
    p["COUP_OM_TO_LA"] = float(np.clip(p.get("COUP_OM_TO_LA", 0.20), 0.15, 0.30))
    return p

def trajectory_guard(omega: float, lam: float) -> bool:
    """Ранний стоп: NaN/Inf/вылет по амплитуде."""
    if not (np.isfinite(omega) and np.isfinite(lam)): return False
    if abs(omega) > OMEGA_CAP or abs(lam) > LAMBDA_CAP: return False
    return True

# ---------- Структуры данных ----------
@dataclass
class ACEParams:
    NOISE: float = 0.012
    MEM_DECAY: float = 0.38
    HYST: float = 0.0065
    L_GAIN: float = 1.30
    COUP_LA_TO_OM: float = 0.20
    COUP_OM_TO_LA: float = 0.20

    @classmethod
    def from_json(cls, path: str) -> "ACEParams":
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        else:
            d = {}
        d = clip_params(d)
        return cls(**d)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clip_params(asdict(self)), f, ensure_ascii=False, indent=2)

# ---------- Хуки (до/после шага) ----------
PreStepHook = Callable[[int, "ACEState", ACEParams], None]
PostStepHook = Callable[[int, "ACEState", ACEParams], None]

# ---------- Состояние ----------
@dataclass
class ACEState:
    t: int = 0
    omega: float = 0.0     # Ω′
    lam: float = 0.0       # Λ′
    d_omega_hist: List[float] = None
    d_lambda_hist: List[float] = None
    regime_hist: List[int] = None  # 0: drift, 1: lock, 2: iterate

    def __post_init__(self):
        if self.d_omega_hist is None: self.d_omega_hist = []
        if self.d_lambda_hist is None: self.d_lambda_hist = []
        if self.regime_hist is None: self.regime_hist = []

# ---------- Ядро ACE ----------
class ACEEngineV04e:
    def __init__(self,
                 params: ACEParams,
                 var_window: int = 300,
                 report_dir: str = "ace_v04e_report"):
        self.params = params
        self.var_window = max(20, int(var_window))
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

        # Хуки
        self.pre_step_hooks: List[PreStepHook] = []
        self.post_step_hooks: List[PostStepHook] = []

        # «Толчки» (nudges) перед интеграцией, если сигнал залипает
        self.nudge_eps = 1e-6
        self.nudge_gain = 0.001

    # ---- API: регистрация хуков ----
    def add_pre_step_hook(self, fn: PreStepHook):
        self.pre_step_hooks.append(fn)

    def add_post_step_hook(self, fn: PostStepHook):
        self.post_step_hooks.append(fn)

    # ---- Один шаг динамики ----
    def step(self, st: ACEState):
        p = self.params = ACEParams(**clip_params(asdict(self.params)))

        # nudges если производные долго нулевые
        if st.d_omega_hist[-5:].count(0.0) == 5 if len(st.d_omega_hist) >= 5 else False:
            st.omega += self.nudge_gain
        if st.d_lambda_hist[-5:].count(0.0) == 5 if len(st.d_lambda_hist) >= 5 else False:
            st.lam += self.nudge_gain

        # --- PRE hooks ---
        for h in self.pre_step_hooks:
            try: h(st.t, st, p)
            except Exception: pass

        # --- интеграция (упрощённая устойчивая схема) ---
        # базовый шум (NOISE) + перекрёстные связи
        eta_o = np.random.normal(0.0, p.NOISE)
        eta_l = np.random.normal(0.0, p.NOISE)

        d_omega = -p.HYST * st.omega + p.COUP_LA_TO_OM * st.lam + eta_o
        d_lambda = -p.MEM_DECAY * st.lam + p.COUP_OM_TO_LA * st.omega + p.L_GAIN * np.tanh(st.omega) + eta_l

        # Эйлер с санитацией
        st.omega = sanitize(st.omega + d_omega)
        st.lam   = sanitize(st.lam   + d_lambda)

        # Страж траектории
        if not trajectory_guard(st.omega, st.lam):
            raise FloatingPointError("Unstable trajectory (guard tripped)")

        # Режим (примерная эвристика)
        regime = 0  # drift
        if abs(d_omega) < 0.5 * p.NOISE and abs(d_lambda) < 0.5 * p.NOISE:
            regime = 1  # lock (затухание)
        elif abs(d_omega) > 5 * p.NOISE or abs(d_lambda) > 5 * p.NOISE:
            regime = 2  # iterate (бурные колебания)

        st.d_omega_hist.append(float(d_omega))
        st.d_lambda_hist.append(float(d_lambda))
        st.regime_hist.append(regime)
        st.t += 1

        # --- POST hooks ---
        for h in self.post_step_hooks:
            try: h(st.t, st, p)
            except Exception: pass

    # ---- Основной прогон ----
    def run(self, steps: int = 4000) -> Dict[str, float]:
        st = ACEState()
        var_series: List[float] = []
        for _ in range(steps):
            try:
                self.step(st)
            except FloatingPointError:
                # аварийный выход: нестабильная комбинация
                break
            # скользящее окно для var_win(Ω′)
            var_series.append(st.omega)
            if len(var_series) > self.var_window:
                var_series.pop(0)

        # Метрики
        var_win = safe_var(np.array(var_series))
        mean_dlambda = safe_mean_abs(np.array(st.d_lambda_hist))
        corr = safe_corr(np.array(st.d_omega_hist[-self.var_window:]),
                         np.array(st.d_lambda_hist[-self.var_window:]))

        regimes = np.array(st.regime_hist, dtype=int) if st.regime_hist else np.array([0], dtype=int)
        N = max(1, regimes.size)
        drift_share = float(np.sum(regimes == 0)) / N * 100.0
        lock_share  = float(np.sum(regimes == 1)) / N * 100.0
        iterate_share = float(np.sum(regimes == 2)) / N * 100.0

        # Переходы режимов на 1k шагов
        trans = int(np.sum(regimes[1:] != regimes[:-1]))
        trans_per_1k = int(round(1000.0 * trans / max(1, N)))

        alive_var = (1e-6 <= var_win <= 1e-3)
        alive_vel = (mean_dlambda >= 1e-2)
        alive_drift = (drift_share >= 20.0)
        verdict = "ALIVE" if (alive_var and alive_vel and alive_drift) else "NOT ALIVE"

        metrics = {
            "Drift share": drift_share,
            "Lock share": lock_share,
            "Iterate share": iterate_share,
            "var_win": var_win,
            "mean_abs_dlambda_dt": mean_dlambda,
            "corr_domega_dlambda": corr,
            "regime_transitions_per_1k": trans_per_1k,
            "verdict": verdict,
        }

        self._save_report(metrics, st)
        return metrics

    # ---- Сохранение отчёта ----
    def _save_report(self, metrics: Dict[str, float], st: ACEState):
        # summary.txt
        lines = []
        lines.append("===== ACE REPORT v0.4e-fix =====")
        lines.append(f"Drift share:          {metrics['Drift share']:.2f} %")
        lines.append(f"Lock share:           {metrics['Lock share']:.2f} %")
        lines.append(f"Iterate share:        {metrics['Iterate share']:.2f} %\n")
        lines.append(f"var_win(Ω′):          {metrics['var_win']:.9e}")
        lines.append(f"mean |dΛ′/dt|:        {metrics['mean_abs_dlambda_dt']:.5f}")
        lines.append(f"Corr(dΩ′, dΛ′):       {metrics['corr_domega_dlambda']:+.3f}\n")
        lines.append(f"Regime transitions /1k steps: {metrics['regime_transitions_per_1k']}\n")
        lines.append("Alive rule:")
        lines.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   [{'OK' if 1e-6 <= metrics['var_win'] <= 1e-3 else 'FAIL'}]")
        lines.append(f"  mean |dΛ′/dt| > 1e-2:            [{'OK' if metrics['mean_abs_dlambda_dt'] >= 1e-2 else 'FAIL'}]")
        lines.append(f"  Drift share ≥ 20%:               [{'OK' if metrics['Drift share'] >= 20.0 else 'FAIL'}]\n")
        lines.append(f"VERDICT: [{metrics['verdict']}]\n")
        lines.append("Notes:")
        p = asdict(self.params)
        lines.append(f"- params: COUP_LA_TO_OM={p['COUP_LA_TO_OM']}, COUP_OM_TO_LA={p['COUP_OM_TO_LA']}, "
                     f"MEM_DECAY={p['MEM_DECAY']}, HYST={p['HYST']}, NOISE={p['NOISE']}, L_GAIN={p['L_GAIN']}")
        lines.append("================================\n")

        with open(os.path.join(self.report_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # data.csv (дельты)
        csv_path = os.path.join(self.report_dir, "data.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "dOmega", "dLambda"])
            for i, (do, dl) in enumerate(zip(
                np.array(st.d_omega_hist, dtype=float),
                np.array(st.d_lambda_hist, dtype=float))):
                w.writerow([i, sanitize(do), sanitize(dl)])

# ---- Утилита: быстрый запуск ----
def run_once(best_params_path: str = "best_params.json",
             steps: int = 4000,
             var_window: int = 300,
             report_dir: str = "ace_v04e_report") -> Dict[str, float]:
    params = ACEParams.from_json(best_params_path)
    eng = ACEEngineV04e(params, var_window=var_window, report_dir=report_dir)
    return eng.run(steps=steps)

if __name__ == "__main__":
    # Простой CLI
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--var_window", type=int, default=300)
    ap.add_argument("--params", type=str, default="best_params.json")
    ap.add_argument("--report_dir", type=str, default="ace_v04e_report")
    args = ap.parse_args()
    m = run_once(best_params_path=args.params,
                 steps=args.steps,
                 var_window=args.var_window,
                 report_dir=args.report_dir)
    print(json.dumps(m, ensure_ascii=False, indent=2))
