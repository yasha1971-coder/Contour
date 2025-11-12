# main.py
# ACE v0.4e-fix — лёгкий раннер с CLI, шоками и отчётом.
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ace_kernel_v04e import ACEEngineV04e, ACEParams, safe_corr, clip


# ---------------------------
# Утилиты
# ---------------------------

def load_params_from_json(path: str, base: Optional[ACEParams] = None) -> ACEParams:
    p = base or ACEParams()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(p, k):
                setattr(p, k, v)
    return p


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fmt_pct(x: float) -> str:
    return f"{x:6.2f} %"


def compute_final_metrics(engine: ACEEngineV04e) -> Dict[str, float]:
    # Используем истории из ядра
    omega = np.array(engine.omega_hist, dtype=float)
    dω = np.array(engine.domega_hist, dtype=float)
    dλ = np.array(engine.dlam_hist, dtype=float)

    var_win = float(np.var(omega, ddof=0)) if len(omega) else 0.0
    mean_abs_dlambda = float(np.mean(np.abs(dλ))) if len(dλ) else 0.0
    corr = safe_corr(dω, dλ)

    total = sum(engine.regime_counts.values()) or 1
    drift_share = 100.0 * engine.regime_counts["drift"] / total
    lock_share = 100.0 * engine.regime_counts["lock"] / total
    iterate_share = 100.0 * engine.regime_counts["iterate"] / total
    reg_transitions_per_1k = float(total) * (1000.0 / max(1, engine.state.t))

    return {
        "var_win": var_win,
        "mean_abs_dlambda": mean_abs_dlambda,
        "corr": corr,
        "drift_share": drift_share,
        "lock_share": lock_share,
        "iterate_share": iterate_share,
        "regime_transitions_per_1k": reg_transitions_per_1k,
    }


def alive_verdict(metrics: Dict[str, float]) -> Dict[str, str]:
    # Критерии ALIVE
    var_ok = 1e-6 <= metrics["var_win"] <= 1e-3
    mean_ok = metrics["mean_abs_dlambda"] >= 1e-2
    drift_ok = metrics["drift_share"] >= 20.0

    verdict = "ALIVE" if (var_ok and mean_ok and drift_ok) else "NOT ALIVE"
    checks = {
        "var": "[OK]" if var_ok else "[FAIL]",
        "mean": "[OK]" if mean_ok else "[FAIL]",
        "drift": "[OK]" if drift_ok else "[FAIL]",
        "verdict": verdict,
    }
    return checks


def parse_shock(s: str) -> Dict[str, str]:
    """
    Формат строки шока:
      "MEM_DECAY,3200,200,1.4"
      ^param   ^start ^duration ^factor
    """
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Bad shock '{s}'. Use 'PARAM,START,DURATION,FACTOR', e.g. 'MEM_DECAY,3200,200,1.4'"
        )
    param, start, duration, factor = parts
    return {
        "param": param,
        "start": int(start),
        "duration": int(duration),
        "factor": float(factor),
    }


# ---------------------------
# Хуки (примерные)
# ---------------------------

def make_pre_integrate_hook():
    # Мягкий приоритет "дыхания": если mean|dΛ′/dt| низок — чуть усиливаем шум.
    def _hook(engine: ACEEngineV04e, state, metrics: Dict[str, float], nudges: Dict[str, float]):
        if metrics.get("mean_abs_dlambda", 0.0) < 1e-2:
            nudges["dNOISE"] = clip(nudges.get("dNOISE", 0.0) + 0.001, -0.004, 0.004)
        return nudges  # можно вернуть изменённые nudges
    return _hook


def make_post_step_hook(log_every: int = 1000):
    def _hook(engine: ACEEngineV04e, state, metrics: Dict[str, float]):
        if state.t % log_every == 0:
            print(
                f"[t={state.t:6d}] var={metrics['var_win_omega']:.3e} "
                f"mean|dΛ′/dt|={metrics['mean_abs_dlambda']:.5f} "
                f"NOISE={engine.params.NOISE:.4f} MEM={engine.params.MEM_DECAY:.3f}"
            )
    return _hook


# ---------------------------
# Шок-менеджер
# ---------------------------

class ShockScheduler:
    """Простой планировщик шоков: параметр умножается на factor на duration шагов."""

    def __init__(self, shocks: List[Dict[str, str]]):
        # Нормализуем в список событий: start->apply, end->revert
        self.events = []
        for sh in shocks:
            start = sh["start"]
            end = sh["start"] + sh["duration"]
            self.events.append(("apply", start, sh["param"], sh["factor"]))
            self.events.append(("revert", end, sh["param"], sh["factor"]))
        self.events.sort(key=lambda e: e[1])
        self.active: Dict[str, float] = {}
        self.ptr = 0
        self.baseline: Dict[str, float] = {}

    def maybe_fire(self, engine: ACEEngineV04e):
        t = engine.state.t
        while self.ptr < len(self.events) and self.events[self.ptr][1] == t:
            kind, _, param, factor = self.events[self.ptr]
            self.ptr += 1
            if kind == "apply":
                if not self.baseline:
                    # запомним базовые значения один раз
                    self.baseline = asdict(engine.params)
                val = getattr(engine.params, param)
                self.active[param] = factor
                setattr(engine.params, param, val * factor)
                print(f"[SHOCK] apply {param} × {factor} at t={t}")
            elif kind == "revert":
                if param in self.active:
                    base = self.baseline.get(param, getattr(engine.params, param))
                    setattr(engine.params, param, base)
                    self.active.pop(param, None)
                    print(f"[SHOCK] revert {param} → {getattr(engine.params, param)} at t={t}")


# ---------------------------
# Основной запуск
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ACE Engine v0.4e-fix — simulation runner"
    )
    parser.add_argument("--steps", type=int, default=5000, help="Количество шагов симуляции")
    parser.add_argument("--seed", type=int, default=42, help="Сид ГПСЧ")
    parser.add_argument("--report-dir", type=str, default="ace_v04e_report", help="Папка для отчёта")
    parser.add_argument("--params", type=str, default="best_params.json", help="JSON с параметрами")
    parser.add_argument("--save-csv", action="store_true", help="Сохранить time-series в CSV")
    parser.add_argument(
        "--shock",
        action="append",
        default=[],
        help="Шок 'PARAM,START,DURATION,FACTOR' (можно указывать несколько раз)",
    )
    args = parser.parse_args()

    # Параметры
    params = load_params_from_json(args.params)

    # Инициализация ядра + хуки
    engine = ACEEngineV04e(
        params=params,
        seed=args.seed,
        hooks={
            "on_pre_integrate": make_pre_integrate_hook(),
            "on_post_step": make_post_step_hook(log_every=1000),
        },
    )

    # Прогрев историй для корректной статистики окна
    for _ in range(engine.params.VAR_WINDOW // 2):
        engine._push_histories()
        engine.domega_hist.append(0.0)
        engine.dlam_hist.append(0.0)

    # Планировщик шоков
    scheduler = ShockScheduler([parse_shock(s) for s in args.shock]) if args.shock else None

    # Запуск
    for _ in range(args.steps):
        if scheduler:
            scheduler.maybe_fire(engine)
        engine.step()

    # Метрики и вердикт
    metrics = compute_final_metrics(engine)
    checks = alive_verdict(metrics)

    # Отчёт
    ensure_dir(args.report_dir)
    summary_path = os.path.join(args.report_dir, "summary.txt")
    last_path = "last_report.txt"

    lines = []
    lines.append("===== ACE REPORT v0.4e-fix =====")
    lines.append(f"Drift share:          {fmt_pct(metrics['drift_share'])}")
    lines.append(f"Lock share:           {fmt_pct(metrics['lock_share'])}")
    lines.append(f"Iterate share:        {fmt_pct(metrics['iterate_share'])}")
    lines.append("")
    lines.append(f"var_win(Ω′):          {metrics['var_win']:.9e}")
    lines.append(f"mean |dΛ′/dt|:        {metrics['mean_abs_dlambda']:.5f}")
    lines.append(f"Corr(dΩ′, dΛ′):       {metrics['corr']:+.3f}")
    lines.append("")
    lines.append(f"Regime transitions /1k steps: {int(metrics['regime_transitions_per_1k']):d}")
    lines.append("")
    lines.append("Alive rule:")
    lines.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   {checks['var']}")
    lines.append(f"  mean |dΛ′/dt| > 1e-2:            {checks['mean']}")
    lines.append(f"  Drift share ≥ 20%:               {checks['drift']}")
    lines.append("")
    lines.append(f"VERDICT: [{checks['verdict']}]")
    lines.append("")
    lines.append("Notes:")
    lines.append(f"- params: COUP_LA_TO_OM={engine.params.COUP_LA_TO_OM}, "
                 f"COUP_OM_TO_LA={engine.params.COUP_OM_TO_LA}, "
                 f"MEM_DECAY={engine.params.MEM_DECAY}, "
                 f"HYST={engine.params.HYST}, "
                 f"NOISE={engine.params.NOISE}, "
                 f"L_GAIN={engine.params.L_GAIN}")
    if args.shock:
        lines.append("- shocks: " + " | ".join(args.shock))
    lines.append("================================")

    text = "\n".join(lines) + "\n"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(last_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)

    # (опционально) Сохраняем time-series
    if args.save_csv:
        try:
            import csv
            ts_path = os.path.join(args.report_dir, "data.csv")
            with open(ts_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "omega", "lambda", "domega", "dlambda"])
                # Восстанавливаем последовательность — ядро хранит только окно, это просто "хвост"
                t0 = max(0, engine.state.t - len(engine.omega_hist))
                for i, (ω, λ, dω, dλ) in enumerate(
                    zip(engine.omega_hist, engine.lam_hist, engine.domega_hist, engine.dlam_hist)
                ):
                    w.writerow([t0 + i + 1, ω, λ, dω, dλ])
            print(f"[saved] {ts_path}")
        except Exception as e:
            print(f"[warn] CSV save failed: {e}")


if __name__ == "__main__":
    main()
