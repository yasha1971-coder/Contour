# auto_scan.py
# Перебор околостабильных параметров с ранней остановкой и отчётом.

from __future__ import annotations
import os, json, itertools, time
import numpy as np

from ace_kernel_v04e import (
    ACEParams, ACEEngineV04e, clip_params, trajectory_guard, run_once
)

REPORT_DIR = "ace_v04e_report"
BEST_PATH = "best_params.json"

# Узкий «околостабильный» грид
NOISE_GRID     = [0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015]
MEM_DECAY_GRID = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41]
HYST_GRID      = [0.0055, 0.0060, 0.0065, 0.0070, 0.0075]
L_GAIN_GRID    = [1.20, 1.25, 1.30, 1.35, 1.40]
COUP_LA_GRID   = [0.18, 0.19, 0.20, 0.21, 0.22]
COUP_OM_GRID   = [0.18, 0.19, 0.20, 0.21, 0.22]

def alive_score(metrics: dict) -> float:
    """Чем выше — тем лучше. Слегка штрафуем за слишком частые переходы."""
    if metrics["verdict"] != "ALIVE":
        return -1e9
    score = (
        0.5 * metrics["Drift share"] / 100.0 +
        0.3 * (1.0 - abs(np.log10(max(1e-12, metrics["var_win"])) + 3.0)/3.0) +
        0.2 * (metrics["mean_abs_dlambda_dt"] / 0.02)
    )
    score -= 0.001 * metrics["regime_transitions_per_1k"]
    return float(score)

def try_params(pdict: dict, steps: int, var_window: int) -> tuple[bool, dict]:
    """Запуск с защитой; возвращает (stable, metrics)."""
    params = ACEParams(**clip_params(pdict))
    eng = ACEEngineV04e(params, var_window=var_window, report_dir=REPORT_DIR)
    try:
        metrics = eng.run(steps=steps)
        # стабильность = без раннего исключения и без NaN
        stable = np.isfinite(metrics["var_win"]) and np.isfinite(metrics["mean_abs_dlambda_dt"])
    except Exception:
        stable = False
        metrics = {"verdict": "NOT ALIVE", "var_win": np.nan, "mean_abs_dlambda_dt": np.nan}
    return stable, metrics

def main(steps=4000, var_window=300, save_report="scan_report.txt"):
    combos = itertools.product(
        NOISE_GRID, MEM_DECAY_GRID, HYST_GRID, L_GAIN_GRID, COUP_LA_GRID, COUP_OM_GRID
    )
    best = None
    best_score = -1e12
    tested = 0
    alive_count = 0

    lines = []
    t0 = time.time()
    for noise, mem, hyst, lg, coup_la, coup_om in combos:
        tested += 1
        pdict = {
            "NOISE": noise,
            "MEM_DECAY": mem,
            "HYST": hyst,
            "L_GAIN": lg,
            "COUP_LA_TO_OM": coup_la,
            "COUP_OM_TO_LA": coup_om,
        }
        stable, metrics = try_params(pdict, steps=steps, var_window=var_window)
        sc = alive_score(metrics)
        if metrics.get("verdict") == "ALIVE":
            alive_count += 1
        if stable and sc > best_score:
            best_score = sc
            best = dict(pdict)

        lines.append(
            f"{tested:05d} | {pdict} | {metrics.get('verdict','ERR'):>9} | "
            f"var={metrics.get('var_win')} mean|dΛ|={metrics.get('mean_abs_dlambda_dt')}"
        )

    dt = time.time() - t0
    os.makedirs(REPORT_DIR, exist_ok=True)
    rep_path = os.path.join(REPORT_DIR, save_report)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("ACE auto-scan report\n")
        f.write(f"tested: {tested}, alive: {alive_count}, time: {dt:.1f}s\n")
        f.write("\n".join(lines))

    # обновляем best_params.json только если нашли стабильный ALIVE
    if best is not None:
        # Финальная проверка
        stable, met = try_params(best, steps=steps, var_window=var_window)
        if stable and met["verdict"] == "ALIVE":
            with open(BEST_PATH, "w", encoding="utf-8") as bf:
                json.dump(clip_params(best), bf, ensure_ascii=False, indent=2)
            print("[OK] best_params.json updated:", best)
        else:
            print("[WARN] Best combo not ALIVE on recheck — best_params.json not updated")
    else:
        print("[WARN] No stable ALIVE combos found")

    print(f"[REPORT] {rep_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--var_window", type=int, default=300)
    ap.add_argument("--report", type=str, default="scan_report.txt")
    args = ap.parse_args()
    main(steps=args.steps, var_window=args.var_window, save_report=args.report)
