# auto_reg_v05.py — ACE v0.5 self-goal loop (NOISE & MEM_DECAY auto-tuner)
# ------------------------------------------------------------------------
# Примеры:
#   python auto_reg_v05.py
#   python auto_reg_v05.py --steps 4000 --cycles 12
#   python auto_reg_v05.py --target-mid 3e-4 --save-every 2
#
# Идея: короткими циклами запускаем ACE, считаем var_win и mean|dΛ′/dt|,
# затем корректируем NOISE/MEM_DECAY по простому адаптивному правилу.
# Сохраняем best_params.json при улучшении и финальные параметры в конце.

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ace_kernel_v04e import ACEEngineV04e

ROOT = Path(".")
OUT = ROOT / "ace_v04e_report"
OUT.mkdir(exist_ok=True)
PARAMS_FILE = ROOT / "best_params.json"
LOGFILE = OUT / "autotune_v05.log"

# --- целевые правила ALIVE (как в твоём репорте) ---
ALIVE_VAR_MIN = 1e-6
ALIVE_VAR_MAX = 1e-3
ALIVE_VEL_MIN = 1e-2
ALIVE_DRIFT_MIN = 0.20

# геометрический центр коридора можно смещать (по умолчанию ~3e-4)
DEFAULT_TARGET_MID = 3e-4

# --- сервис: загрузка/сохранение параметров ---
def load_params() -> Dict[str, float]:
    with PARAMS_FILE.open("r", encoding="utf-8") as f:
        p = json.load(f)
    # дефолты, если чего-то нет
    p.setdefault("COUP_LA_TO_OM", 0.20)
    p.setdefault("COUP_OM_TO_LA", 0.20)
    p.setdefault("MEM_DECAY", 0.35)
    p.setdefault("HYST", 0.006)
    p.setdefault("NOISE", 0.014)
    p.setdefault("VAR_WINDOW", 300)
    return p

def save_params(p: Dict[str, float]) -> None:
    with PARAMS_FILE.open("w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)

# --- метрики/классификация (совместимо с твоим main.py) ---
def rolling_var(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return np.zeros_like(x)
    x = np.asarray(x, dtype=float)
    var = np.empty_like(x)
    s, s2 = 0.0, 0.0
    buf = []
    for i, xi in enumerate(x):
        buf.append(xi)
        s += xi
        s2 += xi*xi
        if len(buf) > w:
            x0 = buf.pop(0)
            s -= x0
            s2 -= x0*x0
        n = len(buf)
        m = s/n
        v = max(0.0, s2/n - m*m)
        var[i] = v if v >= 1e-12 else 0.0
    return var

def run_short(engine: ACEEngineV04e, steps: int, var_window: int) -> Dict[str, float]:
    om, la = [], []
    for _ in range(steps):
        om_p, la_p = engine.step()
        om.append(om_p); la.append(la_p)
    om = np.asarray(om); la = np.asarray(la)
    dOm = np.diff(om, prepend=om[0]); dLa = np.diff(la, prepend=la[0])

    var_series = rolling_var(om, var_window)
    var_win = float(np.mean(var_series[-var_window:]))
    mean_abs_dL = float(np.mean(np.abs(dLa)))
    # корреляция dΩ′—dΛ′ (для наблюдения)
    if np.all(dOm == dOm[0]) or np.all(dLa == dLa[0]):
        corr = 0.0
    else:
        corr = float(np.corrcoef(dOm, dLa)[0,1])
    # доля дрейфа по простому правилу (как у тебя)
    def reg(v):
        if v < 1e-8: return 0
        if ALIVE_VAR_MIN <= v <= ALIVE_VAR_MAX: return 1
        return 2
    regimes = np.array([reg(v) for v in var_series])
    drift_share = float(np.mean(regimes == 1))
    return dict(var_win=var_win, mean_abs_dL=mean_abs_dL, drift_share=drift_share, corr=corr)

# --- регулятор v0.5 ---
def adjust(params: Dict[str, float],
           metrics: Dict[str, float],
           target_mid: float,
           k_noise: float,
           k_mem: float,
           bounds_noise: Tuple[float,float],
           bounds_mem: Tuple[float,float]) -> Dict[str, float]:
    """
    Простейший 2-канальный регулятор:
      e_var = log(var_win / target_mid)  (симметричный по относительной ошибке)
      если e_var > 0 (слишком «грубо»): MEM_DECAY ↑, NOISE ↓
      если e_var < 0 (слишком «гладко»): MEM_DECAY ↓, NOISE ↑
      если mean|dΛ′/dt| ниже порога — немного подталкиваем NOISE вверх
    """
    var_win = metrics["var_win"]
    vel = metrics["mean_abs_dL"]

    # лог-ошибка по дисперсии (стабильнее линейной)
    e_var = float(np.log(max(var_win, 1e-16) / target_mid))

    noise = params["NOISE"]
    memd  = params["MEM_DECAY"]

    # базовые изменения
    noise += (-np.sign(e_var)) * k_noise * min(3.0, abs(e_var))   # var выше цели → шум вниз
    memd  += ( np.sign(e_var)) * k_mem   * min(3.0, abs(e_var))   # var выше цели → память вверх

    # поддержка «дыхания» Λ′: если скорость низка — слегка подкинуть шума
    if vel < ALIVE_VEL_MIN:
        noise += 0.5 * k_noise

    # мягкое ограничение
    noise = float(np.clip(noise, *bounds_noise))
    memd  = float(np.clip(memd,  *bounds_mem))

    params["NOISE"] = noise
    params["MEM_DECAY"] = memd
    return params

def alive_score(metrics: Dict[str, float]) -> Tuple[bool, float]:
    """возвращает (is_alive, score) для сравнения прогресса"""
    v_ok = ALIVE_VAR_MIN <= metrics["var_win"] <= ALIVE_VAR_MAX
    d_ok = metrics["drift_share"] >= ALIVE_DRIFT_MIN
    m_ok = metrics["mean_abs_dL"] >  ALIVE_VEL_MIN
    ok = v_ok and d_ok and m_ok
    # чем ниже |log(var/center)| и выше vel — тем лучше
    center = (ALIVE_VAR_MIN*ALIVE_VAR_MAX) ** 0.5
    score = -abs(np.log(max(metrics["var_win"],1e-16) / center)) + 0.3*(metrics["mean_abs_dL"])
    if not d_ok: score -= 0.3
    return ok, float(score)

def main():
    ap = argparse.ArgumentParser(description="ACE v0.5 self-goal (NOISE & MEM_DECAY) tuner")
    ap.add_argument("--steps", type=int, default=3500, help="steps per cycle")
    ap.add_argument("--cycles", type=int, default=10, help="max tuning cycles")
    ap.add_argument("--target-mid", type=float, default=DEFAULT_TARGET_MID, help="target var_win mid inside [1e-6..1e-3]")
    ap.add_argument("--k-noise", type=float, default=0.002, help="learning rate for NOISE")
    ap.add_argument("--k-mem", type=float, default=0.02, help="learning rate for MEM_DECAY")
    ap.add_argument("--bounds-noise", type=float, nargs=2, default=(0.003, 0.04))
    ap.add_argument("--bounds-mem", type=float, nargs=2, default=(0.12, 0.80))
    ap.add_argument("--save-every", type=int, default=1, help="persist params every N cycles")
    ap.add_argument("--silent", action="store_true", help="less console output")
    args = ap.parse_args()

    params = load_params()
    var_window = int(params.get("VAR_WINDOW", 300))

    best_score = -1e9
    best_params = params.copy()
    best_metrics = None

    log_lines = []
    def log(s: str):
        if not args.silent:
            print(s)
        log_lines.append(s+"\n")

    log("=== v0.5 self-goal loop start ===")
    log(f"init params: NOISE={params['NOISE']:.5f}, MEM_DECAY={params['MEM_DECAY']:.3f}, VAR_WINDOW={var_window}")
    log(f"target mid var = {args.target_mid:g}, ALIVE vel > {ALIVE_VEL_MIN:g}, drift ≥ {ALIVE_DRIFT_MIN:.0%}")

    for c in range(1, args.cycles+1):
        # новый engine с текущими params
        engine = ACEEngineV04e(params.copy())
        metrics = run_short(engine, steps=args.steps, var_window=var_window)
        ok, score = alive_score(metrics)

        log(f"[cycle {c:02d}] var_win={metrics['var_win']:.3e} |dΛ′/dt|={metrics['mean_abs_dL']:.5f} "
            f"drift={metrics['drift_share']:.1%} corr={metrics['corr']:+.3f}  ok={ok} score={score:+.4f}")

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_metrics = metrics.copy()
            # периодически сохраняем прогресс
            if c % args.save_every == 0:
                save_params(best_params)

        if ok:
            log("→ ALIVE reached — committing params and exit.")
            save_params(params)
            break

        # подстройка
        params = adjust(
            params=params,
            metrics=metrics,
            target_mid=args.target_mid,
            k_noise=args.k_noise,
            k_mem=args.k_mem,
            bounds_noise=tuple(args.bounds_noise),
            bounds_mem=tuple(args.bounds_mem),
        )

        log(f"        tune → NOISE={params['NOISE']:.5f}, MEM_DECAY={params['MEM_DECAY']:.3f}")

    # финализация: если не ALIVE — сохранить лучшее
    if best_metrics is not None:
        save_params(best_params)
        log("saved best params achieved during loop.")

    # лог в файл
    with LOGFILE.open("w", encoding="utf-8") as f:
        f.writelines(log_lines)

    # компактный итог
    print("\n=== v0.5 summary ===")
    print(f"best score: {best_score:+.4f}")
    if best_metrics:
        print(f"best var_win={best_metrics['var_win']:.3e} |dΛ′/dt|={best_metrics['mean_abs_dL']:.5f} "
              f"drift={best_metrics['drift_share']:.1%}")
    print(f"saved → {PARAMS_FILE}")

if __name__ == "__main__":
    main()
