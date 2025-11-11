# main.py — ACE v0.4e-fix runner with CLI & shock injection
# ---------------------------------------------------------
# Usage examples:
#   python main.py
#   python main.py --steps 6000
#   python main.py --shock '{"t":2000,"dur":300,"param":"NOISE","factor":2.0}'
#
# Outputs (in ace_v04e_report/):
#   - data.csv               (time series)
#   - evolution.png          (Omega', Lambda', regime)
#   - deltas.png             (dOmega', dLambda')
#   - summary.txt            (formatted report)
# and a copy: last_report.txt in project root.

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ace_kernel_v04e import ACEEngineV04e  # ядро

# ----------------------------- IO UTILS -----------------------------

ROOT = Path(".")
OUT = ROOT / "ace_v04e_report"
OUT.mkdir(exist_ok=True)

SUMMARY = OUT / "summary.txt"
LAST = ROOT / "last_report.txt"
CSV = OUT / "data.csv"
EVOL = OUT / "evolution.png"
DELT = OUT / "deltas.png"

PARAMS_FILE = ROOT / "best_params.json"

def load_params() -> Dict[str, float]:
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"best_params.json not found at {PARAMS_FILE.resolve()}")
    with PARAMS_FILE.open("r", encoding="utf-8") as f:
        p = json.load(f)
    # нормализуем основные ключи; допускаем расширенный словарь
    defaults = {
        "COUP_LA_TO_OM": 0.20,
        "COUP_OM_TO_LA": 0.20,
        "MEM_DECAY": 0.35,
        "HYST": 0.006,
        "NOISE": 0.014,
        "VAR_WINDOW": 300
    }
    for k, v in defaults.items():
        p.setdefault(k, v)
    return p

# ----------------------------- METRICS ------------------------------

ALIVE_VAR_MIN = 1e-6
ALIVE_VAR_MAX = 1e-3
ALIVE_VEL_MIN = 1e-2
ALIVE_DRIFT_MIN = 0.20

def rolling_var(x: np.ndarray, w: int) -> np.ndarray:
    """скользящая дисперсия с численной отсечкой"""
    if w <= 1:
        return np.zeros_like(x)
    x = np.asarray(x, dtype=float)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    cumsum2 = np.cumsum(np.insert(x*x, 0, 0.0))
    n = np.arange(len(x))+1
    # простая реализация окна: дополним начальными значениями
    var = np.empty_like(x)
    for i in range(len(x)):
        j0 = max(0, i-w+1)
        nwin = i - j0 + 1
        s = cumsum[i+1]-cumsum[j0]
        s2 = cumsum2[i+1]-cumsum2[j0]
        m = s/nwin
        v = max(0.0, s2/nwin - m*m)
        var[i] = v
    # численная отсечка анти-нулей
    var[var < 1e-12] = 0.0
    return var

def classify_regime(var_val: float) -> int:
    """
    0 = Lock, 1 = Drift, 2 = Iterate
    Простая евристика по окну варьирования Ω′
    """
    if var_val < 1e-8:
        return 0
    if ALIVE_VAR_MIN <= var_val <= ALIVE_VAR_MAX:
        return 1
    return 2

def compute_report(
    om: np.ndarray,
    la: np.ndarray,
    params: Dict[str, float],
    anti_bumps: int,
    last_bump_t: int,
    var_window: int
) -> Tuple[str, Dict[str, Any]]:
    dOm = np.diff(om, prepend=om[0])
    dLa = np.diff(la, prepend=la[0])

    var_win_series = rolling_var(om, var_window)
    # метрики
    regimes = np.array([classify_regime(v) for v in var_win_series], dtype=int)
    drift_share = float(np.mean(regimes == 1))
    lock_share  = float(np.mean(regimes == 0))
    iter_share  = float(np.mean(regimes == 2))

    var_win = float(np.mean(var_win_series[-var_window:]))  # усредняем «по окну» в конце
    mean_abs_dL = float(np.mean(np.abs(dLa)))

    # корреляция — используем numpy (фикс прежней «поломки»)
    if np.all(dOm == dOm[0]) or np.all(dLa == dLa[0]):
        corr = 0.0
    else:
        c = np.corrcoef(dOm, dLa)
        corr = float(c[0, 1]) if np.isfinite(c[0, 1]) else 0.0

    # переходы (сколько раз режим сменился) на 1000 шагов
    switches = int(np.sum(regimes[1:] != regimes[:-1]))
    rt_per_1k = float(switches) / (len(regimes) / 1000.0)

    # правила ALIVE
    chk_var  = ALIVE_VAR_MIN <= var_win <= ALIVE_VAR_MAX
    chk_vel  = mean_abs_dL > ALIVE_VEL_MIN
    chk_drift = drift_share >= ALIVE_DRIFT_MIN

    verdict = "[ALIVE]" if (chk_var and chk_vel and chk_drift) else "[NOT ALIVE]"

    # формируем текст отчёта
    lines = []
    lines.append("===== ACE REPORT v0.4e-fix =====\n")
    lines.append(f"Drift share:          {drift_share*100:5.2f} %\n")
    lines.append(f"Lock share:           {lock_share*100:5.2f} %\n")
    lines.append(f"Iterate share:        {iter_share*100:5.2f} %\n\n")
    lines.append(f"var_win(Ω′):          {var_win:.6e}\n")
    lines.append(f"mean |dΛ′/dt|:        {mean_abs_dL:.5f}\n")
    lines.append(f"Corr(dΩ′, dΛ′):       {corr:+.3f}\n\n")
    lines.append(f"Regime transitions /1k steps: {int(round(rt_per_1k))}\n\n")
    lines.append("Alive rule:\n")
    lines.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   [{'OK' if chk_var else 'FAIL'}]\n")
    lines.append(f"  mean |dΛ′/dt| > 1e-2:            [{'OK' if chk_vel else 'FAIL'}]\n")
    lines.append(f"  Drift share ≥ 20%:               [{'OK' if chk_drift else 'FAIL'}]\n\n")
    lines.append(f"VERDICT: {verdict}\n\n")
    lines.append("Notes:\n")
    lines.append(f"- anti-stall bumps: {anti_bumps} fired\n")
    lines.append(f"- last bump at t = {last_bump_t}\n")
    # компактно распечатаем параметры
    par_line = ", ".join([f"{k}={params[k]}" for k in ["COUP_LA_TO_OM","COUP_OM_TO_LA","MEM_DECAY","HYST","NOISE"] if k in params])
    lines.append(f"- params: {par_line}\n")
    lines.append("================================\n")

    metrics = {
        "drift_share": drift_share,
        "lock_share": lock_share,
        "iterate_share": iter_share,
        "var_win": var_win,
        "mean_abs_dL_dt": mean_abs_dL,
        "corr": corr,
        "regime_transitions": rt_per_1k,
        "verdict": verdict.strip("[]"),
    }
    return "".join(lines), metrics

# ----------------------------- SHOCK -------------------------------

def apply_shock_if_needed(engine: ACEEngineV04e, t: int, state: Dict[str, Any]) -> None:
    """
    state: dict with keys: shock_cfg, shock_active, shock_backup, shock_end
    """
    scfg = state.get("shock_cfg")
    if not scfg:
        return

    if (not state["shock_active"]) and t == int(scfg["t"]):
        pname = scfg["param"]
        if not hasattr(engine, "params") or pname not in engine.params:
            raise KeyError(f"Param {pname} not found in engine.params")
        state["shock_backup"] = engine.params[pname]
        engine.params[pname] = float(scfg["factor"]) * float(state["shock_backup"])
        state["shock_active"] = True
        state["shock_end"] = t + int(scfg["dur"])

    if state["shock_active"] and t == state["shock_end"]:
        engine.params[scfg["param"]] = state["shock_backup"]
        state["shock_active"] = False

# ----------------------------- RUNNER ------------------------------

def run(steps: int, shock_cfg: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    params = load_params()
    var_window = int(params.get("VAR_WINDOW", 300))

    engine = ACEEngineV04e(params)

    # состояния наблюдения
    om_list = []
    la_list = []

    # сведения о bump'ах из ядра (если нет — будем писать 0 / -1)
    anti_bumps = getattr(engine, "anti_bumps", 0)
    last_bump_t = getattr(engine, "last_bump_t", -1)

    # init shock state
    sstate = {
        "shock_cfg": shock_cfg,
        "shock_active": False,
        "shock_backup": None,
        "shock_end": -1,
    }

    # прогон
    for t in range(steps):
        # включаем/выключаем шок
        if shock_cfg:
            apply_shock_if_needed(engine, t, sstate)

        # один шаг ядра
        # ожидаем, что step() возвращает (OmegaPrime, LambdaPrime)
        om_p, la_p = engine.step()
        om_list.append(om_p)
        la_list.append(la_p)

        # если ядро само считает anti_bumps/last_bump_t — обновим
        anti_bumps = getattr(engine, "anti_bumps", anti_bumps)
        last_bump_t = getattr(engine, "last_bump_t", last_bump_t)

    om = np.asarray(om_list, dtype=float)
    la = np.asarray(la_list, dtype=float)

    # сохраняем CSV (минимальный набор; при желании расширь)
    dOm = np.diff(om, prepend=om[0])
    dLa = np.diff(la, prepend=la[0])
    var_series = rolling_var(om, var_window)
    regimes = np.array([classify_regime(v) for v in var_series], dtype=int)

    with CSV.open("w", encoding="utf-8") as f:
        f.write("t,omega,lambda,dOmega,dLambda,var_win,regime\n")
        for i in range(len(om)):
            f.write(f"{i},{om[i]},{la[i]},{dOm[i]},{dLa[i]},{var_series[i]},{regimes[i]}\n")

    # графики
    try:
        plt.figure(figsize=(10,4))
        plt.plot(om, label="Ω′")
        plt.plot(la, label="Λ′", alpha=0.8)
        plt.title("ACE Evolution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EVOL, dpi=160)
        plt.close()

        plt.figure(figsize=(10,4))
        plt.plot(dOm, label="dΩ′")
        plt.plot(dLa, label="dΛ′", alpha=0.8)
        plt.title("ACE Deltas")
        plt.legend()
        plt.tight_layout()
        plt.savefig(DELT, dpi=160)
        plt.close()
    except Exception:
        # headless среда — не критично
        pass

    report_text, metrics = compute_report(
        om=om,
        la=la,
        params=params,
        anti_bumps=anti_bumps,
        last_bump_t=last_bump_t,
        var_window=var_window
    )

    # если был шок — добавим в хвост отчёта
    if shock_cfg:
        report_text += "\nShock applied:\n"
        report_text += f"- at t = {shock_cfg['t']}, duration = {shock_cfg['dur']}\n"
        report_text += f"- param = {shock_cfg['param']}, factor = {shock_cfg['factor']}\n"

    SUMMARY.write_text(report_text, encoding="utf-8")
    LAST.write_text(report_text, encoding="utf-8")

    return om, la, metrics

# ----------------------------- CLI --------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACE v0.4e-fix runner with shock support")
    p.add_argument("--steps", type=int, default=6000, help="simulation steps")
    p.add_argument("--shock", type=str, default="", help='JSON: {"t":2000,"dur":300,"param":"NOISE","factor":2.0}')
    return p.parse_args()

def main():
    args = parse_args()
    shock_cfg = None
    if args.shock:
        shock_cfg = json.loads(args.shock)
        for k in ("t","dur","param","factor"):
            if k not in shock_cfg:
                raise ValueError(f"shock missing key: {k}")

    om, la, metrics = run(steps=int(args.steps), shock_cfg=shock_cfg)

    # краткий вывод в консоль
    print(SUMMARY.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
