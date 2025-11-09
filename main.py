#!/usr/bin/env python3
# ACE v0.4e-fix — robust main
# - Читает best_params.json с дефолтами
# - Вызывает ядро (run_simulation(...) или ACEngine_v04e(...).run(...))
# - Пишет отчет в ace_v04e_report/summary.txt + дубликат last_report.txt
# - Не падает, если каких-то ключей нет

from __future__ import annotations
import os, json, math, shutil, datetime as dt
from typing import Dict, Any

REPORT_DIR = "ace_v04e_report"
SUMMARY_PATH = os.path.join(REPORT_DIR, "summary.txt")
LAST_REPORT = "last_report.txt"
STEPS = 6000
SEED = 42

# ---- 1) Схема параметров + дефолты (безопасно покрывают пропуски)
REQUIRED_DEFAULTS: Dict[str, Any] = {
    "COUP_LA_TO_OM": 0.21,
    "COUP_OM_TO_LA": 0.20,
    "MEM_DECAY":     0.22,
    "HYST":          0.005,
    "NOISE":         0.02,

    # вспомогательные (используются в ядре/отчетах)
    "VAR_WINDOW":       220,    # окно скользящей дисперсии для Ω′
    "DRIFT_HYST":       0.022,  # гистерезис детекции режима
    "ANTI_STALL_BUMP":  0.012,  # «пинок» против залипания
    "L_GAIN":           1.0     # усиление темпа Λ′ (если поддерживается ядром)
}

def load_params(path: str = "best_params.json") -> Dict[str, Any]:
    user = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                user = json.load(f)
            except Exception:
                # поврежденный JSON — игнорируем и работаем на дефолтах
                user = {}
    # дефолты перекрываются пользовательскими значениями
    params = {**REQUIRED_DEFAULTS, **(user or {})}
    # минимальная валидация
    for k, v in REQUIRED_DEFAULTS.items():
        if k not in params or params[k] is None:
            params[k] = v
    return params

# ---- 2) Тонкий адаптер для ядра: поддержим оба варианта API
def run_engine(params: Dict[str, Any], steps: int = STEPS, seed: int = SEED) -> Dict[str, Any]:
    """
    Возвращает словарь с метриками и, при наличии, временными рядами.
    Ожидаемые ключи метрик (если ядро их не вернет — считаем безопасные дефолты):
      drift_share, lock_share, iterate_share, var_win, mean_abs_dL_dt, corr,
      transitions_per_1k, bumps_count, last_bump_t
    """
    # импортируем ядро
    engine = None
    runner = None
    try:
        # предпочтительно — функция
        from ace_kernel_v04e import run_simulation as _runner  # type: ignore
        runner = _runner
    except Exception:
        try:
            # альтернатива — класс
            from ace_kernel_v04e import ACEngine_v04e  # type: ignore
            engine = ACEngine_v04e(params=params, seed=seed)
        except Exception:
            # последнее — более общий импорт
            try:
                from ace_kernel_v04e import ACEngine  # type: ignore
                engine = ACEngine(params=params, seed=seed)
            except Exception as e:
                raise RuntimeError(
                    "Не найден ни run_simulation(...), ни ACEngine_v04e/ACEngine в ace_kernel_v04e.py"
                ) from e

    # запуск
    if runner is not None:
        out = runner(params=params, steps=steps, seed=seed, save_dir=REPORT_DIR)
    else:
        out = engine.run(steps=steps, save_dir=REPORT_DIR)  # type: ignore

    # обязательные поля по умолчанию
    metrics = {
        "drift_share":       out.get("drift_share", 0.0),
        "lock_share":        out.get("lock_share", 1.0),
        "iterate_share":     out.get("iterate_share", 0.0),
        "var_win":           out.get("var_win", 0.0),
        "mean_abs_dL_dt":    out.get("mean_abs_dL_dt", 0.0),
        "corr":              out.get("corr", 0.0),
        "transitions_per_1k":out.get("transitions_per_1k", 0),
        "bumps_count":       out.get("bumps_count", out.get("anti_stall_bumps", 0)),
        "last_bump_t":       out.get("last_bump_t", None),
        "params":            out.get("params", params),
    }

    # если ядро вернуло ряды — попробуем дооценить var_win/corr
    try:
        omega = out.get("omega_series")
        lam   = out.get("lambda_series")
        if omega is not None and lam is not None:
            import numpy as np
            w = max(8, int(params.get("VAR_WINDOW", 220)))
            if len(omega) >= w:
                x = np.asarray(omega, float)
                # скользящее окно дисперсии последних w значений (последняя оценка)
                tail = x[-w:]
                metrics["var_win"] = float(np.var(tail, ddof=0))
            # корреляция производных
            dO = np.diff(np.asarray(omega, float))
            dL = np.diff(np.asarray(lam, float))
            if dO.size > 5 and dL.size > 5:
                num = float((dO - dO.mean()) @ (dL - dL.mean()))
                den = float((dO.std() * dL.std()) + 1e-12)
                metrics["corr"] = num / den if den > 0 else 0.0
                metrics["mean_abs_dL_dt"] = float(np.mean(np.abs(dL)))
    except Exception:
        pass

    return metrics

# ---- 3) Формирование вердикта в формате ACE REPORT
def alive_checks(m: Dict[str, Any]) -> Dict[str, bool]:
    ok_var   = (1e-6 <= m["var_win"] <= 1e-3)
    ok_meanL = (m["mean_abs_dL_dt"] > 1e-2)
    ok_drift = (m["drift_share"] >= 20.0)
    return {"var": ok_var, "meanL": ok_meanL, "drift": ok_drift}

def write_report(metrics: Dict[str, Any], out_path: str = SUMMARY_PATH) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    chk = alive_checks(metrics)
    verdict = "ALIVE" if all(chk.values()) else "NOT ALIVE"

    # красиво округлим
    def f(x, nd=5):
        if isinstance(x, (int,)):
            return str(x)
        if x == 0:
            return "0.00000"
        return f"{x:.{nd}f}"

    txt = []
    txt.append("===== ACE REPORT v0.4e-fix =====")
    txt.append(f"Drift share:          {f(metrics['drift_share'],2)} %")
    txt.append(f"Lock share:           {f(metrics['lock_share'],2)} %")
    txt.append(f"Iterate share:        {f(metrics['iterate_share'],2)} %\n")
    txt.append(f"var_win(Ω′):          {metrics['var_win']:.6e}")
    txt.append(f"mean |dΛ′/dt|:        {f(metrics['mean_abs_dL_dt'],5)}")
    txt.append(f"Corr(dΩ′, dΛ′):       {f(metrics['corr'],3)}\n")
    txt.append(f"Regime transitions /1k steps: {metrics.get('transitions_per_1k',0)}\n")
    txt.append("Alive rule:")
    txt.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   [{'OK' if chk['var']   else 'FAIL'}]")
    txt.append(f"  mean |dΛ′/dt| > 1e-2:            [{'OK' if chk['meanL'] else 'FAIL'}]")
    txt.append(f"  Drift share ≥ 20%:               [{'OK' if chk['drift'] else 'FAIL'}]\n")
    txt.append(f"VERDICT: [{verdict}]\n")
    txt.append("Notes:")
    txt.append(f"- anti-stall bumps: {metrics.get('bumps_count',0)} fired")
    last_b = metrics.get('last_bump_t')
    if last_b is not None:
        txt.append(f"- last bump at t = {last_b}")
    p = metrics.get("params", {})
    txt.append("- params: COUP_LA_TO_OM=%s, COUP_OM_TO_LA=%s, MEM_DECAY=%s, HYST=%s, NOISE=%s" %
               (p.get("COUP_LA_TO_OM"), p.get("COUP_OM_TO_LA"),
                p.get("MEM_DECAY"), p.get("HYST"), p.get("NOISE")))
    txt.append("================================")
    content = "\n".join(txt)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content + "\n")

    # дублируем в корень для быстрого доступа
    shutil.copyfile(out_path, LAST_REPORT)
    return content

# ---- 4) Точка входа
def main():
    params = load_params()
    metrics = run_engine(params, steps=STEPS, seed=SEED)
    report = write_report(metrics, SUMMARY_PATH)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp}] Report → {SUMMARY_PATH}  (duplicate: {LAST_REPORT})")
    print(report)

if __name__ == "__main__":
    main()
