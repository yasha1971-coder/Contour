# main.py
# Запуск симуляции, отчёт и сохранение результатов

from __future__ import annotations
import json, os
import numpy as np
import matplotlib.pyplot as plt

from ace_kernel_v04e import ACEParams, ACEEngineV04e

REPORT_DIR = "ace_v04e_report"
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------- загрузка параметров ----------------------------------------------

DEFAULT_PARAMS = ACEParams()

def load_params(path="best_params.json") -> ACEParams:
    p = dict(DEFAULT_PARAMS.__dict__)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for k, v in data.items():
            if k in p:
                p[k] = v
    return ACEParams(**p)

# ---------- форматирование отчёта --------------------------------------------

def fmt(x, nd=5):
    if isinstance(x, (int,)):
        return str(x)
    if x == 0:
        return "0"
    # более читаемая экспонента для малых чисел
    return f"{x:.{nd}g}" if abs(x) >= 1e-3 else f"{x:.6e}"

def alive_verdict(m: dict) -> tuple[str, dict]:
    ok_var   = (m["var_win"] >= 1e-6) and (m["var_win"] <= 1e-3)  # <= включительно!
    ok_speed = (m["mean_abs_dla"] > 1e-2)
    ok_drift = (m["drift_share"] >= 20.0)
    verdict = "ALIVE" if (ok_var and ok_speed and ok_drift) else "NOT ALIVE"
    return verdict, {"var": ok_var, "speed": ok_speed, "drift": ok_drift}

def write_report(m: dict, params: ACEParams, path_txt: str):
    verdict, chk = alive_verdict(m)

    lines = []
    lines.append("===== ACE REPORT v0.4e-fix =====")
    lines.append(f"Drift share:          {fmt(m['drift_share'], 4)} %")
    lines.append(f"Lock share:           {fmt(m['lock_share'], 4)} %")
    lines.append(f"Iterate share:        {fmt(m['iterate_share'], 4)} %\n")
    lines.append(f"var_win(Ω′):          {fmt(m['var_win'])}")
    lines.append(f"mean |dΛ′/dt|:        {fmt(m['mean_abs_dla'])}")
    lines.append(f"Corr(dΩ′, dΛ′):       {fmt(m['corr'])}\n")
    lines.append(f"Regime transitions /1k steps: {m['regime_transitions_per_1k']}\n")
    lines.append("Alive rule:")
    lines.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   [{'OK' if chk['var'] else 'FAIL'}]")
    lines.append(f"  mean |dΛ′/dt| > 1e-2:            [{'OK' if chk['speed'] else 'FAIL'}]")
    lines.append(f"  Drift share ≥ 20%:               [{'OK' if chk['drift'] else 'FAIL'}]\n")
    lines.append(f"VERDICT: [{verdict}]\n")
    lines.append("Notes:")
    lines.append(f"- anti-stall bumps: {m['anti_stall_bumps']} fired")
    lines.append(f"- last bump at t = {m['last_bump_t']}")
    lines.append(f"- params: COUP_LA_TO_OM={params.COUP_LA_TO_OM}, "
                 f"COUP_OM_TO_LA={params.COUP_OM_TO_LA}, MEM_DECAY={params.MEM_DECAY}, "
                 f"HYST={params.HYST}, NOISE={params.NOISE}")
    lines.append("================================")

    txt = "\n".join(lines)
    with open(path_txt, "w", encoding="utf-8") as fh:
        fh.write(txt)

    # дублируем для удобства
    with open("last_report.txt", "w", encoding="utf-8") as fh:
        fh.write(txt)

def save_figures(traces: dict):
    # эволюции
    fig1 = plt.figure(figsize=(8, 4.2))
    plt.plot(traces["omega_p"], label="Ω′")
    plt.plot(traces["lambda_p"], label="Λ′")
    plt.title("Evolution")
    plt.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(REPORT_DIR, "evolution.png"))
    plt.close(fig1)

    # дельты
    fig2 = plt.figure(figsize=(8, 4.2))
    plt.plot(traces["d_om"], label="ΔΩ′")
    plt.plot(traces["d_la"], label="ΔΛ′ (gain-applied)")
    plt.title("Deltas")
    plt.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(REPORT_DIR, "deltas.png"))
    plt.close(fig2)

def main():
    params = load_params("best_params.json")
    eng = ACEEngineV04e(params)
    out = eng.run(n_steps=6000)
    m, tr = out["metrics"], out["traces"]

    # csv
    data = np.column_stack([
        tr["omega_p"],
        tr["lambda_p"],
        np.r_[np.nan, tr["d_om"]],
        np.r_[np.nan, tr["d_la"]],
    ])
    np.savetxt(os.path.join(REPORT_DIR, "data.csv"),
               data, delimiter=",",
               header="omega_p,lambda_p,d_om,d_la", comments="", fmt="%.10g")

    # отчёт
    write_report(m, params, os.path.join(REPORT_DIR, "summary.txt"))
    # графики
    save_figures(tr)

if __name__ == "__main__":
    main()
