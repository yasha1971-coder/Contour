import os, csv, json
import numpy as np

from ace_kernel_v04e import ACEngineV04eFix, load_params
from xi_chart_v04e import save_charts

REPORT_DIR = "ace_v04e_report"

def alive_verdict(m):
    cond1 = (1e-6 < m["var_win"] < 1e-3)
    cond2 = (m["mean_abs_dlambda"] > 1e-2)
    cond3 = (m["drift_share"] >= 0.20)
    ok = cond1 and cond2 and cond3
    return ok, cond1, cond2, cond3

def run_and_report(steps=6000, seed=123):
    p = load_params()
    eng = ACEngineV04eFix(p, steps=steps, seed=seed)
    m = eng.run()

    # подготовка данных
    t = np.arange(len(eng.omega_history))
    os.makedirs(REPORT_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(REPORT_DIR, "data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t","Omega","Lambda","dOmega","dLambda","regime"])
        for i in range(len(t)):
            w.writerow([
                i,
                eng.omega_history[i],
                eng.lambda_history[i],
                eng.domega_history[i],
                eng.dlambda_history[i],
                eng.regime_history[i]
            ])

    # графики
    try:
        save_charts(REPORT_DIR, t, np.array(eng.omega_history), np.array(eng.lambda_history),
                    np.array(eng.domega_history), np.array(eng.dlambda_history),
                    eng.regime_history)
    except Exception as e:
        print("Chart error:", e)

    ok, c1, c2, c3 = alive_verdict(m)

    # итоговый текст
    text = []
    text.append("===== ACE REPORT v0.4e-fix =====")
    text.append(f"Drift share:          {m['drift_share']*100:5.2f} %")
    text.append(f"Lock share:           {m['lock_share']*100:5.2f} %")
    text.append(f"Iterate share:        {m['iter_share']*100:5.2f} %\n")
    text.append(f"var_win(Ω′):          {m['var_win']:.6e}")
    text.append(f"mean |dΛ′/dt|:        {m['mean_abs_dlambda']:.5f}")
    text.append(f"Corr(dΩ′, dΛ′):       {m['corr']:.3f}\n")
    text.append(f"Regime transitions /1k steps: {m['regime_switches'] * 1000 // max(len(t),1)}")
    text.append("\nAlive rule:")
    text.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   [{'OK' if c1 else 'FAIL'}]")
    text.append(f"  mean |dΛ′/dt| > 1e-2:            [{'OK' if c2 else 'FAIL'}]")
    text.append(f"  Drift share ≥ 20%:               [{'OK' if c3 else 'FAIL'}]\n")
    text.append(f"VERDICT: [{'ALIVE CANDIDATE' if ok else 'NOT ALIVE'}]\n")
    text.append(f"Notes:")
    text.append(f"- anti-stall bumps: {m['bumps']} fired")
    text.append(f"- last bump at t = {m['last_bump_t']}")
    text.append(f"- params: COUP_LA_TO_OM={p['COUP_LA_TO_OM']}, COUP_OM_TO_LA={p['COUP_OM_TO_LA']}, "
                f"MEM_DECAY={p['MEM_DECAY']}, HYST={p['HYST']}, NOISE={p['NOISE']}")
    text.append("================================")

    report_txt = "\n".join(text)
    print(report_txt)

    with open(os.path.join(REPORT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

if __name__ == "__main__":
    run_and_report(steps=6000, seed=123)
