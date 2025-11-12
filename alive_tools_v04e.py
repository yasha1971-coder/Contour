# Utilities for reporting/plots (safe if matplotlib absent)
import os, json, csv
from typing import Dict, List

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_summary(report_dir: str, results: Dict):
    ensure_dir(report_dir)
    m = results["metrics"]; v = results["verdict"]; p = results["params"]
    lines = []
    lines.append("===== ACE REPORT v0.4e-fix =====")
    lines.append(f"Drift share:          {m['drift_share']:.2f} %")
    lines.append(f"Lock share:           {m['lock_share']:.2f} %")
    lines.append(f"Iterate share:        {m['iterate_share']:.2f} %\n")
    lines.append(f"var_win(Ω′):          {m['var_win']:.6e}")
    lines.append(f"mean |dΛ′/dt|:        {m['mean_abs_dlambda_dt']:.5f}")
    lines.append(f"Corr(dΩ′, dΛ′):       {m['corr_domega_dlambda']:+.3f}\n")
    lines.append(f"Regime transitions /1k steps: {m['regime_transitions_per_1k']}\n")
    lines.append("Alive rule:")
    lines.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   [{'OK' if v['ok_var'] else 'FAIL'}]")
    lines.append(f"  mean |dΛ′/dt| > 1e-2:            [{'OK' if v['ok_vel'] else 'FAIL'}]")
    lines.append(f"  Drift share ≥ 20%:               [{'OK' if v['ok_drift'] else 'FAIL'}]\n")
    lines.append(f"VERDICT: [{v['verdict']}]\n")
    with open(os.path.join(report_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))

def save_time_series(report_dir: str, state_hist: Dict[str, List[float]]):
    ensure_dir(report_dir)
    path = os.path.join(report_dir, "data.csv")
    n = len(state_hist["omega_p"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "omega", "lambda"])
        for i in range(n):
            w.writerow([i, state_hist["omega_p"][i], state_hist["lambda_p"][i]])

def try_save_plots(report_dir: str, state_hist: Dict[str, List[float]]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        t = range(len(state_hist["omega_p"]))
        op = state_hist["omega_p"]
        lp = state_hist["lambda_p"]

        # evolution
        plt.figure()
        plt.plot(t, op, label="Omega'")
        plt.plot(t, lp, label="Lambda'")
        plt.legend(); plt.title("evolution")
        plt.savefig(os.path.join(report_dir, "evolution.png")); plt.close()

        # deltas
        op = np.asarray(op); lp = np.asarray(lp)
        dt_op = np.diff(op, prepend=op[0])
        dt_lp = np.diff(lp, prepend=lp[0])
        plt.figure()
        plt.plot(t, dt_op, label="dOmega/dt")
        plt.plot(t, dt_lp, label="dLambda/dt")
        plt.legend(); plt.title("deltas")
        plt.savefig(os.path.join(report_dir, "deltas.png")); plt.close()
    except Exception:
        # plotting is optional
        pass
