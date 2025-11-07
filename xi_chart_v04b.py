# xi_chart_v04b.py
# Visualization & QC for ACE v0.4b
# Reads ace_log_v04b.csv and produces:
#  - time series of Ω′, Λ′, δ with regime bands
#  - rolling variance var_win(Ω′)
#  - phase plot Ω′ vs Λ′ (colored by regime)
#  - quick Ξ-proxy and summary

import csv
import os
import math
from collections import deque, defaultdict
import matplotlib.pyplot as plt

CSV_PATH = "ace_log_v04b.csv"
OUT_DIR = "ace_v04b_report"

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def load_log(csv_path):
    data = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            data["t"].append(int(row["t"]))
            data["Omega_s"].append(float(row["Omega_s"]))
            data["Lambda_s"].append(float(row["Lambda_s"]))
            data["delta"].append(float(row["delta"]))
            data["dphi"].append(float(row["dphi"]))
            data["regime"].append(row["regime"])
            data["var_omega_win"].append(float(row["var_omega_win"]))
            data["corr_domega_dlambda"].append(float(row["corr_domega_dlambda"]))
            data["alive"].append(row["alive"] == "True" or row["alive"] is True)
    return data

def regime_mask(regimes, name):
    return [1 if r == name else 0 for r in regimes]

def plot_timeseries(data):
    t = data["t"]
    Om = data["Omega_s"]
    La = data["Lambda_s"]
    De = data["delta"]
    reg = data["regime"]

    plt.figure(figsize=(13, 6))
    plt.plot(t, Om, label="Ω′ (smoothed)")
    plt.plot(t, La, label="Λ′ (smoothed)")
    plt.plot(t, De, label="δ", alpha=0.7)

    # regime shading
    for name, color in [("Drift", "#a0f0a0"), ("Lock", "#c8d4ff"), ("Iterate", "#ffd4a8")]:
        mask = regime_mask(reg, name)
        # draw as vertical bands where mask==1
        start = None
        for i, m in enumerate(mask):
            if m == 1 and start is None: start = i
            if (m == 0 or i == len(mask)-1) and start is not None:
                end = i if m == 0 else i
                plt.axvspan(t[start], t[end], color=color, alpha=0.08)
                start = None

    plt.title("ACE v0.4b — Ω′, Λ′, δ over time (regime shaded)")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "timeseries.png"), dpi=180)
    plt.close()

def plot_var_and_corr(data):
    t = data["t"]
    varw = data["var_omega_win"]
    corr = data["corr_domega_dlambda"]

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, varw)
    ax[0].axhline(1e-2, linestyle="--")
    ax[0].set_title("Rolling var(Ω′)")
    ax[0].set_ylabel("variance")

    ax[1].plot(t, corr)
    ax[1].axhline(0.25, linestyle="--")
    ax[1].set_title("corr(dΩ′, dΛ′) — coherence threshold")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("corr")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "variance_and_corr.png"), dpi=180)
    plt.close()

def plot_phase(data):
    Om = data["Omega_s"]
    La = data["Lambda_s"]
    reg = data["regime"]

    colors = {"Drift": "#2b8a3e", "Lock": "#3a66ff", "Iterate": "#ff7f11"}
    plt.figure(figsize=(6.5, 6.5))
    for i in range(len(Om)):
        plt.scatter(Om[i], La[i], s=3, c=colors.get(reg[i], "#999999"))
    plt.xlabel("Ω′ (smoothed)")
    plt.ylabel("Λ′ (smoothed)")
    plt.title("Phase space: Ω′ vs Λ′ (colored by regime)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "phase_space.png"), dpi=180)
    plt.close()

def summarize(data):
    n = len(data["t"])
    drift = sum(1 for r in data["regime"] if r == "Drift")
    lock = sum(1 for r in data["regime"] if r == "Lock")
    itrt = sum(1 for r in data["regime"] if r == "Iterate")

    drift_share = drift / max(1, n)
    var_tail = data["var_omega_win"][-1] if data["var_omega_win"] else float("nan")
    corr_tail = data["corr_domega_dlambda"][-1] if data["corr_domega_dlambda"] else float("nan")
    alive_tail = data["alive"][-1] if data["alive"] else False

    summary = {
        "steps": n,
        "drift_share": round(drift_share, 4),
        "lock_share": round(lock / max(1, n), 4),
        "iterate_share": round(itrt / max(1, n), 4),
        "var_omega_win_last": var_tail,
        "corr_last": corr_tail,
        "alive_last": alive_tail
    }

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("=== ACE v0.4b QC summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Log not found: {CSV_PATH}. Run ace_kernel_v04b.py first.")

    data = load_log(CSV_PATH)
    plot_timeseries(data)
    plot_var_and_corr(data)
    plot_phase(data)
    summarize(data)

    print(f"Artifacts saved to: {os.path.abspath(OUT_DIR)}")
