import os, numpy as np
import matplotlib.pyplot as plt

def save_charts(report_dir, t, omega, lam, dO, dL, regimes):
    os.makedirs(report_dir, exist_ok=True)

    # --- evolution ---
    fig = plt.figure(figsize=(12,8))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(t, omega, label="Ω′ (Omega Prime)")
    ax1.set_title("ACE v0.4e-fix: Omega Prime Evolution"); ax1.legend(); ax1.set_ylabel("Ω′")

    ax2 = plt.subplot(3,1,2)
    ax2.plot(t, lam, color="green", label="Λ′ (Lambda Prime)")
    ax2.set_title("Lambda Prime Evolution"); ax2.legend(); ax2.set_ylabel("Λ′")

    ax3 = plt.subplot(3,1,3)
    reg_map = {"Lock":0,"Drift":1,"Iterate":2}
    reg_vals = np.array([reg_map[r] for r in regimes])
    ax3.plot(t, reg_vals, color="red", linewidth=1, label="Regime (0=Lock,1=Drift,2=Iterate)")
    ax3.set_title("Regime Classification"); ax3.legend(); ax3.set_xlabel("Time Step"); ax3.set_ylabel("Regime")
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "evolution.png"), dpi=160)
    plt.close(fig)

    # --- deltas ---
    fig2 = plt.figure(figsize=(14,8))
    ax21 = plt.subplot(2,1,1)
    ax21.plot(t, dO, label="dΩ′")
    ax21.set_title("Delta Omega Prime"); ax21.legend(); ax21.set_ylabel("dΩ′")

    ax22 = plt.subplot(2,1,2)
    ax22.plot(t, dL, label="dΛ′", color="orange")
    ax22.set_title("Delta Lambda Prime"); ax22.legend(); ax22.set_ylabel("dΛ′"); ax22.set_xlabel("Time Step")
    fig2.tight_layout()
    fig2.savefig(os.path.join(report_dir, "deltas.png"), dpi=160)
    plt.close(fig2)
