# ACE v0.4c — анализ логов и QC-диаграммы
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOG = "ace_log_v04c.csv"

def main():
    df = pd.read_csv(LOG)
    out = Path("ace_v04c_report"); out.mkdir(exist_ok=True)

    # доли режимов
    share = df['regime'].value_counts(normalize=True)
    drift = float(share.get('Drift',0)); lock=float(share.get('Lock',0)); it=float(share.get('Iterate',0))

    # корреляция на последнем окне
    W = 200
    def rolling_corr(a,b,w=W):
        if len(a)<w or len(b)<w: return 0.0
        A=a[-w:]-a[-w:].mean(); B=b[-w:]-b[-w:].mean()
        da=A.std(); db=B.std()
        if da<1e-12 or db<1e-12: return 0.0
        return float(np.dot(A,B)/(w*da*db))

    corr_last = rolling_corr(df['dOmega'].values, df['dLambda'].values, W)
    var_omega = (df['Omega_prime'].tail(60).values - df['Omega_prime'].tail(60).mean())**2
    var_omega = float(var_omega.mean())

    # QC
    alive = (drift>=0.35 and lock<=0.45 and corr_last>=0.25 and var_omega<1e-3)

    # --- визуализация
    # 1) временные ряды
    fig = plt.figure(figsize=(10,4))
    plt.plot(df['Omega_prime'], label="Ω′")
    plt.plot(df['Lambda_prime'], label="Λ′")
    plt.title("ACE v0.4c — time series")
    plt.legend(); plt.tight_layout()
    fig.savefig(out/"timeseries.png"); plt.close(fig)

    # 2) фазовый портрет
    fig = plt.figure(figsize=(5,5))
    plt.plot(df['Omega_prime'], df['Lambda_prime'], linewidth=0.7)
    plt.xlabel("Ω′"); plt.ylabel("Λ′")
    plt.title("phase space (Ω′ vs Λ′)")
    plt.tight_layout()
    fig.savefig(out/"phase_space.png"); plt.close(fig)

    # 3) корреляция и дисперсия
    fig = plt.figure(figsize=(6,4))
    plt.bar(["drift_share","lock_share","iter_share"], [drift,lock,it])
    plt.axhline(0.35, linestyle="--")  # цель по Drift
    plt.title(f"corr_last={corr_last:.3f}, var_win(Ω′)={var_omega:.2e}")
    plt.tight_layout()
    fig.savefig(out/"variance_and_corr.png"); plt.close(fig)

    # summary.txt
    (out/"summary.txt").write_text(
        f"Drift: {drift:.3f}\nLock: {lock:.3f}\nIter: {it:.3f}\n"
        f"corr_last: {corr_last:.3f}\nvar_win(Omega): {var_omega:.3e}\n"
        f"ALIVE: {alive}\nLog: {LOG}\n"
    )
    print("Report saved to:", out)

if __name__ == "__main__":
    main()
