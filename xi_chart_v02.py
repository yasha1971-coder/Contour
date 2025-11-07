# Быстрый анализ лога ACE v0.2 и построение «фазовой» кривой Ω′(t)×Λ′(t)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_log(path="ace_log_v02.csv"):
    df = pd.read_csv(path)
    return df

def summarize(df):
    n = len(df)
    share = df["regime"].value_counts(normalize=True).to_dict()
    var_omega = df["Omega_prime"].rolling(60).var().dropna().mean()
    dLambda = df["Lambda_prime"].diff().abs().rolling(60).mean().dropna().mean()
    dO = df["Omega_prime"].diff()
    dL = df["Lambda_prime"].diff()
    corr = dO.corr(dL)
    print("Steps:", n)
    print("Regime share (%):", {k: round(v*100,2) for k,v in share.items()})
    print("avg var_win(Ω′):", round(var_omega, 6))
    print("avg |dΛ′/dt|:", round(dLambda, 6))
    print("corr(dΩ′, dΛ′):", round(corr, 4))

def plot_phase(df):
    cmap = {"Drift":"#2aa198","Lock":"#268bd2","Iterate":"#cb4b16"}
    colors = df["regime"].map(cmap).values
    plt.figure(figsize=(6,6))
    plt.scatter(df["Omega_prime"], df["Lambda_prime"], c=colors, s=4, alpha=0.6)
    plt.xlabel("Ω′ (circulation)")
    plt.ylabel("Λ′ (axis rhythm)")
    plt.title("ACE v0.2: Phase scatter Ω′ × Λ′")
    plt.tight_layout()
    plt.savefig("xi_phase_v02.png", dpi=180)

def plot_time(df):
    plt.figure(figsize=(10,4))
    plt.plot(df["t"], df["Omega_prime"], label="Ω′")
    plt.plot(df["t"], df["Lambda_prime"], label="Λ′")
    plt.legend(); plt.xlabel("t"); plt.ylabel("value")
    plt.title("ACE v0.2: Ω′ and Λ′ over time")
    plt.tight_layout()
    plt.savefig("xi_time_v02.png", dpi=180)

if __name__ == "__main__":
    df = load_log("ace_log_v02.csv")
    summarize(df)
    plot_phase(df)
    plot_time(df)
    print("Saved: xi_phase_v02.png, xi_time_v02.png")
