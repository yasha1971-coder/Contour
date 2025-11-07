# ACE v0.4 charts — time series, phase portrait, coupling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ace_log_v04.csv")
df["dOmega"]  = df["Omega_prime"].diff()
df["dLambda"] = df["Lambda_prime"].diff()

# rolling corr (ξ trace)
WIN = 60
def rolling_corr(a,b,win):
    out = []
    for i in range(len(a)):
        if i<win: out.append(np.nan); continue
        x = a[i-win+1:i+1]; y = b[i-win+1:i+1]
        sx, sy = np.std(x), np.std(y)
        if sx==0 or sy==0: out.append(0.0)
        else: out.append(float(np.corrcoef(x,y)[0,1]))
    return out

df["xi_roll"] = rolling_corr(df["dOmega"].values, df["dLambda"].values, WIN)

# 1) time series
plt.figure(figsize=(10,4))
plt.plot(df["t"], df["Omega_prime"], label="Ω′")
plt.plot(df["t"], df["Lambda_prime"], label="Λ′", alpha=0.8)
plt.title("ACE v0.4 — time series (Ω′, Λ′)")
plt.xlabel("t"); plt.ylabel("value"); plt.legend()
plt.tight_layout(); plt.savefig("xi_time_v04.png"); plt.close()

# 2) phase portrait
colors = df["regime"].map({"Drift":"#4caf50","Lock":"#2196f3","Iter":"#ff9800"})
plt.figure(figsize=(5,5))
plt.scatter(df["Omega_prime"], df["Lambda_prime"], s=4, c=colors)
plt.title("ACE v0.4 — phase portrait")
plt.xlabel("Ω′"); plt.ylabel("Λ′")
plt.tight_layout(); plt.savefig("xi_phase_v04.png"); plt.close()

# 3) coupling / ξ trace
plt.figure(figsize=(10,3.6))
plt.plot(df["t"], df["xi_roll"])
plt.axhline(0.25, linestyle="--")
plt.title("ACE v0.4 — rolling corr(dΩ′, dΛ′) ≈ ξ")
plt.xlabel("t"); plt.ylabel("ξ (rolling corr)")
plt.tight_layout(); plt.savefig("xi_coupling_v04.png"); plt.close()

# text summary
drift_share = (df["regime"]=="Drift").mean()
omega_var   = df["Omega_prime"].tail(WIN).var()
mean_abs_dL = df["dLambda"].abs().dropna().mean()
coupling    = pd.Series(df["dOmega"]).corr(pd.Series(df["dLambda"]))
alive = (drift_share>=0.60 and omega_var<=1e-2 and mean_abs_dL>=1e-2 and coupling>=0.25)

with open("xi_summary_v04.txt","w",encoding="utf-8") as f:
    f.write("ACE v0.4 summary\n")
    f.write(f"Drift share      : {drift_share:.3f}\n")
    f.write(f"var_win(Ω′)      : {omega_var:.6e}\n")
    f.write(f"mean |dΛ′/dt|    : {mean_abs_dL:.6f}\n")
    f.write(f"corr(dΩ′, dΛ′)   : {coupling:.4f}\n")
    f.write(f"ALIVE            : {alive}\n")
    f.write("Plots: xi_time_v04.png, xi_phase_v04.png, xi_coupling_v04.png\n")
print("Saved: xi_time_v04.png, xi_phase_v04.png, xi_coupling_v04.png, xi_summary_v04.txt")
