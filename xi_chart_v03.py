# Анализ v0.3: графики и сводка
import pandas as pd, numpy as np, matplotlib.pyplot as plt

df = pd.read_csv("ace_log_v03.csv")
df["dO"] = df["Omega"].diff()
df["dL"] = df["Lambda"].diff()

def regime_color(r):
    return {"Drift":"#00aa66","Lock":"#3a6ea5","Iter":"#d17f00"}.get(r,"#777")

# Временные ряды
plt.figure(figsize=(10,4))
plt.plot(df["step"], df["Omega"], label="Ω′")
plt.plot(df["step"], df["Lambda"], label="Λ′", alpha=0.8)
plt.title("ACE v0.3 — Ω′ и Λ′ во времени")
plt.xlabel("step"); plt.legend(); plt.tight_layout()
plt.savefig("xi_time_v03.png", dpi=160)

# Фаза: Ω′ vs Λ′, раскраска по режимам
plt.figure(figsize=(5,5))
for r, sub in df.groupby("regime"):
    plt.scatter(sub["Omega"], sub["Lambda"], s=4, alpha=0.5, c=regime_color(r), label=r)
plt.xlabel("Ω′"); plt.ylabel("Λ′")
plt.title("Фазовый портрет (v0.3)")
plt.legend(markerscale=3); plt.tight_layout()
plt.savefig("xi_phase_v03.png", dpi=160)

# Корреляция производных (скользящее окно)
win = 120
corrs = []
for i in range(len(df)):
    j = max(0, i-win)
    c = np.corrcoef(df["dO"].iloc[j:i+1].fillna(0), df["dL"].iloc[j:i+1].fillna(0))[0,1]
    corrs.append(c)
plt.figure(figsize=(10,3))
plt.plot(df["step"], corrs)
plt.axhline(0.3, color="k", ls="--", lw=1)
plt.title("Скользящая corr(dΩ′, dΛ′) — целевой порог 0.3")
plt.tight_layout()
plt.savefig("xi_coupling_v03.png", dpi=160)

print("Saved: xi_time_v03.png, xi_phase_v03.png, xi_coupling_v03.png")
