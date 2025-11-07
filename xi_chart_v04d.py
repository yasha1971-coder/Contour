# Analysis & plots for ACE v0.4d
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os

CSV = "ace_log_v04d.csv"
OUT = "ace_v04d_report"

def sliding_var(x, win):
    x = np.asarray(x)
    if len(x)<win: return np.array([])
    out = []
    for i in range(win, len(x)+1):
        out.append(np.var(x[i-win:i]))
    return np.array(out)

def run():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV)
    # time series
    plt.figure(figsize=(10,4))
    plt.plot(df["t"], df["OmegaPrime"], label="Ω′")
    plt.plot(df["t"], df["LambdaPrime"], label="Λ′")
    plt.legend(); plt.title("ACE v0.4d — time series")
    plt.xlabel("t"); plt.ylabel("value")
    plt.tight_layout(); plt.savefig(f"{OUT}/timeseries.png"); plt.close()

    # phase portrait
    plt.figure(figsize=(5,5))
    plt.scatter(df["OmegaPrime"], df["LambdaPrime"], s=1, alpha=0.6)
    plt.title("Phase space (Ω′ vs Λ′)")
    plt.xlabel("Ω′"); plt.ylabel("Λ′")
    plt.tight_layout(); plt.savefig(f"{OUT}/phase_space.png"); plt.close()

    # rolling variance + correlation
    win_var = 120
    win_corr = 240
    var_om = sliding_var(df["OmegaPrime"].values, win_var)
    # rolling corr
    corr = []
    x = df["OmegaPrime"].values; y = df["LambdaPrime"].values
    for i in range(win_corr, len(x)+1):
        xs = x[i-win_corr:i]; ys = y[i-win_corr:i]
        sx, sy = np.std(xs), np.std(ys)
        if sx==0 or sy==0: corr.append(0.0)
        else: corr.append(np.corrcoef(xs, ys)[0,1])
    t1 = df["t"].values[win_var-1:len(df)]
    t2 = df["t"].values[win_corr-1:len(df)]

    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(t1, var_om); ax[0].set_ylabel("var_win(Ω′)")
    ax[1].plot(t2, corr);   ax[1].set_ylabel("corr(Ω′,Λ′)"); ax[1].set_xlabel("t")
    ax[0].axhline(1e-3, ls="--", lw=1, color="gray")
    ax[1].axhline(0.25, ls="--", lw=1, color="gray")
    fig.suptitle("QC: variance & coupling")
    plt.tight_layout(); plt.savefig(f"{OUT}/variance_and_corr.png"); plt.close()

    # regime timeline
    colors = {"Drift":"#66bb6a","Lock":"#42a5f5","Iterate":"#ffa726"}
    c = [colors[r] for r in df["regime"].values]
    plt.figure(figsize=(10,1.2))
    plt.scatter(df["t"], [1]*len(df), c=c, s=2)
    plt.yticks([]); plt.title("Regime timeline"); plt.xlabel("t")
    plt.tight_layout(); plt.savefig(f"{OUT}/regime_timeline.png"); plt.close()

    # summary
    drift = (df["regime"]=="Drift").mean()
    lock  = (df["regime"]=="Lock").mean()
    itrt  = (df["regime"]=="Iterate").mean()
    with open(f"{OUT}/summary.txt","w",encoding="utf-8") as f:
        f.write("ACE v0.4d — QC summary\n")
        f.write(f"Steps: {len(df)}\n")
        f.write(f"Regime share (%): Drift={drift*100:.1f}, Lock={lock*100:.1f}, Iterate={itrt*100:.1f}\n")
        f.write(f"avg var_win(Ω′): {np.nanmean(var_om) if var_om.size else float('nan')}\n")
        f.write(f"last corr(Ω′,Λ′): {corr[-1] if len(corr) else float('nan')}\n")

if __name__ == "__main__":
    run()
    print("Saved charts & summary to", OUT)
