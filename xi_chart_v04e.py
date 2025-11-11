# xi_chart_v04e.py
# Построение сводного индекса Xi(t) и слоёв метрик для ACE v0.4e-fix

from __future__ import annotations
import os, json, subprocess
import numpy as np
import matplotlib.pyplot as plt

REPORT_DIR = "ace_v04e_report"
DATA_CSV   = os.path.join(REPORT_DIR, "data.csv")
PARAMS_JSON = "best_params.json"

# ---------- utils -------------------------------------------------------------

def ensure_data():
    """Гарантирует наличие data.csv; если нет — запускает main.py."""
    if not os.path.exists(DATA_CSV):
        os.makedirs(REPORT_DIR, exist_ok=True)
        # создаём данные
        subprocess.run(["python", "main.py"], check=True)

def load_params():
    """Читает VAR_WINDOW (и при необходимости другие будущие поля)."""
    var_window = 300
    if os.path.exists(PARAMS_JSON):
        with open(PARAMS_JSON, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        var_window = int(cfg.get("VAR_WINDOW", var_window))
    return {"VAR_WINDOW": max(10, var_window)}

def rolling_var(x: np.ndarray, win: int) -> np.ndarray:
    win = int(max(2, win))
    out = np.full_like(x, np.nan, dtype=float)
    c = x.astype(float)
    for i in range(len(c)):
        j0 = max(0, i - win + 1)
        seg = c[j0:i+1]
        out[i] = max(np.var(seg), 1e-12)
    return out

def rolling_mean_abs(x: np.ndarray, win: int) -> np.ndarray:
    win = int(max(2, win))
    out = np.full_like(x, np.nan, dtype=float)
    a = np.abs(x.astype(float))
    for i in range(len(a)):
        j0 = max(0, i - win + 1)
        out[i] = float(np.mean(a[j0:i+1]))
    return out

def rolling_corr(x: np.ndarray, y: np.ndarray, win: int) -> np.ndarray:
    win = int(max(3, win))
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        j0 = max(0, i - win + 1)
        xs = x[j0:i+1]
        ys = y[j0:i+1]
        if xs.size >= 3:
            sx = xs.std()
            sy = ys.std()
            if sx > 0 and sy > 0:
                out[i] = float(np.corrcoef(xs, ys)[0, 1])
            else:
                out[i] = 0.0
        else:
            out[i] = np.nan
    # клип корреляции на всякий случай
    out = np.clip(out, -1.0, 1.0)
    out[np.isnan(out)] = 0.0
    return out

def zscore(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------- pipeline ----------------------------------------------------------

def load_data():
    """
    data.csv layout (из main.py):
      col0: omega_p
      col1: lambda_p
      col2: d_om   (с NaN в первом элементе)
      col3: d_la   (gain-applied; с NaN в первом элементе)
    """
    arr = np.genfromtxt(DATA_CSV, delimiter=",", names=True)
    # имена колонок из заголовка: omega_p,lambda_p,d_om,d_la
    om = np.asarray(arr["omega_p"], dtype=float)
    la = np.asarray(arr["lambda_p"], dtype=float)
    dom = np.asarray(arr["d_om"], dtype=float)
    dla = np.asarray(arr["d_la"], dtype=float)
    # заменим первый NaN у дельт на 0 для удобства рисования
    if np.isnan(dom[0]): dom[0] = 0.0
    if np.isnan(dla[0]): dla[0] = 0.0
    return om, la, dom, dla

def compute_layers(om, la, dom, dla, win):
    var_om = rolling_var(om, win)
    spd_la = rolling_mean_abs(dla, win)
    cor_dl = rolling_corr(dom, dla, win)
    return var_om, spd_la, cor_dl

def compute_xi(var_om, spd_la, cor_dl, alpha=0.75, beta=0.25):
    """
    Строим сводный индекс Xi(t) \in [0,1]:
      Xi = sigmoid( alpha*(z(var_om) + z(spd_la)) + beta*cor_dl )
    - var_om и spd_la нормируем z-score
    - вклад корреляции добавляем как есть ([-1,1])
    Параметры alpha/beta можно варьировать.
    """
    zv = zscore(var_om)
    zs = zscore(spd_la)
    raw = alpha*(zv + zs) + beta*cor_dl
    return sigmoid(raw)

def save_layers_plot(var_om, spd_la, cor_dl, win):
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(var_om)
    ax1.set_title(f"Rolling VAR(Ω′), window={win}")
    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.plot(spd_la)
    ax2.set_title(f"Rolling mean |ΔΛ′|, window={win}")
    ax3 = plt.subplot(3,1,3, sharex=ax1)
    ax3.plot(cor_dl)
    ax3.set_title(f"Rolling Corr(ΔΩ′, ΔΛ′), window={win}")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "xi_layers.png"))
    plt.close(fig)

def save_xi_plot(xi, om, la):
    fig = plt.figure(figsize=(10,4.8))
    ax = plt.gca()
    ax.plot(xi, label="Xi(t)")
    ax.set_ylim(0,1)
    ax.set_title("Xi(t) — сводный индекс живости")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "xi_chart.png"))
    plt.close(fig)

    # дополнительный маленький оверлей эволюций (для качественной проверки)
    fig2 = plt.figure(figsize=(10, 4.8))
    ax2 = plt.gca()
    ax2.plot(om, label="Ω′")
    ax2.plot(la, label="Λ′")
    ax2.set_title("Evolution (Ω′, Λ′)")
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(os.path.join(REPORT_DIR, "xi_evolution_overlay.png"))
    plt.close(fig2)

def save_xi_csv(xi):
    out = np.column_stack([np.arange(len(xi)), xi])
    np.savetxt(os.path.join(REPORT_DIR, "xi_series.csv"),
               out, delimiter=",", header="t,xi", comments="", fmt="%.10g")

def main():
    ensure_data()
    cfg = load_params()
    win = int(cfg["VAR_WINDOW"])

    om, la, dom, dla = load_data()
    var_om, spd_la, cor_dl = compute_layers(om, la, dom, dla, win)
    xi = compute_xi(var_om, spd_la, cor_dl, alpha=0.75, beta=0.25)

    save_layers_plot(var_om, spd_la, cor_dl, win)
    save_xi_plot(xi, om, la)
    save_xi_csv(xi)

    # Короткий вывод в консоль
    print("Xi chart saved to:", os.path.join(REPORT_DIR, "xi_chart.png"))
    print("Layers plot saved to:", os.path.join(REPORT_DIR, "xi_layers.png"))
    print("Series saved to:", os.path.join(REPORT_DIR, "xi_series.csv"))
    print(f"Xi summary: mean={np.mean(xi):.4f}, std={np.std(xi):.4f}, min={np.min(xi):.4f}, max={np.max(xi):.4f}")

if __name__ == "__main__":
    main()
