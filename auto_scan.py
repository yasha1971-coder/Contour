# auto_scan.py
# Автоподбор параметров ACE + валидация стабильности (10k шагов) + обновление best_params.json

import os, json, itertools, subprocess, re, sys
from pathlib import Path

# -------- Параметрическая сетка (можно расширять) --------
GRID = {
    "NOISE":     [0.010, 0.012, 0.014],
    "MEM_DECAY": [0.34, 0.36, 0.38, 0.40],
    "HYST":      [0.0055, 0.0065, 0.0075],
}

# -------- Базовые параметры ядра (не перетираются сеткой) --------
BASE = {
    "COUP_LA_TO_OM": 0.18,
    "COUP_OM_TO_LA": 0.20,
    "L_GAIN":        1.3,
    "VAR_WINDOW":    300,
}

# -------- Настройки автозапусков --------
SHORT_STEPS = "4000"
LONG_STEPS  = "10000"  # стабильность
RUNS_DIR    = Path("auto_runs")
RUNS_DIR.mkdir(exist_ok=True)

# -------- Разбор отчёта --------
METRIC_PATTERNS = {
    "drift": r"Drift share:\s+([\d.]+)",
    "var":   r"var_win\(Ω′\):\s+([\deE\+\-\.]+)",
    "mean":  r"mean \|dΛ′/dt\|:\s+([\deE\+\-\.]+)",
}

def extract_metrics(text: str) -> dict:
    out = {}
    for k, pat in METRIC_PATTERNS.items():
        m = re.search(pat, text)
        out[k] = float(m.group(1)) if m else float("nan")
    out["verdict"] = "ALIVE" if "VERDICT: [ALIVE]" in text else "NOT ALIVE"
    return out

def score_metrics(m: dict) -> float:
    # Чем ближе mean к 0.01 и var к 5e-4 — тем лучше; ALIVE даёт бонус
    if any((k not in m or (isinstance(m[k], float) and (m[k] != m[k]))) for k in ("drift","var","mean")):
        return -1.0
    score = 0.0
    if m["verdict"] == "ALIVE":
        score += 10.0
    score += min(m["drift"] / 50.0, 1.0)                    # 0..1
    score += (1.0 - min(abs(m["mean"] - 0.01)/0.01, 1.0))   # 0..1
    score += (1.0 - min(abs(m["var"]  - 5e-4)/5e-4, 1.0))   # 0..1
    return score

def run_once(params: dict, steps: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_params = out_dir / "tmp_params.json"
    tmp_params.write_text(json.dumps(params, indent=2), encoding="utf-8")

    log_path = out_dir / "run.log"
    cmd = [
        sys.executable, "main.py",
        "--steps", steps,
        "--params", str(tmp_params),
        "--report-dir", str(out_dir),
    ]
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    return log_path.read_text(encoding="utf-8")

def main():
    summaries = []
    best_score = -1.0
    best = None

    # ---- Парсим сетку и запускаем короткие прогоны ----
    for noise, mem, hyst in itertools.product(GRID["NOISE"], GRID["MEM_DECAY"], GRID["HYST"]):
        run_name = f"N{noise:.3f}_M{mem:.2f}_H{hyst:.4f}"
        params = {**BASE, "NOISE": noise, "MEM_DECAY": mem, "HYST": hyst}

        print(f"\n>>> Scan run: {run_name}")
        text = run_once(params, SHORT_STEPS, RUNS_DIR / run_name)
        m = extract_metrics(text)
        s = score_metrics(m)

        line = f"{run_name:28s} → {m['verdict']:9s} | drift={m['drift']:5.1f}%  mean={m['mean']:.5f}  var={m['var']:.2e}  score={s:.3f}"
        print(line)
        summaries.append(line)

        if s > best_score:
            best_score = s
            best = {"params": params, "metrics": m, "score": s, "run_name": run_name}

    # ---- Сохраняем сводку сканирования ----
    (RUNS_DIR / "summary.txt").write_text("\n".join(summaries), encoding="utf-8")

    if not best:
        print("\n❌ Не найдено валидных результатов.")
        return

    # ---- Обновляем best_params.json (по результату сканирования) ----
    best_cfg = {**best["params"], **best["metrics"], "score": best["score"], "source_run": best["run_name"]}
    Path("best_params.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    print("\n✅ Preliminary best configuration written to best_params.json")
    print(json.dumps(best_cfg, indent=2))

    # ---- Долгая валидация стабильности (10k шагов) ----
    print("\n🔁 Stability check (10k steps)…")
    stable_dir = RUNS_DIR / f"{best['run_name']}_stability"
    text_long = run_once(best["params"], LONG_STEPS, stable_dir)
    m_long = extract_metrics(text_long)

    # Сохраняем отдельный отчёт по стабильности
    stability_report = (
        "===== STABILITY CHECK (10k) =====\n"
        f"Verdict: {m_long['verdict']}\n"
        f"Drift share: {m_long['drift']:.2f}%\n"
        f"mean |dΛ′/dt|: {m_long['mean']:.5f}\n"
        f"var_win(Ω′): {m_long['var']:.6e}\n"
    )
    (stable_dir / "stability_report.txt").write_text(stability_report, encoding="utf-8")
    print(stability_report)

    # ---- Закрепляем стабильный бест, если ALIVE ----
    if m_long["verdict"] == "ALIVE":
        stable_best = {**best["params"], **m_long, "validated_on_steps": int(LONG_STEPS)}
        Path("stable_best_params.json").write_text(json.dumps(stable_best, indent=2), encoding="utf-8")
        print("🏁 Stable ALIVE confirmed → saved to stable_best_params.json")
    else:
        print("⚠️ Stability run NOT ALIVE. Проверь summary и подправь сетку/параметры.")

    print("\nAll runs completed. See:")
    print(f"  - {RUNS_DIR/'summary.txt'}  (все короткие прогоны)")
    print(f"  - {stable_dir/'stability_report.txt'}  (валидация на 10k)")
    print("  - best_params.json / stable_best_params.json")

if __name__ == "__main__":
    main()
