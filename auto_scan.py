# auto_scan.py
# Автоматический подбор параметров ACE и обновление best_params.json

import os, json, itertools, subprocess, re

# ---- Параметрическая сетка ----
grid = {
    "NOISE": [0.010, 0.012, 0.014],
    "MEM_DECAY": [0.34, 0.36, 0.38, 0.40],
    "HYST": [0.0055, 0.0065, 0.0075],
}

# ---- Базовые параметры ----
base_params = {
    "COUP_LA_TO_OM": 0.18,
    "COUP_OM_TO_LA": 0.20,
    "L_GAIN": 1.3,
    "VAR_WINDOW": 300,
}

os.makedirs("auto_runs", exist_ok=True)
summary_lines = []
best_score = -1
best_params = None

def extract_metrics(text: str) -> dict:
    """Извлекает ключевые метрики из отчёта."""
    def find(pattern, default=0.0):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default
    return {
        "drift": find(r"Drift share:\s+([\d.]+)"),
        "var": find(r"var_win\(Ω′\):\s+([\deE\+\-\.]+)"),
        "mean": find(r"mean \|dΛ′/dt\|:\s+([\deE\+\-\.]+)"),
    }

# ---- Основной перебор ----
for noise, mem, hyst in itertools.product(grid["NOISE"], grid["MEM_DECAY"], grid["HYST"]):
    run_name = f"run_N{noise}_M{mem}_H{hyst}"
    params = {**base_params, "NOISE": noise, "MEM_DECAY": mem, "HYST": hyst}

    with open("tmp_params.json", "w") as f:
        json.dump(params, f)

    print(f"\n>>> Running {run_name}")
    log_path = f"auto_runs/{run_name}.log"
    cmd = [
        "python", "main.py",
        "--steps", "4000",
        "--params", "tmp_params.json",
        "--report-dir", f"auto_runs/{run_name}",
    ]
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    with open(log_path, encoding="utf-8") as f:
        text = f.read()

    verdict = "ALIVE" if "VERDICT: [ALIVE]" in text else "NOT ALIVE"
    metrics = extract_metrics(text)

    # Простая метрика пригодности
    # Чем ближе mean|dΛ′/dt| к 0.01 и var_win в середине диапазона — тем лучше
    score = 0
    if verdict == "ALIVE":
        score += 10
    score += min(metrics["drift"] / 50.0, 1.0)
    score += (1.0 - abs(metrics["mean"] - 0.01) / 0.01)
    score += (1.0 - abs(metrics["var"] - 5e-4) / 5e-4)

    summary = f"{run_name:35s} → {verdict:9s} | drift={metrics['drift']:5.1f}% mean={metrics['mean']:.5f} var={metrics['var']:.2e}"
    print(summary)
    summary_lines.append(summary)

    if score > best_score:
        best_score = score
        best_params = params | metrics | {"score": score, "verdict": verdict}

# ---- Сохранение итогов ----
with open("auto_runs/summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

if best_params:
    print("\n✅ Best configuration found:")
    print(json.dumps(best_params, indent=2))
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print("→ best_params.json updated.")

print("\nAll runs completed. Summary saved to auto_runs/summary.txt")
