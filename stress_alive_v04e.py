# stress_alive_v04e.py
import json, subprocess, sys
from pathlib import Path
from alive_tools_v04e import alive_plus, estimate_recovery_time

REPORT_DIR = Path("ace_v04e_report")

def read_summary():
    s = (REPORT_DIR/"summary.txt").read_text(encoding="utf-8")
    # грубый парсинг
    def take(tag):
        for line in s.splitlines():
            if line.strip().startswith(tag):
                return float(line.split()[-1].replace("%",""))/100.0 if tag.endswith("share:") else float(line.split()[-1])
        return None

    metrics = {
        "drift_share": take("Drift share:"),
        "var_win": take("var_win(Ω′):"),
        "mean_abs_dL_dt": take("mean |dΛ′/dt|:"),
        "corr": take("Corr(dΩ′, dΛ′):"),
        "regime_transitions": take("Regime transitions /1k steps:"),
    }
    return metrics

def run_sim(shock=None, steps=6000):
    if shock:
        sh = json.dumps(shock)
        cmd = ["python","main.py","--steps",str(steps),"--shock",sh]
    else:
        cmd = ["python","main.py","--steps",str(steps)]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def scenario(name, shock):
    print(f"\n=== SCENARIO: {name} ===")
    run_sim(shock=shock)
    metrics = read_summary()
    score = alive_plus(metrics)
    trec = estimate_recovery_time(REPORT_DIR/"data.csv", shock_t=shock["t"]+shock["dur"], window=200)
    print("Metrics:", metrics)
    print("ALIVE+:", score)
    print("T_recover (steps):", trec)
    return {"name":name, "metrics":metrics, "ALIVE+":score, "T_recover":trec}

def main():
    REPORT_DIR.mkdir(exist_ok=True)
    results = []

    # S1: шумовой импульс
    results.append(scenario("NOISE x2 / 250", {"t":2000,"dur":250,"param":"NOISE","factor":2.0}))
    # S2: ослабление связности
    results.append(scenario("COUP_OM_TO_LA ×0.8 / 300", {"t":2600,"dur":300,"param":"COUP_OM_TO_LA","factor":0.8}))
    # S3: усиление памяти (сильное сглаживание)
    results.append(scenario("MEM_DECAY ×1.4 / 200", {"t":3200,"dur":200,"param":"MEM_DECAY","factor":1.4}))

    # сводка
    out = REPORT_DIR/"alive_plus_report.txt"
    lines = ["=== STRESS & ALIVE+ REPORT ===\n"]
    for r in results:
        lines.append(f"\n[{r['name']}]\n")
        for k,v in r["metrics"].items():
            lines.append(f"- {k}: {v}\n")
        lines.append(f"- ALIVE+: {r['ALIVE+']}\n")
        lines.append(f"- T_recover: {r['T_recover']}\n")
    out.write_text("".join(lines), encoding="utf-8")
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
