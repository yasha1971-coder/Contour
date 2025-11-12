# Wider scan (configurable), writes progress to report
import json, itertools, time, os
from ace_kernel_v04e import ACEParams, ACEEngine
from alive_tools_v04e import ensure_dir

GRID = {
    "NOISE":     [0.008, 0.010, 0.012, 0.014, 0.016],
    "MEM_DECAY": [0.30, 0.34, 0.38, 0.42, 0.46],
    "HYST":      [0.0050, 0.0060, 0.0070],
    "L_GAIN":    [1.1, 1.2, 1.3, 1.4],
    "VAR_WINDOW":[300]
}

REPORT = "ace_v04e_report/scan_report.txt"

def scan(base="best_params.json", steps=3000):
    ensure_dir("ace_v04e_report")
    with open(REPORT, "w") as f:
        f.write("ACE auto_scan start\n")

    with open(base, "r") as f:
        basep = json.load(f)

    keys = list(GRID.keys())
    count = 0
    for vals in itertools.product(*[GRID[k] for k in keys]):
        trial = dict(basep)
        for k, v in zip(keys, vals): trial[k] = v
        eng = ACEEngine(ACEParams.from_dict(trial))
        res = eng.run(steps=steps, use_regulator=True)
        m, v = res["metrics"], res["verdict"]
        line = f"{count:06d} {trial} -> {v['verdict']}, var={m['var_win']:.3e}, drift={m['drift_share']:.1f}%\n"
        with open(REPORT, "a") as f: f.write(line)
        if v["verdict"] == "ALIVE":
            with open("best_params.json", "w") as f:
                json.dump(trial, f, indent=2)
        count += 1

if __name__ == "__main__":
    scan()
