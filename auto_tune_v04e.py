# auto_tune_v04e.py
import json, itertools
from ace_kernel_v04e import ACEParams, ACEEngineV04e
from main import alive_verdict

BASE = {
    "COUP_LA_TO_OM": 0.18,
    "COUP_OM_TO_LA": 0.20,
    "MEM_DECAY":     0.40,
    "HYST":          0.007,
    "NOISE":         0.013,
    "VAR_WINDOW":    300,
    "DRIFT_HYST":    0.022,
    "ANTI_STALL_BUMP": 0.012,
    "L_GAIN":        1.35,
}

def try_params(pdict):
    eng = ACEEngineV04e(ACEParams(**pdict))
    m = eng.run(6000)["metrics"]
    ver, _ = alive_verdict(m)
    return ver, m

def main():
    grid = {
        "MEM_DECAY":  [0.38, 0.40, 0.42, 0.44],
        "NOISE":      [0.012, 0.013, 0.014],
        "HYST":       [0.0065, 0.007, 0.0075],
        "L_GAIN":     [1.30, 1.35, 1.40],
    }
    best = None
    for combo in itertools.product(*grid.values()):
        pd = dict(BASE)
        for k, v in zip(grid.keys(), combo):
            pd[k] = v
        ver, m = try_params(pd)
        score = (
            (1 if ver == "ALIVE" else 0),
            -abs(m["var_win"] - 6.5e-04),
            m["mean_abs_dla"],
            m["drift_share"],
        )
        if (best is None) or (score > best[0]):
            best = (score, pd, m, ver)
        if ver == "ALIVE":
            break

    _, pd, m, ver = best
    print("VERDICT:", ver)
    print("PARAMS:", pd)
    print("METRICS:", m)
    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(pd, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
