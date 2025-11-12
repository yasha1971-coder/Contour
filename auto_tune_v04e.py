# Narrow, fast tuner around current params
import json, itertools
from ace_kernel_v04e import ACEParams, ACEEngine
from alive_tools_v04e import save_summary

RANGES = {
    "NOISE":     [0.009, 0.012, 0.015],
    "MEM_DECAY": [0.34, 0.38, 0.42],
    "HYST":      [0.0055, 0.0065, 0.0075],
    "L_GAIN":    [1.2, 1.3, 1.4]
}

def tune(base="best_params.json", steps=4000):
    with open(base, "r") as f:
        basep = json.load(f)

    best = None
    keys = list(RANGES.keys())
    for vals in itertools.product(*[RANGES[k] for k in keys]):
        trial = dict(basep)
        for k, v in zip(keys, vals): trial[k] = v
        eng = ACEEngine(ACEParams.from_dict(trial))
        res = eng.run(steps=steps, use_regulator=True)
        verdict = res["verdict"]["verdict"]
        if verdict == "ALIVE":
            best = trial; break
        # keep the closest by variance distance to center of window
        if best is None:
            best = trial

    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("Saved best_params.json")

if __name__ == "__main__":
    tune()
