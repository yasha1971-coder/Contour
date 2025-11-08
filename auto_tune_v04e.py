import copy, os, csv
import numpy as np
from ace_kernel_v04e import ACEngineV04eFix, load_params

def composite(m):
    # цель: поднять drift, не словить нулевую вар, чуть стимулировать корр
    s = 0.0
    s += min(m["drift_share"]/0.6, 1.0)
    s += min(max(m["corr"], 0.0)/0.25, 1.0)
    s += min(1e-3/max(m["var_win"],1e-10), 2.0)
    return s

def run_candidate(base, tweaks, steps=6000, seed=42):
    p = copy.deepcopy(base)
    p.update(tweaks)
    eng = ACEngineV04eFix(p, steps=steps, seed=seed)
    metrics = eng.run()
    return p, metrics, eng

def choose_and_render(base, candidates=6, steps=6000):
    # вариации вокруг базы
    grid = [
        {"COUP_LA_TO_OM": base["COUP_LA_TO_OM"]*x, "COUP_OM_TO_LA": base["COUP_OM_TO_LA"]*y,
         "MEM_DECAY": base["MEM_DECAY"]+d, "HYST": base["HYST"]+h, "NOISE": base["NOISE"]+n}
        for x,y,d,h,n in [(0.9,0.8,-0.04,0.0,0.002),
                          (1.1,1.25,-0.08,-0.002,0.003),
                          (1.0,1.0,-0.08,0.002,0.003),
                          (1.2,1.1,-0.04,0.002,0.001),
                          (0.95,1.2,-0.06,0.000,0.0025),
                          (1.3,1.3,-0.12,0.000,0.004)]
    ][:candidates]

    best = None
    best_score = -1.0
    records = []
    for i,tw in enumerate(grid,1):
        p, m, _ = run_candidate(base, tw, steps=steps, seed=42+i)
        s = composite(m)
        records.append((i, p, m, s))
        if s > best_score:
            best_score, best = s, (p,m)

    return best, records

if __name__ == "__main__":
    base = load_params()
    (best_p, best_m), recs = choose_and_render(base, candidates=6, steps=6000)
    print("=== Best candidate selected ===")
    print("Best params:", best_p)
    print("[best] metrics:", best_m)
