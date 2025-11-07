# auto_tune_v04e.py
import numpy as np, random
from itertools import product
from ace_kernel_v04e import ACEngine, DEFAULTS

PARAM_GRIDS = {
    'COUP_LA_TO_OM': [0.06, 0.08, 0.10],
    'COUP_OM_TO_LA': [0.05, 0.07, 0.09],
    'MEM_DECAY':     [0.90, 0.92, 0.95],
    'HYST':          [0.012,0.016,0.020],
    'NOISE':         [0.002,0.003,0.004]
}

def composite_score(drift, corr, var_win):
    drift_s = min(drift/0.6, 1.0)
    corr_s  = min((corr if not np.isnan(corr) else 0)/0.25, 1.0)
    var_s   = min(1e-3/max(var_win,1e-10), 2.0)
    return drift_s * corr_s * var_s

def run_ace(params, seed=0):
    p = DEFAULTS.copy(); p.update(params)
    res = ACEngine(**p).run(seed=seed)
    res['score'] = composite_score(res['drift'], res['corr'], res['var_win'])
    return res

def early_stop_success(hist):
    recent = hist[-10:]
    ok = sum(1 for m in recent if (m['drift']>=0.6 and (m['corr'] if m['corr']==m['corr'] else 0)>=0.25 and m['var_win']<1e-3))
    return ok>=3

def random_around(best, ranges):
    out={}
    for k,v in best.items():
        r = ranges.get(k,0.0)
        out[k] = float(np.clip(np.random.normal(v, r), 1e-6, 10))
    return out

def auto_tune(seed=0):
    # Фаза 1: уплотнённая сетка (до 50)
    grid = list(product(*PARAM_GRIDS.values()))[:50]
    keys = list(PARAM_GRIDS.keys())
    best_score, best_params, hist = -1, None, []
    for tpl in grid:
        params = {k:v for k,v in zip(keys, tpl)}
        res = run_ace(params, seed=seed)
        hist.append(res)
        if res['score']>best_score:
            best_score, best_params = res['score'], params
        if early_stop_success(hist): break

    # Фаза 2: Adaptive Random (до 150)
    base_ranges = {'COUP_LA_TO_OM':0.02,'COUP_OM_TO_LA':0.02,'MEM_DECAY':0.03,'HYST':0.006,'NOISE':0.002}
    for i in range(150):
        decay = 0.95**(i//30); rng={k:v*decay for k,v in base_ranges.items()}
        cand = random_around(best_params, rng)
        # фиксируем прочие ключи по лучшим
        p = best_params.copy(); p.update(cand)
        res = run_ace(p, seed=seed+i+1)
        hist.append(res)
        if res['score']>best_score:
            best_score, best_params = res['score'], p
        if early_stop_success(hist): break

    return best_params, hist[-1]
