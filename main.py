# main.py
from ace_kernel_v04e import ACEngine, DEFAULTS
from auto_tune_v04e import auto_tune
import json, os

if __name__ == "__main__":
    os.makedirs("ace_v04e_report", exist_ok=True)
    best_params, last_metrics = auto_tune(seed=7)
    with open("ace_v04e_report/best_params.json","w") as f: json.dump(best_params, f, indent=2)
    print("Best params:", best_params); print("Last metrics:", last_metrics)

    # финальный прогон «лучших»
    p = DEFAULTS.copy(); p.update(best_params)
    eng = ACEngine(**p); res = eng.run(seed=2025)
    print("Final run:", res)
