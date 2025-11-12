# Simple loop to run engine with v0.5 regulator enabled
import json
from ace_kernel_v04e import ACEParams, ACEEngine
from alive_tools_v04e import save_summary, save_time_series, try_save_plots

def run_with_reg(params_path="best_params.json", steps=6000, report_dir="ace_v04e_report"):
    with open(params_path, "r") as f:
        p = ACEParams.from_dict(json.load(f))
    eng = ACEEngine(p)
    res = eng.run(steps=steps, dt=1.0, use_regulator=True)
    save_time_series(report_dir, eng.s.hist)
    save_summary(report_dir, res)
    try_save_plots(report_dir, eng.s.hist)
    print(res["verdict"])
    return res

if __name__ == "__main__":
    run_with_reg()
