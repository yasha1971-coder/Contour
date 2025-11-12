# CLI runner for ACE Engine v0.4e-fix
import argparse, json, os
from ace_kernel_v04e import ACEParams, ACEEngine
from alive_tools_v04e import save_summary, save_time_series, try_save_plots

def load_params(path: str | None) -> ACEParams:
    if path and os.path.exists(path):
        with open(path, "r") as f: return ACEParams.from_dict(json.load(f))
    # default preference: alive_lock → best_params
    if os.path.exists("alive_lock_v04e.json"):
        with open("alive_lock_v04e.json", "r") as f: return ACEParams.from_dict(json.load(f))
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f: return ACEParams.from_dict(json.load(f))
    return ACEParams()

def main():
    ap = argparse.ArgumentParser(description="ACE Engine v0.4e-fix runner")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--params", type=str, default=None, help="JSON file with parameters")
    ap.add_argument("--profile", type=str, default=None, help="alias: '38' uses alive_lock_v04e.json, '42' uses best_params.json")
    ap.add_argument("--report-dir", type=str, default="ace_v04e_report")
    ap.add_argument("--no-reg", action="store_true", help="disable self-goal regulator v0.5")
    args = ap.parse_args()

    pfile = args.params
    if args.profile == "38":
        pfile = "alive_lock_v04e.json"
    elif args.profile == "42":
        pfile = "best_params.json"

    params = load_params(pfile)
    eng = ACEEngine(params)
    res = eng.run(steps=args.steps, use_regulator=not args.no_reg)

    os.makedirs(args.report_dir, exist_ok=True)
    save_time_series(args.report_dir, eng.s.hist)
    save_summary(args.report_dir, res)
    try_save_plots(args.report_dir, eng.s.hist)

    v = res["verdict"]["verdict"]
    print(json.dumps({
        "drift_share": res["metrics"]["drift_share"],
        "lock_share": res["metrics"]["lock_share"],
        "iterate_share": res["metrics"]["iterate_share"],
        "var_win": res["metrics"]["var_win"],
        "mean_abs_dlambda_dt": res["metrics"]["mean_abs_dlambda_dt"],
        "corr_domega_dlambda": res["metrics"]["corr_domega_dlambda"],
        "regime_transitions_per_1k": res["metrics"]["regime_transitions_per_1k"],
        "verdict": v
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
