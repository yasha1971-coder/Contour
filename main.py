# main.py  — ACE v0.4e runner + nice summary
import os, sys, json, re, subprocess, textwrap
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
REPORT_DIR = ROOT / "ace_v04e_report"
SUMMARY_TXT = REPORT_DIR / "summary.txt"
RUN_SUMMARY = ROOT / "run_summary.txt"
BEST_JSON = ROOT / "best_params.json"

ANSI = {
    "reset":"\x1b[0m","bold":"\x1b[1m",
    "green":"\x1b[32m","red":"\x1b[31m","yellow":"\x1b[33m","cyan":"\x1b[36m",
}

def c(color, s): return f"{ANSI[color]}{s}{ANSI['reset']}"

def load_best_params():
    if BEST_JSON.exists():
        try:
            with open(BEST_JSON, "r", encoding="utf-8") as f:
                p = json.load(f)
            print(c("cyan", f"Using best_params.json: {p}"))
            return p
        except Exception as e:
            print(c("yellow", f"Warning: cannot read best_params.json ({e}). Using script defaults."))
    else:
        print(c("yellow", "best_params.json not found — using script defaults."))
    return {}

def run_step(title, cmd):
    print(c("cyan", f"\n— {title}"))
    subprocess.check_call(cmd, cwd=ROOT)

def parse_summary(text:str):
    # robust parser: looks for several possible key names
    # returns dict with drift, corr, var_win, alive (bool/str)
    res = {"drift":None, "corr":None, "var_win":None, "alive":None}
    # unify text
    t = text.replace("%","").lower()
    # drift
    m = re.search(r"(drift(?:_share)?)[^0-9\-+e.]*([\-+0-9.e]+)", t)
    if m: res["drift"] = float(m.group(2))
    # corr
    m = re.search(r"(corr(?:_last|elation)?)[^0-9\-+e.]*([\-+0-9.e]+)", t)
    if m: res["corr"] = float(m.group(2))
    # var window for omega
    m = re.search(r"(var(?:_omega)?(?:_win|_win_last)?)[^0-9\-+e.]*([\-+0-9.e]+)", t)
    if m: res["var_win"] = float(m.group(2))
    # alive flag
    m = re.search(r"(alive(?:_last)?)[^a-z]*(true|false)", t)
    if m: res["alive"] = (m.group(2) == "true")
    return res

def pretty_print(metrics:dict):
    drift = metrics.get("drift")
    corr  = metrics.get("corr")
    varw  = metrics.get("var_win")
    alive = metrics.get("alive")

    lines = []
    lines.append("")
    lines.append(ANSI["bold"] + "=== ACE v0.4e — run summary ===" + ANSI["reset"])
    lines.append(f"Drift share : {drift}")
    lines.append(f"corr(dΩ',dΛ') : {corr}")
    lines.append(f"var_win(Ω') : {varw}")
    # verdict (targets: drift≥0.60, corr≥0.25, var_win<1e-3)
    ok_drift = (drift is not None and drift >= 0.60)
    ok_corr  = (corr  is not None and corr  >= 0.25)
    ok_var   = (varw  is not None and varw  < 1e-3)
    alive_rule = ok_drift and ok_corr and ok_var

    if alive_rule:
        verdict = c("green", "ACE ALIVE  ✓")
    else:
        verdict = c("red",    "ACE NOT ALIVE  ✗")
    human_alive = f"(reported alive: {alive})" if isinstance(alive, bool) else "(reported alive: n/a)"
    lines.append(f"Verdict       : {verdict}  {human_alive}")

    # guidance
    tip = []
    if not ok_drift: tip.append("↑ drift (raise COUP_* a bit)")
    if not ok_corr:  tip.append("↑ coupling (COUP_OM_TO_LA/COUP_LA_TO_OM) or ↓ HYST")
    if not ok_var:   tip.append("↓ rigidity (slightly ↑ NOISE or ↓ MEM_DECAY)")
    if tip:
        lines.append(c("yellow", "Hints: " + "; ".join(tip)))

    block = "\n".join(lines) + "\n"
    print(block)
    RUN_SUMMARY.write_text(re.sub("\x1b\\[[0-9;]*m","",block), encoding="utf-8")
    print(c("cyan", f"Saved: {RUN_SUMMARY.name}"))

def main():
    _ = load_best_params()  # prints info if exists

    # 1) run kernel (creates CSV)
    run_step("Step 1/2 — run kernel", [sys.executable, "ace_kernel_v04e.py"])

    # 2) generate charts + summary.txt
    run_step("Step 2/2 — generate charts", [sys.executable, "xi_chart_v04e.py"])

    # 3) read summary & print big verdict
    if SUMMARY_TXT.exists():
        txt = SUMMARY_TXT.read_text(encoding="utf-8", errors="ignore")
        metrics = parse_summary(txt)
        pretty_print(metrics)
        print(c("cyan", f"Report dir: {REPORT_DIR}"))
    else:
        print(c("red", "summary.txt not found — check logs / filenames"))
        # fallback: try to open any summary.txt in repo
        cand = list(ROOT.glob("**/summary.txt"))
        if cand:
            txt = cand[0].read_text(encoding="utf-8", errors="ignore")
            metrics = parse_summary(txt)
            pretty_print(metrics)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(c("red", f"Subprocess failed: {e}"))
        sys.exit(1)
    except Exception as e:
        print(c("red", f"Error: {e}"))
        sys.exit(1)
