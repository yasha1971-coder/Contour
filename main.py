# main.py — авто-модификатор вокруг ACE v0.4e

import json, os, re, sys, subprocess, time
from copy import deepcopy

BEST_FILE = "best_params.json"
REPORT_DIR = "ace_v04e_report"
SUMMARY = os.path.join(REPORT_DIR, "summary.txt")

# --- утилиты --------------------------------------------------------------

def load_best():
    with open(BEST_FILE, "r") as f:
        return json.load(f)

def save_params(p):
    with open(BEST_FILE, "w") as f:
        json.dump(p, f, indent=2)

def run_kernel_once():
    # 1) ядро
    subprocess.check_call([sys.executable, "ace_kernel_v04e.py"])
    # 2) графики/сводка
    subprocess.check_call([sys.executable, "xi_chart_v04e.py"])

def parse_summary():
    """
    Ожидаем строки вида:
      Drift_share: 0.9697
      Corr_last: 0.0000
      Var_omega_win: 0.000000e+00
      Mean_abs_dlambda: 3.2e-03
      Alive: false/true   (может отсутствовать в старых версиях)
    Возвращаем словарь метрик с float’ами.
    """
    with open(SUMMARY, "r") as f:
        txt = f.read()

    def find_float(label, default=None):
        m = re.search(rf"{label}\s*:\s*([-\d\.eE\+]+)", txt)
        return float(m.group(1)) if m else default

    def find_bool(label, default=None):
        m = re.search(rf"{label}\s*:\s*(true|false)", txt, flags=re.I)
        if not m: return default
        return m.group(1).lower() == "true"

    return {
        "drift":         find_float("Drift_share", 0.0),
        "corr":          find_float("Corr_last", 0.0),
        "var_win":       find_float("Var_omega_win", 0.0),
        "mean_dlambda":  find_float("Mean_abs_dlambda", 0.0),
        "alive":         find_bool("Alive", None),
    }

def score(m):
    # Композитный скор: поощряем высокий drift, ненулевую корреляцию и ненулевую variance
    drift = m.get("drift", 0.0)
    corr  = max(0.0, m.get("corr", 0.0))
    varw  = m.get("var_win", 0.0)
    # Лёгкий бонус, если варьируется Ω′ (var_win > 0)
    var_bonus = 0.15 if (varw is not None and varw > 0.0) else 0.0
    return 0.6*drift + 0.35*corr + var_bonus

def print_metrics(tag, m):
    print(f"[{tag}] drift={m['drift']:.4f}  corr={m['corr']:.4f}  var_win={m['var_win']:.3e}  mean|dΛ'|={m['mean_dlambda']:.3e}  alive={m['alive']}")

# --- генерация кандидатных наборов вокруг текущего ------------------------

def gen_candidates(p0):
    """
    Малый локальный поиск вокруг текущего best_params:
    - немного двигаем коэффициенты связи
    - ослабляем память и чуть увеличиваем шум (чтобы «раскачать» Ω′)
    - лёгкий сдвиг гистерезиса
    """
    c = []

    def clamp(x, lo, hi): return max(lo, min(hi, x))

    base = deepcopy(p0)
    c.append(base)  # текущий как нулевой кандидат

    tweaks = [
        # усилить связь и уменьшить память, чуть повысить шум
        {"COUP_LA_TO_OM": +0.02, "COUP_OM_TO_LA": +0.01, "MEM_DECAY": -0.05, "NOISE": +0.0015, "HYST": -0.002},
        # балансная пара
        {"COUP_LA_TO_OM": +0.01, "COUP_OM_TO_LA": +0.02, "MEM_DECAY": -0.04, "NOISE": +0.0010, "HYST": +0.000},
        # мягче связь, но ещё меньше память и больше шум
        {"COUP_LA_TO_OM": +0.00, "COUP_OM_TO_LA": +0.00, "MEM_DECAY": -0.07, "NOISE": +0.0020, "HYST": +0.002},
        # чуть сильнее гистерезис чтобы избегать дрожания
        {"COUP_LA_TO_OM": +0.015,"COUP_OM_TO_LA": +0.015,"MEM_DECAY": -0.03,"NOISE": +0.0010,"HYST": +0.004},
        # агрессивный вариант для «размораживания»
        {"COUP_LA_TO_OM": +0.03, "COUP_OM_TO_LA": +0.03, "MEM_DECAY": -0.08, "NOISE": +0.0025, "HYST": -0.002},
    ]

    for t in tweaks:
        p = deepcopy(p0)
        p["COUP_LA_TO_OM"] = clamp(p["COUP_LA_TO_OM"] + t["COUP_LA_TO_OM"], 0.02, 0.20)
        p["COUP_OM_TO_LA"] = clamp(p["COUP_OM_TO_LA"] + t["COUP_OM_TO_LA"], 0.02, 0.20)
        p["MEM_DECAY"]     = clamp(p["MEM_DECAY"]     + t["MEM_DECAY"],     0.70, 0.98)
        p["NOISE"]         = clamp(p["NOISE"]         + t["NOISE"],         0.0005, 0.010)
        p["HYST"]          = clamp(p["HYST"]          + t["HYST"],          0.008, 0.030)
        c.append(p)
    return c

# --- основная процедура ----------------------------------------------------

if __name__ == "__main__":
    print("=== ACE v0.4e — auto-modifier run ===")

    # текущая база
    base_params = load_best()
    print("Base params:", base_params)

    candidates = gen_candidates(base_params)

    best_s, best_m, best_p = -1e9, None, None

    for idx, params in enumerate(candidates):
        print(f"\n--- Candidate #{idx+1}/{len(candidates)} ---")
        save_params(params)
        # чистим прошлую сводку, чтобы не перепутать
        if os.path.exists(SUMMARY):
            try: os.remove(SUMMARY)
            except: pass

        run_kernel_once()
        m = parse_summary()
        print_metrics("metrics", m)
        s = score(m)
        print(f"score = {s:.4f}")
        if s > best_s:
            best_s, best_m, best_p = s, m, deepcopy(params)

    # оставляем победителя
    print("\n=== Best candidate selected ===")
    print("Best params:", best_p)
    print_metrics("best", best_m)
    save_params(best_p)

    # финальный прогон для красивых графиков (уже с лучшими параметрами)
    print("\nFinal pass to render charts with best params...")
    run_kernel_once()
    print("\nDone. See report folder:", REPORT_DIR)
