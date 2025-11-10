# ace_kernel_v04e.py
# Autonomous Complexity Engine — v0.4e-fix (self-contained)
# - симуляция Ω′, Λ′ с куплингом, памятью, шумом и гистерезисом
# - отчёт с критериями "живости"
# - авто-подбор параметров (--tune) до VERDICT: [ALIVE]
#
# Зависимости: numpy, matplotlib (опционально для графиков)

import os, sys, json, math, csv, time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import numpy as np

# ---------------------------
# Константы и критерии ALIVE
# ---------------------------
ALIVE_VAR_MIN = 1e-6
ALIVE_VAR_MAX = 1e-3
ALIVE_MEAN_DLDL_MIN = 1e-2
ALIVE_DRIFT_MIN = 20.0
STEPS = 6000           # длительность прогона
DT = 1.0               # шаг "времени"
OUT_DIR = "ace_v04e_report"

# ---------------------------
# Параметры ядра
# ---------------------------
DEFAULT_PARAMS = {
    "COUP_LA_TO_OM": 0.20,
    "COUP_OM_TO_LA": 0.22,
    "MEM_DECAY":     0.32,
    "HYST":          0.007,
    "NOISE":         0.014,
    "VAR_WINDOW":    300
}

@dataclass
class Metrics:
    drift_share: float
    lock_share: float
    iterate_share: float
    var_win: float
    mean_abs_dldt: float
    corr_d_om_d_la: float
    transitions_per_1k: float
    bumps_fired: int
    last_bump_t: int

# ---------------------------
# Вспомогательные функции
# ---------------------------

def load_params(path="best_params.json") -> Dict[str, float]:
    params = DEFAULT_PARAMS.copy()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            params.update(data)
        except Exception:
            pass
    # страховка на VAR_WINDOW (часто пропадал)
    if "VAR_WINDOW" not in params:
        params["VAR_WINDOW"] = DEFAULT_PARAMS["VAR_WINDOW"]
    return params

def save_report_text(text: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open("last_report.txt", "w", encoding="utf-8") as f:
        f.write(text)

def fmt(x, nd=5):
    if isinstance(x, (int,)):
        return str(x)
    if x == 0:
        return "0"
    p = f"{x:.{nd}e}" if (abs(x) < 1e-3 or abs(x) >= 1e3) else f"{x:.{nd}f}"
    return p

def rolling_var(arr: np.ndarray, win: int) -> np.ndarray:
    """скользящая дисперсия (population, без ddof)"""
    if win <= 1:
        return np.zeros_like(arr)
    cumsum = np.cumsum(arr)
    cumsum2 = np.cumsum(arr*arr)
    var = np.zeros_like(arr)
    var[:win-1] = np.nan
    # для каждого окна [i-win+1, i]
    var[win-1:] = (cumsum2[win-1:] - np.concatenate(([0], cumsum2[:-win])) \
                  - ( (cumsum[win-1:] - np.concatenate(([0], cumsum[:-win])) )**2 )/win )/win
    # численно безопасно
    var = np.clip(var, 0.0, None)
    return var

def classify_regime(var_win_val: float, mean_abs_dldt_local: float, hyst: float) -> str:
    # Простая, но практичная логика режимов:
    # - Lock: очень малая вариативность + почти нет движения Λ′
    # - Drift: вариативность в "живом" коридоре (наш ALIVE промежуток по var)
    # - Iterate: всё остальное (более грубые или бурные режимы)
    if (var_win_val < max(ALIVE_VAR_MIN*0.5, 1e-7)) and (mean_abs_dldt_local < max(ALIVE_MEAN_DLDL_MIN*0.2, 1e-3)):
        return "Lock"
    if ALIVE_VAR_MIN <= var_win_val <= ALIVE_VAR_MAX*(1.0 + 2*hyst):
        return "Drift"
    return "Iterate"

# ---------------------------
# Динамика ACE
# ---------------------------

def run_simulation(params: Dict[str, float], steps: int = STEPS, dt: float = DT) -> Tuple[str, Metrics, Dict[str, np.ndarray]]:
    np.random.seed(42)  # детерминированность прогона

    # Распаковка параметров
    c_la_to_om = float(params["COUP_LA_TO_OM"])
    c_om_to_la = float(params["COUP_OM_TO_LA"])
    mem_decay   = float(params["MEM_DECAY"])
    hyst        = float(params["HYST"])
    noise_amp   = float(params["NOISE"])
    win         = int(params.get("VAR_WINDOW", DEFAULT_PARAMS["VAR_WINDOW"]))

    # Состояния
    Omega = 1.0   # Ω′ (целевой поток ~1)
    Lambda = 0.95 # Λ′ (ритм)
    mem = 0.0     # внутренняя "память"

    a_om = 0.05  # базовые "жёсткости"
    a_la = 0.03

    om = np.zeros(steps)
    la = np.zeros(steps)
    d_om = np.zeros(steps)
    d_la = np.zeros(steps)
    regime = np.zeros(steps, dtype=int)  # 0=Lock,1=Drift,2=Iterate

    var_om_roll = np.zeros(steps)
    mean_abs_dldt_roll = np.zeros(steps)

    # анти-стоп (если "застыли" слишком гладко)
    bumps_fired = 0
    last_bump_t = -1

    # буфер для скользящей метрики |dΛ′/dt|
    win_buf = []

    for t in range(steps):
        # шум
        n_om = noise_amp * np.random.randn()
        n_la = noise_amp * np.random.randn()

        # адаптация памяти
        # mem стремится к (Ω − Λ), распад со скоростью mem_decay
        mem += mem_decay * ((Omega - Lambda) - mem) * dt

        # динамика
        dOmega = -a_om*(Omega - 1.0) + c_la_to_om*(Lambda - Omega) + n_om
        dLambda = -a_la*Lambda + c_om_to_la*(Omega - Lambda) + 0.1*mem + n_la

        Omega += dOmega * dt
        Lambda += dLambda * dt

        om[t] = Omega
        la[t] = Lambda
        d_om[t] = dOmega
        d_la[t] = dLambda

        # расчёт rolling var Ω′
        if t+1 >= win:
            var_om_win = np.var(om[t-win+1:t+1])
        else:
            var_om_win = np.nan
        var_om_roll[t] = var_om_win

        # rolling mean |dΛ′/dt|
        win_buf.append(abs(dLambda))
        if len(win_buf) > win:
            win_buf.pop(0)
        mean_abs_dldt_roll[t] = float(np.mean(win_buf))

        # классификация режима по текущим "окнам"
        vw = var_om_win if not np.isnan(var_om_win) else 0.0
        mabs = mean_abs_dldt_roll[t]
        r = classify_regime(vw, mabs, hyst)
        regime[t] = 0 if r == "Lock" else (1 if r == "Drift" else 2)

        # анти-стоп: очень долгое "тихое" окно
        if (t > win*2) and (vw < 1e-8) and (mabs < 1e-4):
            Omega += 0.02*np.sign(np.random.randn())
            Lambda += 0.02*np.sign(np.random.randn())
            bumps_fired += 1
            last_bump_t = t

    # метрики
    # финальные окна (если NaN — берём последнее не-NaN)
    valid = ~np.isnan(var_om_roll)
    var_win_val = float(var_om_roll[valid][-1] if np.any(valid) else 0.0)
    mean_abs_dldt = float(mean_abs_dldt_roll[-1])

    # корреляция производных
    try:
        corr = float(np.corrcoef(d_om, d_la)[0,1])
        if math.isnan(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    # доли режимов
    lock_share = 100.0 * np.mean(regime == 0)
    drift_share = 100.0 * np.mean(regime == 1)
    iterate_share = 100.0 * np.mean(regime == 2)

    # число переключений режимов
    transitions = int(np.sum(np.diff(regime) != 0))
    transitions_per_1k = transitions / (steps/1000.0)

    # вердикт
    pass_var = (ALIVE_VAR_MIN < var_win_val < ALIVE_VAR_MAX)
    pass_mean = (mean_abs_dldt > ALIVE_MEAN_DLDL_MIN)
    pass_drift = (drift_share >= ALIVE_DRIFT_MIN)
    alive = pass_var and pass_mean and pass_drift
    verdict = "[ALIVE]" if alive else "[NOT ALIVE]"

    # подготовим текст отчёта
    lines = []
    lines.append("===== ACE REPORT v0.4e-fix =====")
    lines.append(f"Drift share:          {fmt(drift_share,2)} %")
    lines.append(f"Lock share:           {fmt(lock_share,2)} %")
    lines.append(f"Iterate share:        {fmt(iterate_share,2)} %\n")
    lines.append(f"var_win(Ω′):          {fmt(var_win_val)}")
    lines.append(f"mean |dΛ′/dt|:        {fmt(mean_abs_dldt)}")
    lines.append(f"Corr(dΩ′, dΛ′):       {fmt(corr,3)}\n")
    lines.append(f"Regime transitions /1k steps: {transitions_per_1k:.0f}\n")
    lines.append("Alive rule:")
    lines.append(f"  var_win(Ω′) in (1e-6 .. 1e-3):   [{'OK' if pass_var else 'FAIL'}]")
    lines.append(f"  mean |dΛ′/dt| > 1e-2:            [{'OK' if pass_mean else 'FAIL'}]")
    lines.append(f"  Drift share ≥ 20%:               [{'OK' if pass_drift else 'FAIL'}]\n")
    lines.append(f"VERDICT: {verdict}\n")
    lines.append("Notes:")
    lines.append(f"- anti-stall bumps: {bumps_fired} fired")
    lines.append(f"- last bump at t = {last_bump_t}")
    lines.append(f"- params: COUP_LA_TO_OM={params['COUP_LA_TO_OM']}, "
                 f"COUP_OM_TO_LA={params['COUP_OM_TO_LA']}, MEM_DECAY={params['MEM_DECAY']}, "
                 f"HYST={params['HYST']}, NOISE={params['NOISE']}")
    lines.append("================================\n")
    report_text = "\n".join(lines)

    # сохранить CSV с историей (минимально)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t","Omega","Lambda","dOmega","dLambda","regime","var_win","mean_abs_dldt"])
        for t in range(steps):
            w.writerow([t, om[t], la[t], d_om[t], d_la[t], int(regime[t]),
                        0.0 if np.isnan(var_om_roll[t]) else var_om_roll[t],
                        mean_abs_dldt_roll[t]])

    save_report_text(report_text)

    metrics = Metrics(
        drift_share=drift_share,
        lock_share=lock_share,
        iterate_share=iterate_share,
        var_win=var_win_val,
        mean_abs_dldt=mean_abs_dldt,
        corr_d_om_d_la=corr,
        transitions_per_1k=transitions_per_1k,
        bumps_fired=bumps_fired,
        last_bump_t=last_bump_t
    )

    return verdict, metrics, {
        "Omega": om, "Lambda": la, "dOmega": d_om, "dLambda": d_la,
        "regime": regime, "var_roll": var_om_roll, "mean_abs_dldt_roll": mean_abs_dldt_roll
    }

# ---------------------------
# АВТО-ПОДБОР ДО [ALIVE]
# ---------------------------

def autotune_until_alive(base_params: Dict[str,float]) -> Dict[str,float]:
    """
    Перебирает небольшую, но «умную» сетку вокруг рабочих значений.
    Останавливается на первом найденном [ALIVE], сохраняет best_params.json.
    """
    # сетка компактная, чтобы не гонять сотни прогонов впустую
    grid = {
        "COUP_LA_TO_OM": [round(base_params["COUP_LA_TO_OM"] + d, 2) for d in (-0.03, -0.01, 0.0, +0.01, +0.03)],
        "COUP_OM_TO_LA": [round(base_params["COUP_OM_TO_LA"] + d, 2) for d in (-0.06, -0.03, 0.0, +0.03, +0.06)],
        "MEM_DECAY":     [round(x,2) for x in np.clip(
                             np.array([base_params["MEM_DECAY"] + d for d in (-0.12,-0.08,-0.04,0.0,+0.04,+0.08,+0.12)]),
                             0.05, 0.8)],
        "HYST":          [round(x,3) for x in np.clip(
                             np.array([base_params["HYST"] + d for d in (-0.003,-0.002,-0.001,0.0,+0.001,+0.002)]),
                             0.002, 0.02)],
        "NOISE":         [round(x,3) for x in np.clip(
                             np.array([base_params["NOISE"] + d for d in (-0.005,-0.003,-0.002,0.0,+0.002,+0.003)]),
                             0.003, 0.05)],
    }

    tried = 0
    best = None
    best_score = -1e9

    from itertools import product
    for c1, c2, md, hy, nz in product(
        grid["COUP_LA_TO_OM"], grid["COUP_OM_TO_LA"],
        grid["MEM_DECAY"], grid["HYST"], grid["NOISE"]
    ):
        tried += 1
        params = base_params.copy()
        params.update({"COUP_LA_TO_OM": c1, "COUP_OM_TO_LA": c2,
                       "MEM_DECAY": md, "HYST": hy, "NOISE": nz})
        verdict, m, _ = run_simulation(params, steps=STEPS)

        pass_var  = (ALIVE_VAR_MIN < m.var_win < ALIVE_VAR_MAX)
        pass_mean = (m.mean_abs_dldt > ALIVE_MEAN_DLDL_MIN)
        pass_drift= (m.drift_share >= ALIVE_DRIFT_MIN)

        if pass_var and pass_mean and pass_drift:
            # нашли живое — зафиксировали и выходим
            with open("best_params.json","w",encoding="utf-8") as f:
                json.dump(params, f, indent=2)
            print("\n✅ ALIVE FOUND")
            print(json.dumps(params, indent=2))
            print(f"Metrics: var={m.var_win:.3e}, mean|dΛ′/dt|={m.mean_abs_dldt:.5f}, drift={m.drift_share:.2f}%")
            return params

        # иначе — накапливаем лучший «почти живой»
        # скоринг: по выполненным условиям + близость var к центру и величина mean
        score = 0.0
        score += 1.0 if pass_var else -abs(math.log10(max(m.var_win,1e-12)) - math.log10(5e-5))
        score += 1.0 if pass_mean else (m.mean_abs_dldt / ALIVE_MEAN_DLDL_MIN)
        score += 1.0 if pass_drift else (m.drift_share / ALIVE_DRIFT_MIN)

        if score > best_score:
            best_score = score
            best = (params, m)

        # небольшая печать прогресса
        if tried % 25 == 0:
            print(f"... tried {tried} combos; best so far: "
                  f"var={best[1].var_win:.2e}, mean={best[1].mean_abs_dldt:.5f}, drift={best[1].drift_share:.1f}%")

    # ничего живого не нашли — оставим лучший кандидат, но предупредим
    if best:
        params, m = best
        with open("best_params.json","w",encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print("\n❌ ALIVE not found within grid. Best candidate saved to best_params.json:")
        print(json.dumps(params, indent=2))
        print(f"Metrics: var={m.var_win:.3e}, mean|dΛ′/dt|={m.mean_abs_dldt:.5f}, drift={m.drift_share:.2f}%")
        return params
    else:
        print("\n❌ ALIVE not found and no best candidate (unexpected). Using base params.")
        return base_params

# ---------------------------
# Точка входа
# ---------------------------

def main():
    params = load_params()
    if "--tune" in sys.argv:
        print(">>> AUTOTUNE mode: searching for [ALIVE] configuration ...")
        params = autotune_until_alive(params)
        # финальный подтверждающий прогон
        verdict, m, _ = run_simulation(params, steps=STEPS)
        print(f"\nFinal verdict after autotune: {verdict}")
        print(f"var={m.var_win:.3e}, mean|dΛ′/dt|={m.mean_abs_dldt:.5f}, drift={m.drift_share:.2f}%")
    else:
        verdict, m, _ = run_simulation(params, steps=STEPS)
        print(open("last_report.txt","r",encoding="utf-8").read())

if __name__ == "__main__":
    main()
