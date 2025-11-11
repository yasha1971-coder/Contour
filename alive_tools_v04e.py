# alive_tools_v04e.py
import csv, math
from pathlib import Path

ALIVE_VAR_MIN = 1e-6
ALIVE_VAR_MAX = 1e-3
ALIVE_VEL_MIN = 1e-2
ALIVE_DRIFT_MIN = 0.20

def _in_alive_band(var_win, vmean, drift_share):
    return (ALIVE_VAR_MIN <= var_win <= ALIVE_VAR_MAX) and (vmean > ALIVE_VEL_MIN) and (drift_share >= ALIVE_DRIFT_MIN)

def alive_plus(metrics):
    """
    metrics: dict with keys:
      'drift_share','var_win','mean_abs_dL_dt','corr','regime_transitions'
    Возвращает скаляр ALIVE+ в [0..1] (эвристика).
    """
    drift = max(0.0, min(1.0, metrics["drift_share"]))  # 0..1 уже в долях
    # нормируем var_win лог-шкалой к центру окна
    varv = metrics["var_win"]
    # если вне окна — штрафуем
    if varv <= 0:
        var_score = 0.0
    else:
        mid = math.sqrt(ALIVE_VAR_MIN * ALIVE_VAR_MAX)  # геом. центр
        span = math.log10(ALIVE_VAR_MAX/ALIVE_VAR_MIN)
        var_score = max(0.0, 1.0 - abs(math.log10(varv/mid))/span*2.0)  # 1 в центре, 0 у границ и дальше

    vel = metrics["mean_abs_dL_dt"]
    vel_score = max(0.0, min(1.0, (vel - ALIVE_VEL_MIN) / ALIVE_VEL_MIN))  # 0 при пороге, 1 при 2x порога

    # режимные переходы — лёгкий бонус за «живую шероховатость»
    rt = metrics.get("regime_transitions", 0.0)
    rt_score = max(0.0, min(1.0, rt / 400.0))  # 400/1k шагов ≈ 1.0

    # корреляция — хотим небольшие |corr| (слишком сильная связь = жёсткая связка; слишком хаос тоже плохо)
    corr = abs(metrics.get("corr", 0.0))
    corr_score = max(0.0, 1.0 - min(1.0, corr))  # чем ближе к 0, тем лучше

    # веса
    w_drift, w_var, w_vel, w_rt, w_corr = 0.30, 0.25, 0.25, 0.10, 0.10
    score = (w_drift*drift + w_var*var_score + w_vel*vel_score + w_rt*rt_score + w_corr*corr_score)
    return round(max(0.0, min(1.0, score)), 4)

def estimate_recovery_time(data_csv_path, shock_t, window=200):
    """
    data.csv -> ищем первый момент после конца шока, где окно длиной `window`
    удовлетворяет всем ALIVE критериям (по метрикам, агрегированным в summary).
    Ожидаем, что в data.csv есть столбцы: 't','var_win','dL_abs','drift_share'
    Если их нет, адаптируй имена к своим.
    """
    p = Path(data_csv_path)
    if not p.exists():
        return None

    T, VAR, VEL, DRIFT = [], [], [], []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            T.append(int(row.get("t", len(T))))
            VAR.append(float(row.get("var_win", "0")))
            VEL.append(float(row.get("dL_abs", "0")))
            DRIFT.append(float(row.get("drift_share", "0")))

    start = max(0, int(shock_t))
    n = len(T)
    for i in range(start, n-window):
        vwin = max(ALIVE_VAR_MIN, min(ALIVE_VAR_MAX, sum(VAR[i:i+window])/window))  # средняя по окну (ограничим)
        vmean = sum(VEL[i:i+window])/window
        dshare = sum(DRIFT[i:i+window])/window
        if _in_alive_band(vwin, vmean, dshare):
            return T[i] - shock_t
    return None
