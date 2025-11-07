# ace_kernel_v04e.py
import numpy as np
from collections import deque

# ---------- Параметры по умолчанию ----------
DEFAULTS = dict(
    EPS=0.74, DELTA=0.10, ETA=0.012,
    COUP_LA_TO_OM=0.08, COUP_OM_TO_LA=0.06,
    MEM_DECAY=0.92, HYST=0.016,
    NOISE=0.003,              # базовый фоновый шум
    DRIFT_TARGET=0.62,        # целевая доля Drift
    ELASTIC_OM=0.04,          # «бюджет» дыхания Ω′
    WIN_VAR=60, WIN_DRIFT=300,# окна для var и дрейфа
    MAX_STEPS=6000,
    MICRO_NOISE=1e-3,         # микро-импульс при застывании
    GRACE=10                  # минимальная длительность режима
)

# ---------- Вспомогательные инварианты ----------
def clip_norm(x, max_norm=10.0):
    n = np.linalg.norm(x)
    return x if n <= max_norm else x * (max_norm / (n + 1e-12))

def rolling_var(series, w):
    if len(series) < w: return np.nan
    a = np.array(series)[-w:]
    return float(np.var(a))

def rolling_corr(a, b, w):
    if len(a) < w or len(b) < w: return np.nan
    x, y = np.array(a)[-w:], np.array(b)[-w:]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0: return 0.0
    return float(np.corrcoef(x, y)[0,1])

def soft_barrier(value, lo, hi):
    """Плавное удержание Ω′ в [lo,hi] с эластичностью."""
    if value < lo:
        return lo + np.tanh(value - lo)
    if value > hi:
        return hi - np.tanh(value - hi)
    return value

# Lyapunov-подобная метрика когерентного дыхания
def L_metric(p_drift, rho, p_lock, sigma_l, sigma0=0.02, alpha=0.5, beta=2.0, gamma=0.3):
    return (1 - p_drift) + alpha*(1 - abs(rho)) + beta*(p_lock**2) + gamma*np.exp(-(sigma_l**2)/(sigma0**2 + 1e-12))

# ---------- Ядро ----------
class ACEngine:
    def __init__(self, **kw):
        self.p = DEFAULTS.copy(); self.p.update(kw)
        self.reset()

    def reset(self):
        self.t = 0
        self.OmegaPrime = 1.0
        self.LambdaPrime = 0.14
        self.regime = "Drift"
        self._last_switch = -1e9

        self.h_omega, self.h_lambda = deque(maxlen=20000), deque(maxlen=20000)
        self.h_regime = deque(maxlen=20000)
        self.h_domega, self.h_dlambda = deque(maxlen=20000)

        self.mem_lambda = 0.0
        self.drift_window = deque(maxlen=self.p['WIN_DRIFT'])

    # классификация с гистерезисом и grace-period
    def classify(self, dphi):
        eps, delta = self.p['EPS'], self.p['DELTA']
        low, high = eps - delta, eps + delta
        # липкость Drift: расширяем окно, если уже в Drift
        if self.regime == "Drift":
            low -= 0.02; high += 0.02
        if dphi < low:  r = "Lock"
        elif dphi > high: r = "Iterate"
        else: r = "Drift"
        # grace: не менять слишком часто
        if self.t - self._last_switch < self.p['GRACE']: 
            return self.regime
        if r != self.regime:
            self._last_switch = self.t
        return r

    def p_controller(self):
        # регулятор «цели Drift» на окне
        win = min(len(self.drift_window), self.p['WIN_DRIFT'])
        drift_share_win = sum(self.drift_window)/max(1,win)
        error = self.p['DRIFT_TARGET'] - drift_share_win
        if abs(error) < self.p['HYST']: return 0.0
        # фильтрация через NOISE (как альфа): чем больше NOISE, тем медленней
        adj = 0.6 * ((1-self.p['NOISE']) * error)
        # ограничим масштаб
        return np.clip(adj, -0.05, 0.05)

    def step(self, dphi):
        # --- режим
        self.regime = self.classify(dphi)
        self.drift_window.append(1.0 if self.regime=="Drift" else 0.0)

        # --- мягкая память Λ′
        self.mem_lambda = self.p['MEM_DECAY']*self.mem_lambda + (1-self.p['MEM_DECAY'])*(self.LambdaPrime - 0.14)

        # --- куплинг
        CO1, CO2 = self.p['COUP_LA_TO_OM'], self.p['COUP_OM_TO_LA']
        dOmega = CO1 * self.mem_lambda + (np.random.randn()*self.p['NOISE'])
        dLambda = CO2 * (self.OmegaPrime - 1.0) + (np.random.randn()*self.p['NOISE'])

        # режимная модуляция
        if self.regime=="Drift":
            dOmega += 0.004*np.sin(0.03*self.t)
            dLambda += 0.006*np.sin(0.025*self.t)
        elif self.regime=="Lock":
            dOmega *= 0.5; dLambda *= 0.5
        else: # Iterate
            dOmega *= 1.2; dLambda *= 1.2

        # --- анти-stall: если var_win(Ω′) слишком мал — микро-импульс
        var_win = rolling_var(self.h_omega, self.p['WIN_VAR'])
        if not np.isnan(var_win) and var_win < 1e-7:
            dOmega += np.sign(np.random.randn()) * self.p['MICRO_NOISE']

        # --- обновление и эластичность Ω′
        newOmega = self.OmegaPrime + dOmega
        center = 1.0
        newOmega = center + np.clip(newOmega-center, -self.p['ELASTIC_OM'], self.p['ELASTIC_OM'])
        newOmega = soft_barrier(newOmega, center-0.5, center+0.5)

        newLambda = self.LambdaPrime + dLambda
        # безопасные клипы
        newLambda = np.clip(newLambda, -1.0, 1.0)

        # лог и коммит
        self.h_domega.append(newOmega - self.OmegaPrime)
        self.h_dlambda.append(newLambda - self.LambdaPrime)
        self.OmegaPrime, self.LambdaPrime = newOmega, newLambda
        self.h_omega.append(self.OmegaPrime); self.h_lambda.append(self.LambdaPrime)
        self.h_regime.append(self.regime)
        self.t += 1

    def run(self, seed=0):
        np.random.seed(seed)
        for _ in range(self.p['MAX_STEPS']):
            # входной сигнал: смесь синуса и шума
            dphi = self.p['EPS'] + 0.12*np.sin(0.017*self.t) + np.random.randn()*0.02
            # адаптивный шум: усиливаем в Lock/Iterate
            if self.regime!="Drift": dphi += np.random.randn()*0.01
            # регулятор смещает пороги (эффект «липкости Drift»)
            adj = self.p_controller()
            self.p['EPS'] += adj*0.02
            self.p['DELTA'] = np.clip(self.p['DELTA'] - 0.5*adj, 0.06, 0.14)
            self.step(dphi)

        # QC
        drift = sum(1 for r in self.h_regime if r=="Drift")/len(self.h_regime)
        lock  = sum(1 for r in self.h_regime if r=="Lock")/len(self.h_regime)
        var_om = rolling_var(self.h_omega, self.p['WIN_VAR'])
        corr = rolling_corr(self.h_domega, self.h_dlambda, self.p['WIN_VAR'])
        sigma_l = np.std(np.array(self.h_dlambda)[-self.p['WIN_VAR']:])
        L = L_metric(drift, corr if not np.isnan(corr) else 0.0, lock, sigma_l)
        alive = (drift>=0.60) and (var_om<1e-3) and (np.nan_to_num(np.mean(np.abs(self.h_dlambda[-self.p['WIN_VAR']:])),0)>1e-2) and (corr>=0.25)

        return dict(
            drift=drift, lock=lock, var_win=var_om, corr=corr, L=L, alive=alive
        )
