# env_adapter.py — внешний адаптер: энтропия/дисперсия/шок

from collections import deque
import numpy as np

class EnvAdapter:
    def __init__(self, window=50, shock_sigma=2.5, bins=16):
        self.window = int(window)
        self.shock_sigma = float(shock_sigma)
        self.bins = int(bins)
        self.buf = deque(maxlen=self.window)
        self.var_hist = deque(maxlen=self.window)

    @staticmethod
    def _entropy(arr, bins=16):
        if len(arr) < 4:
            return 0.0
        hist, _ = np.histogram(arr, bins=bins, density=True)
        p = hist / (hist.sum() + 1e-12)
        return float(-(p * np.log(p + 1e-12)).sum())

    def ingest(self, x):
        """Кормите сюда очередной внешний сэмпл (число). Возвращает dict со статами среды."""
        self.buf.append(float(x))
        if len(self.buf) < 4:
            return {"entropy": 0.0, "var": 0.0, "shock_z": 0.0}

        arr = np.array(self.buf, dtype=float)
        v = float(np.var(arr))
        self.var_hist.append(v)

        # шок как z-score текущей variance относительно истории
        if len(self.var_hist) > 3:
            m = float(np.mean(self.var_hist))
            s = float(np.std(self.var_hist) + 1e-12)
            z = (v - m) / (s if s > 1e-12 else 1.0)
        else:
            z = 0.0

        return {
            "entropy": self._entropy(arr, bins=self.bins),
            "var": v,
            "shock_z": float(z),
        }
