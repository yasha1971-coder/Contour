# Simple external environment adapter (optional)
import numpy as np
from collections import deque

class EntropyAdapter:
    def __init__(self, window: int = 50):
        self.window = int(window)
        self.buf = deque(maxlen=self.window)

    def ingest(self, x: float):
        self.buf.append(float(x))

    def entropy(self) -> float:
        if len(self.buf) < 10:
            return 0.0
        arr = np.asarray(self.buf)
        hist, _ = np.histogram(arr, bins=12, density=True)
        p = hist + 1e-12
        return float(-np.sum(p * np.log(p)))

    def __call__(self, engine) -> float:
        # example: couple to internal omega' just to demonstrate
        self.ingest(engine.s.omega_p)
        return self.entropy()
