import numpy as np
from dataclasses import dataclass

@dataclass
class Grid1D:
    Nx: int
    Lx: float
    def x(self) -> np.ndarray:
        return np.linspace(-self.Lx/2, self.Lx/2, self.Nx, endpoint=False)
    def k_rfft(self) -> np.ndarray:
        dx = self.Lx / self.Nx
        return 2*np.pi * np.fft.rfftfreq(self.Nx, d=dx)
    def exact_fit_box(self, k_target: float, harmonics: int = 2) -> float:
        m = max(1, int(harmonics))
        return 2*np.pi * m / float(k_target)