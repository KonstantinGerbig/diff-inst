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
    
    def exact_fit_Lx(self, k_target: float, harmonics: int) -> float:
        lam = 2.0 * np.pi / max(k_target, 1e-30)
        return harmonics * lam