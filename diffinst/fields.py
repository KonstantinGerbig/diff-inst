import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class State:
    Nx: int
    sigma: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    uy: Optional[np.ndarray] = None

    @staticmethod
    def zeros(Nx: int, enable_gas: bool = True) -> "State":
        sigma = np.ones(Nx, dtype=np.float64)
        vx = np.zeros(Nx, dtype=np.float64)
        vy = np.zeros(Nx, dtype=np.float64)
        uy = np.zeros(Nx, dtype=np.float64) if enable_gas else None
        return State(Nx, sigma, vx, vy, uy)