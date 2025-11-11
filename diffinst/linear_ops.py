from __future__ import annotations
import numpy as np
from .config import Config

def L_of_k(cfg: Config, k: float) -> np.ndarray:
    """
    Returns the 4x4 matrix M(k) such that n X = M X for X = [S, vx, vy, uy]^T.
    S ≡ Σ̂_d / Σ_d,0. Complex dtype.
    """
    k = float(k)
    ik = 1j * k
    Om = float(cfg.Omega)
    q  = float(cfg.q)
    tS = float(cfg.ts)
    D0 = float(cfg.D_0)
    nu = float(cfg.nu_0)
    betad = float(cfg.beta_diff)
    betav = float(cfg.beta_visc)
    eps = float(cfg.eps)
    nug = float(cfg.nu_g)

    M = np.zeros((4,4), dtype=np.complex128)

    # n S = i k vx
    M[0,0] = 0.0
    M[0,1] = ik

    # n vx = 2Ω vy - (1/tS + 4/3 ν k^2) vx + (ik/tS)(2+β_diff) D0 S + (4/3) i k^3 ν D0 S
    M[1,0] = ik * (2.0 + betad) * D0 / tS + (4.0/3.0) * 1j * (k**3) * nu * D0
    M[1,1] = - (1.0/tS) - (4.0/3.0) * nu * (k**2)
    M[1,2] = 2.0 * Om

    # n vy = -(2-q)Ω vx + (1/tS)(uy - vy) - ν k^2 vy + i k q ν Ω (1+β_visc) S
    M[2,0] = 1j * k * q * nu * Om * (1.0 + betav)
    M[2,1] = - (2.0 - q) * Om
    M[2,2] = - (1.0/tS) - nu * (k**2)
    M[2,3] = 1.0 / tS

    # n uy = (ε/tS)(vy - uy) - ν_g k^2 uy
    M[3,2] = eps / tS
    M[3,3] = - (eps / tS) - nug * (k**2)

    return M

def evp_solve_at_k(cfg: Config, k: float):
    """
    Solve eigenproblem n X = M X and return eigenvalues and eigenvectors.
    Sorted by descending real(n).
    """
    from scipy.linalg import eig
    M = L_of_k(cfg, k)
    w, V = eig(M)
    order = np.argsort(w.real)[::-1]
    return w[order], V[:,order]