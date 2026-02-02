from __future__ import annotations
import numpy as np
from .config import Config

def L_of_k(cfg: Config, k: float) -> np.ndarray:
    """
    Returns linear operator M(k):

    - If cfg.enable_gas == True:
         4x4 operator for [S, vx, vy, uy]

    - If cfg.enable_gas == False:
         3x3 operator for [S, vx, vy]
    """

    k = float(k)
    ik = 1j * k

    Om  = float(cfg.Omega)
    q   = float(cfg.q)
    tS  = float(cfg.ts)

    D0    = float(cfg.D_0)
    nu    = float(cfg.nu_0)
    betad = float(cfg.beta_diff)
    betav = float(cfg.beta_visc)

    # gas-specific
    eps = float(getattr(cfg, "eps", 0.0))
    nug = float(getattr(cfg, "nu_g", 0.0))

    use_gas = bool(getattr(cfg, "enable_gas", True))

    # ---------------------------------------------------------
    # GAS-COUPLED MODEL: 4×4
    # ---------------------------------------------------------
    if use_gas:
        M = np.zeros((4, 4), dtype=np.complex128)

        # n S = i k vx
        M[0,0] = 0.0
        M[0,1] = ik

        # n vx
        M[1,0] = ik * (2.0 + betad) * D0 / tS + (4/3)*1j*(k**3)*nu*D0
        M[1,1] = - (1.0/tS) - (4/3)*nu*(k**2)
        M[1,2] = 2 * Om

        # n vy
        M[2,0] = ik * q * nu * Om * (1 + betav)
        M[2,1] = - (2 - q) * Om
        M[2,2] = - (1.0/tS) - nu*(k**2)
        M[2,3] = 1.0/tS

        # n uy
        M[3,2] = eps/tS
        M[3,3] = - eps/tS - nug*(k**2)

        return M

    # ---------------------------------------------------------
    # DUST-ONLY MODEL: 3×3
    # Fields: S, vx, vy
    # ---------------------------------------------------------
    M = np.zeros((3, 3), dtype=np.complex128)

    # n S = i k vx
    M[0,0] = 0.0
    M[0,1] = ik
    M[0,2] = 0.0

    # n vx
    M[1,0] = ik * (2.0 + betad) * D0 / tS + (4/3)*1j*(k**3)*nu*D0
    M[1,1] = -1.0/tS - (4/3)*nu*(k**2)
    M[1,2] = 2 * Om

    # n vy
    M[2,0] = ik * q * nu * Om * (1 + betav)
    M[2,1] = -(2 - q) * Om
    M[2,2] = -1.0/tS - nu*(k**2)

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
    return w[order], V[:, order]