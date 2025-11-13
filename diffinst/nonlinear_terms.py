# diffinst/nonlinear_terms.py
from __future__ import annotations
import numpy as np

def _p(params, a, b=None, default=None):
    if a in params:
        return params[a]
    if b and (b in params):
        return params[b]
    return default


def compute_D_nu(Sigma: np.ndarray, params: dict):
    """
    Compute diffusion and viscosity coefficients D, nu for a given Sigma,
    assuming Sigma has already been clamped to a positive floor.
    """
    S0 = float(params.get("S0", float(np.mean(Sigma))))
    D0 = float(_p(params, "D0", "D_0", 0.0))
    nu0 = float(_p(params, "nu0", "nu_0", 0.0))
    beta_diff = float(params.get("beta_diff", 0.0))
    beta_visc = float(params.get("beta_visc", 0.0))

    # power-law closures
    D_raw  = D0  * (Sigma / S0) ** beta_diff
    nu_raw = nu0 * (Sigma / S0) ** beta_visc

    # mild safety clip to avoid insane values if Sigma dips near the floor
    if D0 != 0.0:
        D = np.clip(D_raw, 1e-4 * D0, 1e4 * D0)
    else:
        D = D_raw

    if nu0 != 0.0:
        nu = np.clip(nu_raw, 1e-4 * nu0, 1e4 * nu0)
    else:
        nu = nu_raw

    return D, nu


def rhs(state: dict, params: dict, ops):
    """
    Dust:
      dSigma/dt + d( Sigma*vx )/dx = 0
      dvx/dt + vx dvx/dx - 2Ω vy = Rx
      dvy/dt + vx dvy/dx + (2 - q)Ω vx = Ry

    Gas (axisymmetric incompressible, ux*=0):
      duy/dt = (ε/ts)(vy - uy) + νg lap(uy)

    Rx = -(1/Σ) d/dx[ (D^2/Σ) (dΣ/dx)^2 ] - (vx/ts)
         - (1/Σ)(2+βdiff)(D/ts) dΣ/dx
         + (4/3)(1/Σ) d/dx[ ν Σ d/dx( vx + (D/Σ) dΣ/dx ) ]

    Ry = (uy - vy)/ts + (1/Σ) d/dx[ ν Σ ( dvy/dx - qΩ ) ]
    """
    Sigma = state["Sigma"]
    vx        = state["vx"]
    vy        = state["vy"]
    uy        = state["uy"]  # always present

    Omega = float(params.get("Omega", 0.0))
    q     = float(params.get("q", 1.5))
    ts    = float(params.get("ts", 1.0))
    eps   = float(params.get("eps", params.get("epsilon", 0.0)))
    nu_g  = float(params.get("nu_g", 0.0))

    # closures using clamped Sigma
    D, nu = compute_D_nu(Sigma, params)
    beta_diff = float(params.get("beta_diff", 0.0))

    invS = 1.0 / Sigma
    dSdx = ops.dx(Sigma)   # derivative of clamped Sigma

    # Continuity (now uses clamped Sigma as well)
    dSigma_dt = -ops.dx(Sigma * vx)

    # vx equation
    term_D_nl      = -invS * ops.dx((D * D / Sigma) * (dSdx * dSdx))
    term_drag_x    = -(vx) / ts  # ux* = 0 for axisymmetric incompressible gas
    term_gradDdrag = -(invS) * (2.0 + beta_diff) * (D / ts) * dSdx
    corr           = vx + (D * dSdx) * invS
    term_visc_x    = (4.0 / 3.0) * invS * ops.dx(nu * Sigma * ops.dx(corr))
    dvx_dt = -vx * ops.dx(vx) + 2.0 * Omega * vy + term_D_nl + term_drag_x + term_gradDdrag + term_visc_x

    # vy equation
    term_drag_y = (uy - vy) / ts
    term_visc_y = invS * ops.dx(nu * Sigma * (ops.dx(vy) - q * Omega))
    dvy_dt = -vx * ops.dx(vy) - (2.0 - q) * Omega * vx + term_drag_y + term_visc_y

    # uy equation (gas closure)
    duy_dt = (eps / ts) * (vy - uy) + nu_g * ops.lap(uy)

    return {"Sigma": dSigma_dt, "vx": dvx_dt, "vy": dvy_dt, "uy": duy_dt}


def rhs_split(state: dict, params: dict, ops):
    """
    IMEX split:
      implicit (frozen each step):
        vx: (4/3) * nu_bar * lap(vx)
        vy: nu_bar * lap(vy)
        uy: nu_g * lap(uy)
      explicit: all remaining variable-coefficient and nonlinear terms.
    """
    Sigma = state["Sigma"]
    S0    = float(params.get("S0", float(np.mean(Sigma))))
    nu0   = float(_p(params, "nu0", "nu_0", 0.0))
    nu_bar = nu0  # freeze at Sigma ~ S0

    stiff = {
        "Sigma": 0.0,
        "vx": (4.0 / 3.0) * nu_bar,
        "vy": nu_bar,
        "uy": float(params.get("nu_g", 0.0)),
    }
    full = rhs(state, params, ops)
    return full, stiff