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
    Compute diffusion and viscosity coefficients D, nu from the power-law closures:
        D  = D0  * (Sigma/S0)**beta_diff
        nu = nu0 * (Sigma/S0)**beta_visc
    with only a tiny floor to avoid literal division by zero.
    """
    S0        = float(params.get("S0", float(np.mean(Sigma))))
    D0        = float(_p(params, "D0", "D_0", 0.0))
    nu0       = float(_p(params, "nu0", "nu_0", 0.0))
    beta_diff = float(params.get("beta_diff", 0.0))
    beta_visc = float(params.get("beta_visc", 0.0))

    # tiny floor purely for numeric safety; this does *not* regularize the model,
    # it only avoids divide-by-zero if Sigma hits exactly 0
    sigma_floor = 1e-14 * max(S0, 1.0)
    Sigma_safe = np.where(Sigma > sigma_floor, Sigma, sigma_floor)

    ratio = Sigma_safe / S0
    D_raw  = D0  * ratio**beta_diff
    nu_raw = nu0 * ratio**beta_visc

    return D_raw, nu_raw


def rhs(state: dict, params: dict, ops):
    """
    Dust:
      dSigma/dt + d( Sigma*vx )/dx = 0
      dvx/dt + vx dvx/dx - 2Ω vy = Rx
      dvy/dt + vx dvy/dx + (2 - q)Ω vx = Ry

    Gas (axisymmetric incompressible, ux*=0):
      duy/dt = (ε/ts)(vy - uy) + νg lap(uy)

    If 'uy' is missing from state or is None, we interpret this as a
    dust-only model: we still allow dust to feel drag against a
    *fixed* gas background (uy = 0), but we do not evolve a gas equation.
    """

    Sigma = state["Sigma"]
    vx        = state["vx"]
    vy        = state["vy"]

    has_gas = ("uy" in state) and (state["uy"] is not None)
    uy      = state["uy"] if has_gas else np.zeros_like(Sigma)


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

    out = {
        "Sigma": dSigma_dt,
        "vx":    dvx_dt,
        "vy":    dvy_dt,
    }

    # Gas uy equation only if present
    if has_gas:
        duy_dt = (eps / ts) * (vy - uy) + nu_g * ops.lap(uy)
        out["uy"] = duy_dt

    return out


def rhs_split(state: dict, params: dict, ops):
    """
    IMEX split:
      implicit (frozen each step):
        vx: (4/3) * nu_bar * lap(vx)
        vy: nu_bar * lap(vy)
        uy: nu_g * lap(uy)   [only if gas is present]

      explicit: all remaining variable-coefficient and nonlinear terms.

    If 'uy' is missing or None, we treat it as dust-only:
      - no stiff uy term (uy not in `stiff`)
      - rhs() does not return 'uy'.
    """
    Sigma = state["Sigma"]
    has_gas = ("uy" in state) and (state["uy"] is not None)

    S0    = float(params.get("S0", float(np.mean(Sigma))))
    nu0   = float(_p(params, "nu0", "nu_0", 0.0))
    nu_bar = nu0  # freeze at Sigma ~ S0

    stiff = {
        "Sigma": 0.0,
        "vx":    (4.0 / 3.0) * nu_bar,
        "vy":    nu_bar,
    }
    if has_gas:
        stiff["uy"] = float(params.get("nu_g", 0.0))

    full = rhs(state, params, ops)
    return full, stiff