# diffinst/nonlinear_terms.py
from __future__ import annotations
import numpy as np

def _p(params, a, b=None, default=None):
    if a in params:
        return params[a]
    if b and (b in params):
        return params[b]
    return default


def smooth_gaussian(f: np.ndarray, Lx: float, frac: float = 0.25):
    """
    Spectral Gaussian smoother:
        f_smooth = IFFT[ exp(-(k/kc)^2) * FFT[f] ]

    Parameters
    ----------
    f : ndarray (real)
    Lx : domain size
    frac : fraction of Nyquist frequency at which the Gaussian is ~exp(-1)

    Returns
    -------
    ndarray : smoothed field, same shape
    """
    Nx = f.size
    dk = 2.0 * np.pi / Lx
    k = dk * np.fft.fftfreq(Nx)       # full symmetric k-array
    k_nyq = np.max(np.abs(k))
    k_c = frac * k_nyq                # characteristic cutoff

    g = np.exp(-(k / k_c)**2)         # Gaussian filter
    F = np.fft.fft(f)
    f_smooth = np.fft.ifft(F * g).real
    return f_smooth

def _validate_closure(Sigma_eff, S0, D_raw, nu_raw, where="compute_D_nu"):
    """Debug helper: raise with useful diagnostics if closures misbehave."""
    problems = []

    if not np.all(np.isfinite(Sigma_eff)):
        problems.append("Sigma_eff has non-finite values")

    ratio = Sigma_eff / S0
    if np.any(ratio <= 0):
        problems.append("Sigma_eff/S0 has non-positive values")

    if not np.all(np.isfinite(D_raw)):
        problems.append("D_raw has non-finite values")
    if not np.all(np.isfinite(nu_raw)):
        problems.append("nu_raw has non-finite values")

    if problems:
        msg = (
            f"[{where}] closure diagnostic failed:\n"
            f"  issues: {', '.join(problems)}\n"
            f"  Sigma_eff: min={Sigma_eff.min():.3e}, max={Sigma_eff.max():.3e}\n"
            f"  D_raw:     min={np.nanmin(D_raw):.3e}, max={np.nanmax(D_raw):.3e}\n"
            f"  nu_raw:    min={np.nanmin(nu_raw):.3e}, max={np.nanmax(nu_raw):.3e}"
        )
        raise FloatingPointError(msg)


def compute_D_nu(Sigma: np.ndarray, params: dict):
    """
    Compute diffusion & viscosity closures using a *smoothed* Sigma_eff.

    Steps:
      (1) Smooth Sigma spectrally
      (2) Enforce floor
      (3) Compute power-law closures
      (4) Validate
      (5) Clip to reasonable bandwidth around (D0, nu0)
    """
    S0 = float(params.get("S0", float(np.mean(Sigma))))
    D0 = float(_p(params, "D0", "D_0", 0.0))
    nu0 = float(_p(params, "nu0", "nu_0", 0.0))
    beta_diff = float(params.get("beta_diff", 0.0))
    beta_visc = float(params.get("beta_visc", 0.0))
    Lx = float(params.get("Lx", 1.0))        # require Lx in params
    smooth_frac = float(params.get("closure_smooth_frac", 0.2))

    # --- (1) Smooth Sigma using spectral Gaussian ---
    Sigma_smooth = smooth_gaussian(Sigma, Lx, frac=smooth_frac)

    # --- (2) Enforce a positive floor ---
    sigma_floor = 1e-12 * max(S0, 1.0)
    Sigma_eff = np.maximum(Sigma_smooth, sigma_floor)

    ratio = Sigma_eff / S0

    # --- (3) Power-law closures ---
    D_raw  = D0  * ratio**beta_diff
    nu_raw = nu0 * ratio**beta_visc

    # --- (4) Validation BEFORE clipping ---
    _validate_closure(Sigma_eff, S0, D_raw, nu_raw, where="compute_D_nu")

    # --- (5) Clip to reasonable ranges ---
    clip_lo = 1e-3
    clip_hi = 1e3
    D = np.clip(D_raw,  clip_lo * D0, clip_hi * D0) if D0 != 0 else D_raw
    nu = np.clip(nu_raw, clip_lo * nu0, clip_hi * nu0) if nu0 != 0 else nu_raw

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