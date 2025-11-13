# diffinst/solvers/dedalus_backend.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import logging

import numpy as np

from ..config import Config
from ..linear_ops import evp_solve_at_k
from ..io_utils import StreamWriter

# Dedalus
try:
    import dedalus.public as d3
    _HAS_DEDALUS = True
except Exception:
    _HAS_DEDALUS = False

def ensure_dedalus_available():
    if not _HAS_DEDALUS:
        raise ImportError(
            "Dedalus is not available. "
            "Install it in your active environment:\n\n"
            "  conda activate dedalus3\n"
            "  pip install dedalus\n"
        )

@dataclass(frozen=True)
class DedalusRunArgs:
    stop_time: float
    dt: float
    save_stride: int
    # IC / seeding controls (parity with NonlinearNative)
    amp: float = 1e-3            # if amp_is_physical: physical |Sigma - S0|
    seed: int | None = None
    k0: int = 1
    seed_mode: str = "eigen"     # "eigen" or "cos" or "noise"
    k_phys: float | None = None
    amp_is_physical: bool = True
    amp_metric: str = "max"      # "max" or "rms"
    init_state: dict | None = None  # real-space arrays: Sigma, vx, vy, uy
    print_stride: int = 200

class DedalusBackend:
    """
    1D periodic IVP in Dedalus (RealFourier), HB form with shear/drag on LHS (implicit).
    """

    def __init__(self, cfg: Config, outdir: Path, args: DedalusRunArgs):
        ensure_dedalus_available()
        self.cfg = cfg
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.args = args

        # Domain / grid
        self.Nx = int(cfg.Nx)
        self.Lx = float(cfg.Lx)
        self.dtype = np.float64

        # Dedalus objects
        self._build_domain()
        self._build_fields()
        self._build_operators_and_params()
        self._init_state()               # sets initial field data on grid
        self._build_problem_and_solver() # compiles equations

        # IO
        man = self._manifest()                          # build the dict
        self.writer = StreamWriter(self.outdir, man)    # pass manifest to writer
        self._write_checkpoint(0, self.solver.sim_time)
        self.writer.append_metric(self._metrics_row(self.solver.sim_time))

        # log
        self.log = logging.getLogger(__name__)

    # -------------------- Dedalus domain / fields --------------------

    def _build_domain(self):
        # --- NEW: exact-fit override so Lx matches native runs ---
        Lx_eff = self.Lx
        ef = getattr(self.cfg, "exact_fit", None)
        if isinstance(ef, dict) and ef.get("enable", False):
            k_target = float(ef["K_target"])
            harm = int(ef.get("harmonics", 2))
            # same rule as Grid1D.exact_fit_box: Lx so that m wavelengths fit exactly
            lam = 2.0 * np.pi / max(k_target, 1e-30)
            Lx_eff = harm * lam
        self.Lx = Lx_eff
        # ---------------------------------------------------------

        coords = d3.CartesianCoordinates("x", "y")
        dist = d3.Distributor(coords, dtype=self.dtype)
        xbasis = d3.RealFourier(coords["x"], size=self.Nx,
                                bounds=(-0.5 * self.Lx, 0.5 * self.Lx),
                                dealias=3/2)
        self.coords = coords
        self.dist = dist
        self.xbasis = xbasis
        self.x   = dist.local_grid(xbasis)   # (Nx,1)
        self.x1d = np.ravel(self.x)          # (Nx,)

    def _build_fields(self):
        # Dust fields: Σ and v=(vx, vy); Gas: uy (axisymmetric closure)
        self.sig = self.dist.Field(name="sig", bases=self.xbasis)
        self.v   = self.dist.VectorField(self.coords, name="v", bases=self.xbasis)
        self.uy  = self.dist.Field(name="uy", bases=self.xbasis)

        # Unit vectors
        ex, ey = self.coords.unit_vector_fields(self.dist)
        self.ex, self.ey = ex, ey
        self.vx = self.v @ self.ex
        self.vy = self.v @ self.ey

    def _build_operators_and_params(self):
        # Derivatives
        self.dx = lambda f: d3.Differentiate(f, self.coords["x"])
        self.lap = lambda f: d3.Laplacian(f)

        # Parameters
        self.S0 = float(getattr(self.cfg, "S0", getattr(self.cfg, "sig_0", 1.0)))
        self.D0 = float(getattr(self.cfg, "D0", getattr(self.cfg, "D_0", 0.0)))
        self.nu0 = float(getattr(self.cfg, "nu0", getattr(self.cfg, "nu_0", 0.0)))
        self.beta_diff = float(getattr(self.cfg, "beta_diff", 0.0))
        self.beta_visc = float(getattr(self.cfg, "beta_visc", 0.0))

        self.Omega = float(self.cfg.Omega)
        self.q = float(self.cfg.q)
        self.ts = float(self.cfg.ts)
        self.eps = float(getattr(self.cfg, "eps", getattr(self.cfg, "epsilon", 0.0)))
        self.nu_g = float(self.cfg.nu_g)

        # Smooth positive surrogate for Σ:   Σ_eff ≈ max(Σ, Σ_floor) but C^1-smooth
        self.sig_floor = 1e-12
        self.smooth_eps = 1e-12

        def sig_eff(sig):
            s = sig - self.sig_floor
            return 0.5*(s + d3.sqrt(s*s + self.smooth_eps*self.smooth_eps)) + self.sig_floor

        self.sig_eff = sig_eff

        # Use Σ_eff everywhere we divide or exponentiate
        self.inv_sig = lambda sig: 1.0 / self.sig_eff(sig)

        def D_of(sig):
            se = self.sig_eff(sig)
            return self.D0 * d3.exp(self.beta_diff * d3.log(se / self.S0))

        def nu_of(sig):
            se = self.sig_eff(sig)
            return self.nu0 * d3.exp(self.beta_visc * d3.log(se / self.S0))

        self.D_of = D_of
        self.nu_of = nu_of

    # -------------------- Initialization --------------------

    def _init_state(self):
        rng = np.random.default_rng(self.args.seed)
        S0 = self.S0

        if self.args.init_state is not None:
            Sigma = np.asarray(self.args.init_state["Sigma"], dtype=self.dtype).reshape(-1, 1)
            vx = np.asarray(self.args.init_state["vx"], dtype=self.dtype).reshape(-1, 1)
            vy = np.asarray(self.args.init_state["vy"], dtype=self.dtype).reshape(-1, 1)
            uy = np.asarray(self.args.init_state["uy"], dtype=self.dtype).reshape(-1, 1)

        elif self.args.seed_mode == "eigen" and (self.args.k_phys is not None):
            k = float(self.args.k_phys)
            w, V = evp_solve_at_k(self.cfg, k)
            v = V[:, 0]  # [S, vx, vy, uy] of the dominant mode

            expikx = np.exp(1j * k * (self.x - float(self.x.min())))
            S_raw = (v[0] * expikx).real
            vx_raw = (v[1] * expikx).real
            vy_raw = (v[2] * expikx).real
            uy_raw = (v[3] * expikx).real

            if self.args.amp_is_physical:
                if self.args.amp_metric == "rms":
                    a_now = float(np.sqrt(np.mean(S_raw**2)))
                else:
                    a_now = float(np.max(np.abs(S_raw)))
                scale = (self.args.amp / max(a_now, 1e-30)) if a_now != 0 else 0.0
            else:
                a_target = self.args.amp * S0
                if self.args.amp_metric == "rms":
                    a_now = float(np.sqrt(np.mean(S_raw**2)))
                else:
                    a_now = float(np.max(np.abs(S_raw)))
                scale = (a_target / max(a_now, 1e-30)) if a_now != 0 else 0.0

            Sigma = (S0 + scale * S_raw).reshape(-1, 1)
            vx = (scale * vx_raw).reshape(-1, 1)
            vy = (scale * vy_raw).reshape(-1, 1)
            uy = (scale * uy_raw).reshape(-1, 1)

        elif self.args.seed_mode == "noise":
            # Random noise in Sigma, zero velocities
            if self.args.amp_is_physical:
                amp_phys = self.args.amp
            else:
                amp_phys = self.args.amp * S0

            # N(0,1) noise scaled by amp_phys
            Sigma_1d = S0 + amp_phys * rng.standard_normal(self.Nx)
            Sigma = Sigma_1d.reshape(-1, 1)
            vx = np.zeros_like(Sigma)
            vy = np.zeros_like(Sigma)
            uy = np.zeros_like(Sigma)

        else:
            # Cosine seed
            k0 = int(self.args.k0)
            phase = 0.0
            if self.args.amp_is_physical:
                amp_phys = self.args.amp
            else:
                amp_phys = self.args.amp * S0

            Sigma_1d = self.S0 + amp_phys * np.cos( 2*np.pi*self.args.k0*(self.x1d - self.x1d.min())/self.Lx + phase)
            Sigma = Sigma_1d.reshape(-1, 1)
            vx = np.zeros_like(Sigma); vy = np.zeros_like(Sigma); uy = np.zeros_like(Sigma)

        # Load into Dedalus fields
        self.sig["g"] = Sigma
        self.v["g"][0] = vx
        self.v["g"][1] = vy
        self.uy["g"] = uy

    # -------------------- Equations & solver --------------------

    def _build_problem_and_solver(self):
        dx  = self.dx
        lap = self.lap
        ex, ey = self.ex, self.ey

        sig = self.sig
        vx  = self.v @ ex
        vy  = self.v @ ey
        uy  = self.uy

        # Closures
        D  = self.D_of(sig)
        nu = self.nu_of(sig)

        # Convenience
        grad = lambda f: d3.Gradient(f)
        dSdx = dx(sig)

        # --- CONTINUITY (RHS) ---
        cont_rhs = -dx(vx * sig)  # matches your script

        # --- MOMENTUM HB FORM ---
        # LHS (implicit)
        shear_hb     = -2.0*self.Omega*vy*ex + 0.5*self.Omega*vx*ey
        drag_on_dust = (vx*ex + (vy - uy)*ey) / self.ts

        # RHS (explicit) exactly as your script
        dust_advection         = -(self.v @ grad(self.v))
        nonlinear_diffusion_hb = -(1.0/sig) * dx( (D*D/sig) * (dx(sig)*dx(sig)) ) * ex
        dust_pressure_hb       = -(1.0/sig) * (2.0 + self.beta_diff) * (D/self.ts) * dx(sig) * ex
        dust_viscosity_hb      = (1.0/sig) * (
            (4.0/3.0) * dx( sig*nu*dx( vx + (D/sig)*dx(sig) ) ) * ex
            + dx( sig*nu*( dx(vy) - (self.q*self.Omega) ) ) * ey
        )

        # --- GAS AZIMUTHAL ---
        drag_on_gas = (self.eps/self.ts) * (uy - vy)  # LHS
        gas_visc    = self.nu_g * lap(uy)             # RHS

        # Problem and solver
        problem = d3.IVP([sig, self.v, uy], namespace=locals())

        # continuity
        problem.add_equation("dt(sig) = cont_rhs")

        # momentum (HB form) with shear + drag_on_dust on LHS
        problem.add_equation(
            "dt(v) + shear_hb + drag_on_dust = dust_advection + nonlinear_diffusion_hb + dust_pressure_hb + dust_viscosity_hb"
        )

        # gas azimuthal: drag_on_gas on LHS
        problem.add_equation("dt(uy) + drag_on_gas = gas_visc")

        # Fixed RK443 with user dt
        #timestepper = d3.RK443
        #
        timestepper = d3.RK222
        self.solver = problem.build_solver(timestepper)

        # Stop condition: sim_time
        self.solver.stop_sim_time = float(self.args.stop_time)

    # -------------------- I/O --------------------

    def _manifest(self) -> Dict[str, object]:
        return {
            "kind": "nonlinear_dedalus",
            "config": self.cfg.source_file,
            "Nx": self.Nx,
            "Lx": self.Lx,
            "backend": "dedalus",
            "dt": self.args.dt,
            "tstop": self.args.stop_time,
            "save_stride": self.args.save_stride,
            "amp": self.args.amp,
            "seed_mode": self.args.seed_mode,
            "k_phys": self.args.k_phys,
        }

    def _write_checkpoint(self, istep: int, t: float):
        fn = self.outdir / "checkpoints" / f"chk_{istep:06d}.npz"
        fn.parent.mkdir(parents=True, exist_ok=True)

        # Get real-space data at output scale
        self.v.change_scales(1); self.sig.change_scales(1); self.uy.change_scales(1)

        Sigma = np.copy(self.sig["g"]).squeeze(-1)
        vx    = np.copy(self.v["g"][0]).squeeze(-1)
        vy    = np.copy(self.v["g"][1]).squeeze(-1)
        uy    = np.copy(self.uy["g"]).squeeze(-1)
        x     = np.copy(self.x).squeeze(-1)

        np.savez_compressed(fn, t=float(t), x=x, Sigma=Sigma, vx=vx, vy=vy, uy=uy)

    def _metrics_row(self, t: float) -> Dict[str, float]:
        # Simple spectral amplitude diagnostic at the first nonzero bin
        self.sig.change_scales(1)
        g = np.copy(self.sig["g"]).squeeze(-1)
        s = g - g.mean()
        ak = np.fft.rfft(s)
        amp1 = float(np.abs(ak[1])) if ak.size > 1 else float(np.abs(ak[0]))
        return {"t": float(t), "mode1_amp": amp1, "mass": float(np.mean(self.sig["g"]))}

    # -------------------- Run loop --------------------

    def run(self):
        t = self.solver.sim_time
        dt = float(self.args.dt)
        save_stride = int(self.args.save_stride)
        print_stride = int(self.args.print_stride)

        istep = 0
        while self.solver.proceed:
            istep += 1
            self.solver.step(dt)
            t = self.solver.sim_time

            # Pull data onto numpy for checks
            self.v.change_scales(1); self.sig.change_scales(1); self.uy.change_scales(1)
            gsig = np.copy(self.sig["g"]).squeeze(-1)
            gvx  = np.copy(self.v["g"][0]).squeeze(-1)
            gvy  = np.copy(self.v["g"][1]).squeeze(-1)
            guy  = np.copy(self.uy["g"]).squeeze(-1)

            # Progress log
            if istep % print_stride == 0:
                self.log.info("Dedalus step=%d  t=%.6e  dt=%.3e", istep, t, dt)

            # NaN/Inf guard — abort cleanly
            if not (np.isfinite(gsig).all() and np.isfinite(gvx).all()
                    and np.isfinite(gvy).all() and np.isfinite(guy).all()):
                self.log.error("Non-finite values detected at step=%d t=%.6e — aborting.", istep, t)
                self._write_checkpoint(istep, t)  # last snapshot
                raise FloatingPointError("Detected NaN/Inf in fields during Dedalus run.")

            # metrics (only after sanity check)
            self.writer.append_metric(self._metrics_row(t))

            # checkpoint
            if (istep % save_stride == 0) or (t >= self.args.stop_time):
                self._write_checkpoint(istep, t)

        return {"t_final": float(t), "steps": int(istep), "Nx": self.Nx, "Lx": self.Lx}

# --------- Convenience function to match your runtime orchestration ---------

def run_dedalus_backend(
    cfg: Config,
    outdir: Path,
    stop_time: float,
    dt: float,
    save_stride: int,
    amp: float = 1e-3,
    seed: int | None = None,
    k0: int = 1,
    seed_mode: str = "eigen",
    k_phys: float | None = None,
    amp_is_physical: bool = True,
    amp_metric: str = "max",
    init_state: dict | None = None,
):
    ensure_dedalus_available()
    args = DedalusRunArgs(
        stop_time=stop_time,
        dt=dt,
        save_stride=save_stride,
        amp=amp,
        seed=seed,
        k0=k0,
        seed_mode=seed_mode,
        k_phys=k_phys,
        amp_is_physical=amp_is_physical,
        amp_metric=amp_metric,
        init_state=init_state,
    )
    solver = DedalusBackend(cfg, Path(outdir), args)
    return solver.run()