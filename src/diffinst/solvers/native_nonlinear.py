from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import logging

import numpy as np
from numpy.fft import rfftfreq

from ..config import Config
from ..linear_ops import L_of_k, evp_solve_at_k
from ..io_utils import StreamWriter
from ..grid import Grid1D


from ..operators import PSOperators
from ..nonlinear_terms import rhs_split


@dataclass(frozen=True)
class NonlinearRunArgs:
    stop_time: float
    dt: float
    save_stride: int
    amp: float = 1e-3      # if amp_is_physical: physical |Sigma - S0|; else fractional of S0
    seed: int | None = None
    k0: int = 1
    init_metric: str = "max"
    seed_mode: str = "cos"     # "cos" or "eigen"  or "noise"      # 
    k_phys: float | None = None                          # 
    amp_is_physical: bool = True                         # 
    amp_metric: str = "max"                              #  ("max" or "rms")
    init_state: dict | None = None
    print_stride: int = None

class NonlinearNative:
    """
    Nonlinear pseudo-spectral IMEX integrator in x-space with gas uy always present.
    Explicit: advection, all variable-coefficient and nonlinear fluxes.
    Implicit (frozen): (4/3) nu_bar lap(vx), nu_bar lap(vy), nu_g lap(uy).
    """

    def __init__(self, cfg: Config, outdir: Path, args: NonlinearRunArgs,
                 grid: Grid1D | None = None,
                 writer: StreamWriter | None = None,
                 enable_gas = True):
        self.cfg = cfg
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.args = args

        self.enable_gas = bool(enable_gas)

        # grid
        if grid is None:
            self.Nx = int(cfg.Nx); self.Lx = float(cfg.Lx)
            self.dx = self.Lx / self.Nx
            self.x  = np.linspace(-0.5*self.Lx, 0.5*self.Lx, self.Nx, endpoint=False)
        else:
            self.Nx = int(getattr(grid, "Nx", cfg.Nx))
            self.Lx = float(getattr(grid, "Lx", cfg.Lx))
            self.dx = self.Lx / self.Nx
            self.x  = np.linspace(-0.5*self.Lx, 0.5*self.Lx, self.Nx, endpoint=False)

        # spectral operators
        self.ops = PSOperators(nx=self.Nx, Lx=self.Lx, dealias=True)

        # writer
        if writer is None:
            self.writer = StreamWriter(self.outdir)
            self._write_manifest()
        else:
            self.writer = writer

        # state (include uy always)
        self.state = self._init_state(init_state=getattr(self.args, "init_state", None))

        # log
        self.log = logging.getLogger(__name__)

    def _write_manifest(self):
        man = {
            "kind": "nonlinear_native",
            "config": self.cfg.source_file,
            "Nx": self.Nx, "Lx": self.Lx,
            "dt": self.args.dt, "stop_time": self.args.stop_time,
            "save_stride": self.args.save_stride,
            "amp": self.args.amp, "k0": self.args.k0,
        }
        (self.outdir/"run.json").write_text(json.dumps(man, indent=2))


    def _init_state(self, init_state=None):
        rng = np.random.default_rng(self.args.seed)
        S0   = float(getattr(self.cfg, "S0", getattr(self.cfg, "sig_0", 1.0)))
        amp  = float(self.args.amp)
        k0   = int(self.args.k0)
        x    = self.x
        use_gas = self.enable_gas

        # ---------- Case 1: init_state provided ----------
        if init_state is not None:
            Sigma = np.asarray(init_state["Sigma"])
            vx    = np.asarray(init_state["vx"])
            vy    = np.asarray(init_state["vy"])

            if use_gas:
                # if uy missing in file, just use zeros
                uy = np.asarray(init_state.get("uy", np.zeros_like(Sigma)))
                return {"Sigma": Sigma, "vx": vx, "vy": vy, "uy": uy}
            else:
                # dust-only: ignore any uy in file
                return {"Sigma": Sigma, "vx": vx, "vy": vy}

        # ---------- Case 2: seed_mode == "eigen" ----------
        if self.args.seed_mode == "eigen" and (self.args.k_phys is not None):
            k_phys = float(self.args.k_phys)

            # EVP eigenvector v = [S, vx, vy, uy] at k_phys
            w, V = evp_solve_at_k(self.cfg, k_phys)
            v = V[:, 0]

            expikx = np.exp(1j * (k_phys * (x - x.min())))
            S_raw  = (v[0] * expikx).real
            vx_raw = (v[1] * expikx).real
            vy_raw = (v[2] * expikx).real
            uy_raw = (v[3] * expikx).real

            # normalize
            if self.args.amp_is_physical:
                a_now = (np.sqrt(np.mean(S_raw**2))
                         if self.args.amp_metric == "rms"
                         else np.max(np.abs(S_raw)))
                scale = (amp / max(a_now, 1e-30)) if a_now != 0 else 0.0
            else:
                a_target = amp * S0
                a_now = (np.sqrt(np.mean(S_raw**2))
                         if self.args.amp_metric == "rms"
                         else np.max(np.abs(S_raw)))
                scale = (a_target / max(a_now, 1e-30)) if a_now != 0 else 0.0

            Sigma = S0 + scale * S_raw
            vx    = scale * vx_raw
            vy    = scale * vy_raw

            if use_gas:
                uy = scale * uy_raw
                return {"Sigma": Sigma, "vx": vx, "vy": vy, "uy": uy}
            else:
                return {"Sigma": Sigma, "vx": vx, "vy": vy}

        # ---------- Case 3: seed_mode == "noise" ----------
        elif self.args.seed_mode == "noise":
            if self.args.amp_is_physical:
                amp_phys = self.args.amp
            else:
                amp_phys = self.args.amp * S0

            Sigma = S0 + amp_phys * rng.standard_normal(self.Nx)
            vx    = np.zeros_like(Sigma)
            vy    = np.zeros_like(Sigma)
            if use_gas:
                uy = np.zeros_like(Sigma)
                return {"Sigma": Sigma, "vx": vx, "vy": vy, "uy": uy}
            else:
                return {"Sigma": Sigma, "vx": vx, "vy": vy}

        # ---------- Case 4: seed_mode == "cos" (default) ----------
        else:  # "cos"
            phase = 0.0
            amp_phys = amp if self.args.amp_is_physical else amp * S0

            Sigma = S0 + amp_phys * np.cos(k0 * 2.0*np.pi*(x - x.min())/self.Lx + phase)
            vx    = np.zeros_like(Sigma)
            vy    = np.zeros_like(Sigma)
            if use_gas:
                uy = np.zeros_like(Sigma)
                return {"Sigma": Sigma, "vx": vx, "vy": vy, "uy": uy}
            else:
                return {"Sigma": Sigma, "vx": vx, "vy": vy}
            

    def _params(self) -> dict:
        if not hasattr(self.cfg, "ts"):
            raise ValueError("Config missing 'ts' (stopping). Add ts: ... to YAML.")
        
        # expose physics knobs; mirror naming used in nonlinear_terms
        return dict(
            # closures
            D0=float(getattr(self.cfg, "D0", getattr(self.cfg, "D_0", 0.0))),
            beta_diff=float(getattr(self.cfg, "beta_diff", 0.0)),
            nu0=float(getattr(self.cfg, "nu0", getattr(self.cfg, "nu_0", 0.0))),
            beta_visc=float(getattr(self.cfg, "beta_visc", 0.0)),
            S0=float(getattr(self.cfg, "S0", getattr(self.cfg, "sig_0", 1.0))),
            # dynamics
            Omega=float(getattr(self.cfg, "Omega", 0.0)),
            q=float(getattr(self.cfg, "q", 1.5)),
            ts=float(getattr(self.cfg, "ts", 1.0)),
            eps=float(getattr(self.cfg, "eps", getattr(self.cfg, "epsilon", 0.0))),
            nu_g=float(getattr(self.cfg, "nu_g", 0.0)),
            # flags
            enable_gas=self.enable_gas,
            enable_piecewise_diffusion=bool(getattr(self.cfg, "enable_piecewise_diffusion", False)),
            sigma_sat_factor=float(getattr(self.cfg, "sigma_sat_factor", 2.0)),
        )

    # IMEX RK2 using rhs_split (same structure; now includes uy)
    def _step_imex_rk2(self, state, dt):
        full0, stiff = rhs_split(state, self._params(), self.ops)
        a = 0.5; b = 0.5

        def apply_implicit(yx, nu, fac_dt):
            if nu == 0.0: return yx
            return self.ops.invert_I_minus_a_dt_nu_lap(yx, fac_dt * nu)

        # stage A
        yA = {}
        for key in state:
            nu = stiff[key]
            Ly0 = nu * self.ops.lap(state[key]) if nu != 0.0 else 0.0
            rhsA = state[key] + a * dt * (full0[key] - Ly0)
            yA[key] = apply_implicit(rhsA, nu, a*dt)

        # stage B
        fullA, _ = rhs_split(yA, self._params(), self.ops)
        y1 = {}
        for key in state:
            nu = stiff[key]
            Ly0 = nu * self.ops.lap(state[key]) if nu != 0.0 else 0.0
            rhsB = state[key] + dt * (b * fullA[key] + (1.0 - b) * full0[key] - b * Ly0)
            y1[key] = apply_implicit(rhsB, nu, b*dt)

        return y1

    def _write_checkpoint(self, istep: int, t: float):
        fn = self.outdir / "checkpoints" / f"chk_{istep:06d}.npz"
        fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fn, t=t, x=self.x, **self.state)

    def _diag(self, t: float) -> dict:
        Sig = self.state["Sigma"]
        amp1 = np.abs(np.fft.rfft(Sig - Sig.mean())[1])
        return {"t": t, "mass": float(Sig.mean()), "mode1_amp": float(amp1)}

    def run(self):
        t = 0.0
        dt = float(self.args.dt)
        stop_time = float(self.args.stop_time)
        save_stride = int(self.args.save_stride)

        if self.args.print_stride:
            print_stride = int(self.args.print_stride)
        else:
            print_stride = None

        istep = 0
        self._write_checkpoint(istep, t)
        self.writer.write_metric(self._diag(t))

        while t < stop_time:
            istep += 1
            t_next = min(t + dt, stop_time)
            self.state = self._step_imex_rk2(self.state, t_next - t)
            t = t_next

            # Progress log
            if print_stride:
                if istep % print_stride == 0:
                    self.log.info("Native nonlinear step=%d  t=%.6e  dt=%.3e", istep, t, dt)

            self.writer.write_metric(self._diag(t))
            if istep % save_stride == 0 or t >= stop_time:
                self._write_checkpoint(istep, t)
            

        return {"t_final": t, "steps": istep, "Nx": self.Nx, "Lx": self.Lx}