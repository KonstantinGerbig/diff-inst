# diffinst/solvers/native.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.fft import rfftfreq

from ..config import Config
from ..linear_ops import L_of_k, evp_solve_at_k
from ..io_utils import StreamWriter
from ..grid import Grid1D


@dataclass(frozen=True)
class LinearRunArgs:
    stop_time: float
    dt: float
    save_stride: int
    k_target: float | None = None
    amp: float = 1e-3
    seed: int | None = None
    amp_is_physical: bool = False
    amp_metric: str = 'max' # "max" or "rms" for the physical metric
    init_state: dict | None = None


class LinearNative:
    """
    Linear time-domain integrator for the 1D axisymmetric system.

    Method: exact per-k evolution in spectral space.
      For each k-bin, diagonalize L(k) once:
        X_k(t+dt) = V exp(Λ dt) V^{-1} X_k(t)
      where X_k = [S, vx, vy, uy]^T and n X = L X is the EVP form.
    """

    def __init__(self, cfg: Config, outdir: Path, args: LinearRunArgs,
                 grid: Grid1D | None = None,
                 writer: StreamWriter | None = None):
        self.cfg = cfg
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.args = args

        # --- grid (use provided orchestrated grid if available)
        if grid is None:
            self.Nx = int(cfg.Nx)
            self.Lx = float(cfg.Lx)
            self.dx = self.Lx / self.Nx
            self.x = np.linspace(-0.5 * self.Lx, 0.5 * self.Lx, self.Nx, endpoint=False)
        else:
            self.Nx = int(getattr(grid, "Nx", cfg.Nx))
            self.Lx = float(getattr(grid, "Lx", cfg.Lx))
            self.dx = self.Lx / self.Nx
            self.x  = np.linspace(-0.5 * self.Lx, 0.5 * self.Lx, self.Nx, endpoint=False)

        self.k = 2.0 * np.pi * rfftfreq(self.Nx, d=self.dx)
        self.Nk = len(self.k)

        self.Xhat = np.zeros((self.Nk, 4), dtype=np.complex128)
        self._spec_cache = {}
        self._prepare_spectral_operators()

        # --- I/O writer (use orchestrated writer if provided)
        if writer is None:
            self.writer = StreamWriter(self.outdir)
            self._write_manifest()
        else:
            self.writer = writer  # manifest already written by orchestrator

        # init state
        self._init_state(init_state=getattr(self.args, "init_state", None))

    # ---------------- helpers ----------------

    def _prepare_spectral_operators(self):
        """Precompute (V, V^{-1}, λ) for each k-bin."""
        for i, kk in enumerate(self.k):
            # Build linear operator M(k) such that n X = M X
            M = L_of_k(self.cfg, float(kk))
            # eig returns w (λ) and V columns
            w, V = np.linalg.eig(M)
            # Order by real part (dominant first), for consistency
            order = np.argsort(w.real)[::-1]
            w = w[order]; V = V[:, order]
            # Inverse once
            Vinv = np.linalg.inv(V)
            self._spec_cache[i] = (V, Vinv, w)

    def _write_manifest(self):
        man = {
            "kind": "linear_native",
            "config": self.cfg.source_file,
            "Nx": self.Nx,
            "Lx": self.Lx,
            "dt": self.args.dt,
            "stop_time": self.args.stop_time,
            "save_stride": self.args.save_stride,
            "k_target": self.args.k_target,
            "amp": self.args.amp,
        }
        (self.outdir / "run.json").write_text(json.dumps(man, indent=2))

    def _nearest_k_index(self, k_target: float) -> int:
        return int(np.argmin(np.abs(self.k - float(k_target))))
    
    # exact physical-amplitude rescale
    def _current_physical_amp(self, metric: str = "max") -> float:
        """Compute physical amplitude of the density perturbation s(x)."""
        s_x = np.fft.irfft(self.Xhat[:, 0], n=self.Nx)
        if metric.lower() == "rms":
            return float(np.sqrt(np.mean(s_x**2)))
        return float(np.max(np.abs(s_x)))

    def _rescale_to_physical_amp(self, target: float, metric: str = "max") -> None:
        a = self._current_physical_amp(metric)
        if a == 0.0:
            return
        self.Xhat *= (target / a)

    def _init_state(self, init_state=None):
        rng = np.random.default_rng(self.args.seed)

        if init_state is not None:
            # Expect real-space arrays: Sigma, vx, vy, uy ; convert to spectral Xhat=[S_hat,vx_hat,vy_hat,uy_hat]
            Sigma = np.asarray(init_state["Sigma"])
            vx    = np.asarray(init_state["vx"])
            vy    = np.asarray(init_state["vy"])
            uy    = np.asarray(init_state["uy"])
            # S = Sigma - S0 (perturbation)
            S0 = float(getattr(self.cfg, "S0", getattr(self.cfg, "sig_0", 1.0)))
            s = Sigma - S0
            S_hat  = np.fft.rfft(s, n=self.Nx)
            vx_hat = np.fft.rfft(vx, n=self.Nx)
            vy_hat = np.fft.rfft(vy, n=self.Nx)
            uy_hat = np.fft.rfft(uy, n=self.Nx)
            self.Xhat[:, 0] = S_hat
            self.Xhat[:, 1] = vx_hat
            self.Xhat[:, 2] = vy_hat
            self.Xhat[:, 3] = uy_hat
            self.writer.write_metric({"t": 0.0, "note": "init_from_file"})
            return

        if self.args.k_target is not None:
            # Seed with the dominant eigenvector at nearest k-bin
            ik = self._nearest_k_index(self.args.k_target)
            k_used = self.k[ik]
            w, V = evp_solve_at_k(self.cfg, float(k_used))
            vdom = V[:, 0]  # [S, vx, vy, uy]
            # First seed with a provisional spectral scale (robustly rescaled below)
            s0 = vdom[0] if vdom[0] != 0 else 1.0 + 0j
            self.Xhat[:] = 0.0
            self.Xhat[ik, :] = (self.args.amp / s0) * vdom
            # If user asked for physical amplitude, rescale exactly
            if self.args.amp_is_physical:
                self._rescale_to_physical_amp(self.args.amp, metric=self.args.amp_metric)
            self.writer.write_metric({"t": 0.0, "note": "init_eigen", "k_index": int(ik), "k_used": float(k_used)})
        else:
            # Tiny random spectrum for S; zeros for velocities by default
            self.Xhat[:, 0] = (self.args.amp * 1e-1) * (rng.standard_normal(self.Nk) + 1j * rng.standard_normal(self.Nk))
            if self.args.amp_is_physical:
                self._rescale_to_physical_amp(self.args.amp, metric=self.args.amp_metric)
            self.writer.write_metric({"t": 0.0, "note": "init_noise"})

    # ---------------- stepping ----------------

    def step_exact(self, dt: float):
        """Exact per-k update using cached spectra."""
        for i in range(self.Nk):
            V, Vinv, lam = self._spec_cache[i]
            # y = V^{-1} X
            y = Vinv @ self.Xhat[i, :]
            # y <- exp(λ dt) * y
            y *= np.exp(lam * dt)
            # X <- V y
            self.Xhat[i, :] = V @ y

    # ---------------- diagnostics ----------------

    def _amp_at_k(self, i: int) -> float:
        # density amplitude |S_k|
        return float(np.abs(self.Xhat[i, 0]))

    def _write_checkpoint(self, istep: int, t: float):
        # Save the spectral state compactly
        fn = self.outdir / "checkpoints" / f"chk_{istep:06d}.npz"
        fn.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fn, t=t, k=self.k, Xhat=self.Xhat)

    # ---------------- main run ----------------

    def run(self):
        t = 0.0
        dt = float(self.args.dt)
        stop_time = float(self.args.stop_time)
        save_stride = int(self.args.save_stride)

        # diagnostics: pick a tracking bin (if requested)
        k_track_i = None
        if self.args.k_target is not None:
            k_track_i = self._nearest_k_index(self.args.k_target)

        istep = 0
        self._write_checkpoint(istep, t)

        while t < stop_time:
            istep += 1
            t_next = min(t + dt, stop_time)
            self.step_exact(t_next - t)
            t = t_next

            # metrics
            row = {"t": t, "dt": dt}
            if k_track_i is not None:
                row["k_track"] = float(self.k[k_track_i])
                row["amp_S_k"] = self._amp_at_k(k_track_i)
            self.writer.write_metric(row)

            if istep % save_stride == 0 or t >= stop_time:
                self._write_checkpoint(istep, t)

        return {"t_final": t, "steps": istep, "Nx": self.Nx, "Lx": self.Lx}
    

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
    seed_mode: str = "cos"     # "cos" or "eigen"        # 
    k_phys: float | None = None                          # 
    amp_is_physical: bool = True                         # 
    amp_metric: str = "max"                              #  ("max" or "rms")
    init_state: dict | None = None

class NonlinearNative:
    """
    Nonlinear pseudo-spectral IMEX integrator in x-space with gas uy always present.
    Explicit: advection, all variable-coefficient and nonlinear fluxes.
    Implicit (frozen): (4/3) nu_bar lap(vx), nu_bar lap(vy), nu_g lap(uy).
    """

    def __init__(self, cfg: Config, outdir: Path, args: NonlinearRunArgs,
                 grid: Grid1D | None = None,
                 writer: StreamWriter | None = None):
        self.cfg = cfg
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.args = args

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
        S0   = float(getattr(self.cfg, "S0", getattr(self.cfg, "sig_0", 1.0)))
        amp  = float(self.args.amp)
        k0   = int(self.args.k0)
        x = self.x

        if init_state is not None:
            # Use provided arrays directly
            return {
                "Sigma": np.asarray(init_state["Sigma"]),
                "vx":    np.asarray(init_state["vx"]),
                "vy":    np.asarray(init_state["vy"]),
                "uy":    np.asarray(init_state["uy"]),
            }

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
                if self.args.amp_metric == "rms":
                    a_now = float(np.sqrt(np.mean(S_raw**2)))
                else:
                    a_now = float(np.max(np.abs(S_raw)))
                scale = (amp / max(a_now, 1e-30)) if a_now != 0 else 0.0
            else:
                # interpret amp as fractional of S0 (physical amplitude = amp * S0)
                a_target = amp * S0
                if self.args.amp_metric == "rms":
                    a_now = float(np.sqrt(np.mean(S_raw**2)))
                else:
                    a_now = float(np.max(np.abs(S_raw)))
                scale = (a_target / max(a_now, 1e-30)) if a_now != 0 else 0.0

            Sigma = S0 + scale * S_raw
            vx    = scale * vx_raw
            vy    = scale * vy_raw
            uy    = scale * uy_raw

        else:
            # cosine seed (legacy)
            phase = 0.0
            Sigma = S0 * (1.0 + (amp if self.args.amp_is_physical else amp * S0) * 
                          np.cos(k0 * 2.0*np.pi * (x - x.min()) / self.Lx + phase) / max(S0, 1e-30))
            vx = np.zeros_like(Sigma)
            vy = np.zeros_like(Sigma)
            uy = np.zeros_like(Sigma)

        return {"Sigma": Sigma, "vx": vx, "vy": vy, "uy": uy}

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

        istep = 0
        self._write_checkpoint(istep, t)
        self.writer.write_metric(self._diag(t))

        while t < stop_time:
            istep += 1
            t_next = min(t + dt, stop_time)
            self.state = self._step_imex_rk2(self.state, t_next - t)
            t = t_next

            self.writer.write_metric(self._diag(t))
            if istep % save_stride == 0 or t >= stop_time:
                self._write_checkpoint(istep, t)

        return {"t_final": t, "steps": istep, "Nx": self.Nx, "Lx": self.Lx}