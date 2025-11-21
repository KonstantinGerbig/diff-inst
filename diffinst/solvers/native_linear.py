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

        # gas toggle & variable count
        self.use_gas = bool(getattr(cfg, "enable_gas", True))
        self.nvar = 4 if self.use_gas else 3  # [S,vx,vy,(uy)]

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

        self.Xhat = np.zeros((self.Nk, self.nvar), dtype=np.complex128)
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
            # Expect real-space arrays: Sigma, vx, vy, (uy optional).
            Sigma = np.asarray(init_state["Sigma"])
            vx    = np.asarray(init_state["vx"])
            vy    = np.asarray(init_state["vy"])
            # uy may or may not exist; for dust-only we ignore it anyway
            uy = np.asarray(init_state.get("uy", np.zeros_like(Sigma)))

            S0 = float(getattr(self.cfg, "S0", getattr(self.cfg, "sig_0", 1.0)))
            s = Sigma - S0

            S_hat  = np.fft.rfft(s,  n=self.Nx)
            vx_hat = np.fft.rfft(vx, n=self.Nx)
            vy_hat = np.fft.rfft(vy, n=self.Nx)
            uy_hat = np.fft.rfft(uy, n=self.Nx)

            if self.nvar == 3:
                # dust-only: [S, vx, vy]
                self.Xhat[:, 0] = S_hat
                self.Xhat[:, 1] = vx_hat
                self.Xhat[:, 2] = vy_hat
            else:
                # dust+gas: [S, vx, vy, uy]
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