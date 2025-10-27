from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import time
from .config import Config
from .grid import Grid1D
from .fields import State
from .io_utils import StreamWriter
from typing import Optional, Tuple
from .solvers.native import LinearNative, LinearRunArgs

def _build_grid_with_exact_fit(cfg: Config) -> Grid1D:
    grid = Grid1D(cfg.Nx, cfg.Lx)
    ef = cfg.exact_fit if isinstance(cfg.exact_fit, dict) else {}
    if ef.get("enable", False):
        k_target = float(ef["K_target"])
        harm = int(ef.get("harmonics", 2))
        Lx_new = grid.exact_fit_box(k_target, harm)
        grid = Grid1D(cfg.Nx, Lx_new)
    return grid

def prepare_run(cfg: Config, outdir: str | Path, kind: str, backend: str = "native") -> Tuple[Grid1D, State, StreamWriter]:
    """Common orchestration: grid, state, writer+manifest."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    grid = _build_grid_with_exact_fit(cfg)
    state = State.zeros(cfg.Nx, enable_gas=cfg.enable_gas)

    manifest = {
        "kind": kind,
        "config": cfg.source_file,
        "Nx": cfg.Nx,
        "Lx": grid.Lx,
        "backend": backend,
    }
    writer = StreamWriter(outdir, manifest)
    return grid, state, writer

class Runner:
    def __init__(self, cfg: Config, outdir: str | Path = "runs/out"):
        self.cfg = cfg
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def execute(self) -> dict:
        # shared orchestration
        grid, state, writer = prepare_run(self.cfg, self.outdir, kind="dry", backend=self.cfg.solver["backend"])

        arrays = {"sigma": state.sigma, "vx": state.vx, "vy": state.vy}
        if state.uy is not None:
            arrays["uy"] = state.uy

        writer.write_checkpoint(step=0, t=0.0, arrays=arrays)
        writer.append_metric({"step": 0, "t": 0.0, "mode": "dry", "backend": self.cfg.solver["backend"]})
        return {"Nx": self.cfg.Nx, "Lx": grid.Lx, "checkpoints": 1, "backend": self.cfg.solver["backend"]}    



def run_linear_native(cfg: Config,
                      outdir: str | Path,
                      tstop: float,
                      dt: float,
                      save_stride: int,
                      k_target: Optional[float] = None,
                      amp: float = 1e-3,
                      seed: Optional[int] = None,
                      amp_is_physical: bool = False,
                      amp_metric: str = "max",
                      ) -> dict:
    # shared setup
    grid, state_unused, writer = prepare_run(cfg, outdir, kind="linear_native", backend="native")

    args = LinearRunArgs(
        tstop=tstop,
        dt=dt,
        save_stride=save_stride,
        k_target=k_target,
        amp=amp,
        seed=seed,
        amp_is_physical=amp_is_physical,
        amp_metric=amp_metric,
    )
    solver = LinearNative(cfg, Path(outdir), args, grid=grid, writer=writer)
    return solver.run()