# scripts/run_nonlinear.py

import argparse
import json
from pathlib import Path
import numpy as np
import logging
import shutil

from diffinst import Config
from diffinst.runtime import run_nonlinear   # <-- import the dispatcher (not _native)
from dataclasses import replace


def _write_dry(outdir: Path, cfg: Config, backend: str) -> None:
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "run.json").write_text(json.dumps({
        "kind": "dry",
        "config": cfg.source_file,
        "Lx": cfg.Lx,
        "backend": backend,
        "mode": "nonlinear",
    }, indent=2))
    x = np.linspace(-0.5 * cfg.Lx, 0.5 * cfg.Lx, int(cfg.Nx), endpoint=False)
    np.savez_compressed(outdir / "checkpoints" / "chk_000000.npz",
                        t=0.0, x=x, Sigma=np.full_like(x, getattr(cfg, "S0", 1.0)),
                        vx=np.zeros_like(x), vy=np.zeros_like(x))
    with (outdir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({"t": 0.0, "note": "dry_run"}) + "\n")
    print(f"[dry] wrote one checkpoint; backend = {backend} | mode = nonlinear | Lx = {cfg.Lx}")

def main():

    # Configure logging at CLI entry
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s :: %(message)s",
    )


    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--backend", choices=["native","dedalus"], default=None,
                    help="Override YAML solver.backend for this run.")

    ap.add_argument("--mode", choices=["nonlinear"], default=None)
    ap.add_argument("--stop_time", type=float, default=10.0,
                    help="Total integration time (code units).")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--save-stride", type=int, default=100)

    ap.add_argument("--print-stride", type=int, default=None)

    # Seeding controls
    ap.add_argument(
        "--seed-mode",
        choices=["cos", "eigen", "noise"],
        default="cos",
        help=(
            "cos: Sigma = S0 + A cos(m x); "
            "eigen: seed EVP eigenmode at --k; "
            "noise: Sigma = S0 + A * N(0,1) with vx=vy=uy=0."
        ),
    )
    ap.add_argument("--init-k", type=int, default=1,
                    help="Harmonic index m for cosine seeding (used when --seed-mode=cos).")
    ap.add_argument("--k", type=float, default=None,
                    help="Physical wavenumber (rad/unit) for eigenmode seeding.")
    ap.add_argument("--amp", type=float, default=1e-3,
                    help="Amplitude; physical if --amp-physical else fractional of S0.")
    ap.add_argument("--amp-physical", action="store_true")
    ap.add_argument("--amp-metric", choices=["max","rms"], default="max")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                help="Path to .npz with arrays Sigma,vx,vy,uy to use as initial state.")

    ap.add_argument("--orbits", type=float, default=None,
                    help="If set, stop_time = orbits * 2π / Omega.")
    
    ap.add_argument("--Nx", type=int, default=None)

    ap.add_argument("--force", action="store_true",
                    help="Delete the output directory if it already exists.")

    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)

    outdir = Path(args.outdir)
    # force delete existing directory
    if outdir.exists() and args.force:
        print(f"[INFO] Removing existing output directory: {outdir}")
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)



    if args.Nx is not None:
        cfg = replace(cfg, Nx=int(args.Nx))

    init_state = None
    if args.init_from:
        Z = np.load(args.init_from)
        init_state = {k: Z[k] for k in ["Sigma","vx","vy","uy"]}

        # sanity check: IC resolution must match cfg.Nx
        n_ic = init_state["Sigma"].shape[0]
        if n_ic != int(cfg.Nx):
            raise SystemExit(
                f"init-from file has Nx={n_ic}, but cfg.Nx={cfg.Nx} (after any --Nx override). "
                f"Either regenerate the IC at resolution {cfg.Nx} or change --Nx to {n_ic}."
            )

    # Compute stop_time from orbits if requested
    stop_time = args.stop_time
    if args.orbits is not None:
        Omega = float(getattr(cfg, "Omega", 1.0))
        stop_time = float(args.orbits) * 2.0 * np.pi / max(Omega, 1e-12)

    # Backend preference: CLI flag wins; else YAML; fallback native
    backend = args.backend or (getattr(cfg, "solver", {}) or {}).get("backend", "native")

    if args.dry_run and args.mode is None:
        _write_dry(outdir, cfg, backend); return
    

    if args.mode == "nonlinear":
        info = run_nonlinear(
            cfg=cfg,
            outdir=outdir,
            stop_time=stop_time,
            dt=args.dt,
            save_stride=args.save_stride,
            init_k=int(args.init_k),
            amp=float(args.amp),
            seed=args.seed,
            seed_mode=str(args.seed_mode),
            k_phys=float(args.k) if args.k is not None else None,
            amp_is_physical=bool(args.amp_physical),
            amp_metric=str(args.amp_metric),
            init_state=init_state,
            backend=backend,  # <-- pass to dispatcher,
            print_stride=args.print_stride
        )
        print("[nonlinear] done:", info)
        return

    raise SystemExit("No mode selected. Use --dry-run or --mode nonlinear.")

if __name__ == "__main__":
    main()