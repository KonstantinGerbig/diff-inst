# scripts/run_linear.py
import argparse
import json
import numpy as np
from pathlib import Path
from diffinst import Config
from diffinst.runtime import run_linear_native


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write manifest/metrics/checkpoint without evolving.")
    # linear mode
    ap.add_argument("--mode", choices=["linear"], default=None,
                    help="Select run mode. 'linear' uses the native spectral harness.")
    ap.add_argument("--stop_time", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--save-stride", type=int, default=50)
    ap.add_argument("--k", type=float, default=None,
                    help="If set, seed the dominant eigenmode at this k (rad/unit).")
    ap.add_argument("--amp", type=float, default=1e-3,
                    help="Initial |S_k| amplitude when seeding eigenmode.")
    ap.add_argument("--amp-physical", action="store_true",
                    help="Interpret --amp as physical amplitude of Σ-Σ0 at t=0.")
    ap.add_argument("--amp-metric", choices=["max","rms"], default="max",
                    help="Physical amplitude metric when --amp-physical is used.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                help="Path to .npz with arrays Sigma,vx,vy,uy to use as initial state.")

    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    init_state = None
    if args.init_from:
        Z = np.load(args.init_from)
        init_state = {k: Z[k] for k in ["Sigma","vx","vy","uy"]}

    # simple dry-run (existing behavior)
    if args.dry_run and args.mode is None:
        # write minimal files, as before
        (outdir / "checkpoints").mkdir(exist_ok=True)
        (outdir / "run.json").write_text(json.dumps({
            "kind": "dry",
            "config": cfg.source_file,
            "Lx": cfg.Lx,
            "backend": "native"
        }, indent=2))
        with (outdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"t": 0.0, "note": "dry_run"}) + "\n")
        np.savez_compressed(outdir / "checkpoints" / "chk_000000.npz",
                            t=0.0, x=np.linspace(-0.5*cfg.Lx, 0.5*cfg.Lx, int(cfg.Nx), endpoint=False))
        print("[dry] wrote one checkpoint; backend = native | Lx =", cfg.Lx)
        return

    # linear mode (native spectral)
    if args.mode == "linear":
        info = run_linear_native(
            cfg=cfg,
            outdir=outdir,
            stop_time=args.stop_time,
            dt=args.dt,
            save_stride=args.save_stride,
            k_target=args.k,
            amp=args.amp,
            seed=args.seed,
            amp_is_physical=bool(args.amp_physical),
            amp_metric=str(args.amp_metric),
            init_state=init_state,
        )
        print("[linear] done:", info)
        return

    raise SystemExit("No mode selected. Use --dry-run or --mode linear.")

if __name__ == "__main__":
    main()