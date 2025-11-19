# scripts/make_ic_eigen.py

import argparse
from pathlib import Path

import numpy as np

from diffinst import Config
from diffinst.linear_ops import evp_solve_at_k
from diffinst.analysis_api import save_ic_npz
from diffinst.grid import Grid1D


def build_ic(
    cfg: Config,
    outpath: Path,
    k_phys: float,
    amp_phys: float,
    Nx_override: int | None = None,
    phase: float = 0.0,
    exact_fit_harm : int | None = None,
) -> Path:
    """
    Build a single eigenmode IC and write it as a compressed .npz file.

    - cfg:        physics/background parameters
    - outpath:    where to write the IC (npz)
    - k_phys:     physical wavenumber (rad/unit)
    - amp_phys:   physical amplitude of |Sigma - S0| (max norm)
    - Nx_override: if not None, use this Nx for the grid instead of cfg.Nx
    - phase:      optional phase offset in the eigenmode exp(i k x + phase)
    """
    # Effective resolution / box
    Nx = int(Nx_override) if Nx_override is not None else int(cfg.Nx)
    base_Lx = float(cfg.Lx)
    grid = Grid1D(Nx=Nx, Lx=base_Lx)

    if exact_fit_harm is not None:
        Lx = grid.exact_fit_Lx(k_phys, exact_fit_harm)
    else:
        Lx = grid.Lx

    # Background Sigma_0
    S0 = float(getattr(cfg, "S0", getattr(cfg, "sig_0", 1.0)))

    # Physical grid
    x = np.linspace(-0.5 * Lx, 0.5 * Lx, Nx, endpoint=False)

    # Solve EVP at k_phys
    w, V = evp_solve_at_k(cfg, k_phys)
    v = V[:, 0]  # dominant eigenvector [Sigma, vx, vy, uy]

    eikx = np.exp(1j * (k_phys * (x - x.min()) + phase))

    S_raw  = (v[0] * eikx).real
    vx_raw = (v[1] * eikx).real
    vy_raw = (v[2] * eikx).real
    uy_raw = (v[3] * eikx).real

    # Scale so that max |Sigma - S0| = amp_phys
    a_now = float(np.max(np.abs(S_raw))) if np.any(S_raw != 0.0) else 0.0
    if a_now == 0.0:
        scale = 0.0
    else:
        scale = amp_phys / a_now

    Sigma0 = S0 + scale * S_raw
    vx0    = scale * vx_raw
    vy0    = scale * vy_raw
    uy0    = scale * uy_raw

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "k_phys": float(k_phys),
        "amp_phys": float(amp_phys),
        "Nx": int(Nx),
        "Lx": float(Lx),
        "S0": float(S0),
        "phase": float(phase),
        "config": str(cfg.source_file),
    }

    save_ic_npz(outpath, Sigma0, vx0, vy0, uy0, meta=meta)
    return outpath


def main():
    ap = argparse.ArgumentParser(
        description="Generate an eigenmode initial condition and save to .npz."
    )
    ap.add_argument("--config", required=True,
                    help="Path to experiment YAML (physics / base parameters).")
    ap.add_argument("--defaults", default=None,
                    help="Optional defaults YAML (if your Config.from_yaml uses it).")
    ap.add_argument("--out", required=True,
                    help="Output .npz file for the IC.")
    ap.add_argument("--k", type=float, required=True,
                    help="Physical wavenumber k (rad/unit).")
    ap.add_argument("--amp", type=float, default=1e-6,
                    help="Physical amplitude of |Sigma - S0| (max norm).")
    ap.add_argument("--Nx", type=int, default=None,
                    help="Optional grid size override for IC (defaults to cfg.Nx).")
    ap.add_argument("--phase", type=float, default=0.0,
                    help="Optional phase offset in the eigenmode.")
    ap.add_argument("--exact-fit-harm", type=int, default=None,
                    help="Optional integer m to make Lx fit exactly m eigenmode")

    args = ap.parse_args()

    config_path = Path(args.config)
    defaults_path = Path(args.defaults) if args.defaults is not None else None

    # Load Config with or without defaults, depending on your setup
    if defaults_path is not None:
        cfg = Config.from_yaml(config_path, defaults_path)
    else:
        cfg = Config.from_yaml(config_path)

    outpath = build_ic(
        cfg=cfg,
        outpath=Path(args.out),
        k_phys=float(args.k),
        amp_phys=float(args.amp),
        Nx_override=args.Nx,
        phase=float(args.phase),
        exact_fit_harm=args.exact_fit_harm
    )
    print(f"[make_ic_eigen] IC written to {outpath} "
          f"(k={args.k}, amp={args.amp}, Nx={args.Nx or cfg.Nx})")


if __name__ == "__main__":
    main()