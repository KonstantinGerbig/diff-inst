import argparse
import json
from pathlib import Path
import numpy as np
from diffinst import Config
from diffinst.linear_ops import evp_solve_at_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kmin", type=float, default=1.0)
    ap.add_argument("--kmax", type=float, default=1e3)
    ap.add_argument("--nk", type=int, default=128)
    ap.add_argument("--out", default="runs/evp")
    ap.add_argument("--plot", action="store_true",
                help="Save a preview plot (PNG) to the output directory.")
    ap.add_argument("--plot-name", default="preview_growth.png",
                help="Filename for the saved plot (inside --out).")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    ks = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk)
    growth = np.empty(args.nk); freq = np.empty(args.nk)
    dom_eig = np.empty(args.nk, dtype=np.complex128)

    # dominant eigenvector (optional; store first 4 comps)
    Vdom = np.empty((args.nk, 4), dtype=np.complex128)

    for i, k in enumerate(ks):
        w, V = evp_solve_at_k(cfg, k)
        dom = w[0]
        dom_eig[i] = dom
        growth[i] = dom.real
        freq[i]   = dom.imag
        Vdom[i,:] = V[:,0]

    # save a small table
    np.savez_compressed(outdir/"growth_table.npz",
                        k=ks, growth=growth, freq=freq,
                        dom=dom_eig, Vdom=Vdom)

    # write a tiny manifest
    with open(outdir/"run.json", "w") as f:
        json.dump({"config": cfg.source_file,
                   "kmin": args.kmin, "kmax": args.kmax, "nk": args.nk}, f, indent=2)

    # console summary
    imax = int(np.argmax(growth))
    print(f"[evp] max growth at k={ks[imax]:.4g}: gamma={growth[imax]:.4g}, omega={freq[imax]:.4g}")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,4))
        ax.loglog(ks, np.maximum(growth, 1e-16), label="growth rate")
        ax.loglog(ks, np.abs(freq), "--", label="|oscillation freq|")
        plt.axvline(ks[imax], ls="--", lw=1)
        ax.set_xlabel("wavenumber k")
        ax.set_ylabel("rate")
        ax.set_title("EVP dispersion")
        ax.legend()
        fig.tight_layout()
        out_png = outdir / args.plot_name
        fig.savefig(out_png, dpi=160)
        print(f"[evp] saved plot -> {out_png}")

if __name__ == "__main__":
    main()