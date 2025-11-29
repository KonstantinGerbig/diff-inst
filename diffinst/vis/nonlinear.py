from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..analysis_api import (
    load_config_from_run,
    evp_gamma,
    max_sigma_series,
    load_nonlinear_run,
    load_nonlinear_Sigma_series,
)


def plot_sigma_max_vs_time(
    branches: Dict[str, Dict[str, Optional[Path | str]]],
    k_phys: float,
    figsize=(7, 4),
):
    """
    Plot max_x Sigma(t) for nonlinear eigenmode runs (native + Dedalus),
    and overplot a *single* EVP line using gamma(k_phys) from the first
    branch that has data.

    Parameters
    ----------
    branches : dict
        Mapping label -> dict with keys (any subset may be None):

            {
              "run_native":  Path or str or None,
              "run_dedalus": Path or str or None,
            }

    k_phys : float
        Physical wavenumber of the eigenmode.

    figsize : tuple
        Figure size.
    """
    labels = list(branches.keys())
    if not labels:
        raise ValueError("No branches provided to plot_sigma_max_vs_time.")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Optional: manual colors by label
    color_by_label: Dict[str, str] = {
        # "32": "#d55e00",
        # "64": "#0072b2",
        # ...
    }

    # store one representative time series for EVP overlay
    T_ref = None
    Sig0_ref = None
    dSig0_ref = None
    cfg_ref = None

    for i, label in enumerate(labels):
        runs = branches[label]
        col = color_by_label.get(label, f"C{i}")

        run_native = runs.get("run_native")
        run_ded = runs.get("run_dedalus")

        # ---------- native nonlinear ----------
        T_nat, Smax_nat = np.array([]), np.array([])
        if run_native is not None:
            T_nat, Smax_nat = max_sigma_series(run_native)
            if T_nat.size > 0:
                ax.plot(
                    T_nat,
                    Smax_nat,
                    linestyle="-",
                    color=col,
                    label=f"{label} (native)",
                )

        # ---------- Dedalus nonlinear ----------
        if run_ded is not None:
            T_ded, Smax_ded = max_sigma_series(run_ded)
            if T_ded.size > 0:
                ax.plot(
                    T_ded,
                    Smax_ded,
                    linestyle="--",
                    color=col,
                    label=f"{label} (Dedalus)",
                )

        # ---------- candidate for EVP reference ----------
        if (T_ref is None) and (T_nat.size > 0 or run_ded is not None):
            # prefer native if present, else Dedalus
            if T_nat.size > 0:
                T_ref = T_nat
                Smax_ref = Smax_nat
                cfg_run = run_native
            else:
                T_ref, Smax_ref = max_sigma_series(run_ded)
                cfg_run = run_ded

            try:
                cfg_ref = load_config_from_run(cfg_run)
                Sig0_ref = float(getattr(cfg_ref, "sig_0", getattr(cfg_ref, "S0", 1.0)))
                Sig_max0 = float(Smax_ref[0])
                dSig0_ref = Sig_max0 - Sig0_ref
            except Exception as e:
                print(f"[plot_sigma_max_vs_time] could not set EVP reference from {label}: {e}")
                T_ref = None
                cfg_ref = None

    # ---------- single EVP line ----------
    if (T_ref is not None) and (cfg_ref is not None) and (dSig0_ref is not None):
        try:
            gamma = evp_gamma(cfg_ref, k_phys)
            t0 = float(T_ref[0])
            Sig_lin = Sig0_ref + dSig0_ref * np.exp(gamma * (T_ref - t0))

            ax.plot(
                T_ref,
                Sig_lin,
                linestyle=":",
                linewidth=1.5,
                color="k",
                label=rf"EVP (γ={gamma:.3g})",
            )
        except Exception as e:
            print(f"[plot_sigma_max_vs_time] EVP overlay failed: {e}")

    ax.set(xlim =(T_ref[0], T_ref[-1]))
    ax.set_yscale("log")
    ax.set_xlabel(r"$t\Omega^{-1}$")
    ax.set_ylabel(r"$\max_x \Sigma(x,t)$")
    #ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(r"Nonlinear eigenmode growth and breakdown")

    fig.tight_layout()
    return fig, ax

def plot_sigma_snapshots(
    run_dir: Path | str,
    n_snap: int = 4,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    title: Optional[str] = None,
    figsize=(6, 3),
):
    """
    Plot Sigma(x) snapshots from a single nonlinear run, to visualize
    how the wave steepens and collapses.

    Parameters
    ----------
    run_dir : path-like
        Nonlinear run directory.

    n_snap : int
        Number of snapshots to show (evenly spaced in the selected time window).

    t_min, t_max : float or None
        Optional time window. If None, use full range available.

    title : str or None
        Optional panel title.

    figsize : tuple
        Figure size.
    """
    run_dir = Path(run_dir)
    Nx, Lx, files, _ = load_nonlinear_run(run_dir)
    if not files:
        raise ValueError(f"No checkpoints found in {run_dir}")

    # load all times + Sigma to choose snapshots
    T, Sig = load_nonlinear_Sigma_series(files)
    if T.size == 0:
        raise ValueError(f"No data in checkpoints for {run_dir}")

    # determine window
    if t_min is None:
        t_min = float(T[0])
    if t_max is None:
        t_max = float(T[-1])

    mask = (T >= t_min) & (T <= t_max)
    if not np.any(mask):
        raise ValueError("No snapshots in the specified time window.")

    T_sel = T[mask]
    Sig_sel = Sig[mask]

    # choose snapshot indices evenly spaced in this subset
    idx = np.linspace(0, len(T_sel) - 1, n_snap).astype(int)
    T_snap = T_sel[idx]
    Sig_snap = Sig_sel[idx]

    # x-grid from first checkpoint
    with np.load(files[0]) as Z0:
        x = np.asarray(Z0["x"])

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    cmap = plt.get_cmap("viridis")
    for j, (tt, Sx) in enumerate(zip(T_snap, Sig_snap)):
        frac = 0.1 + 0.8 * (j / max(1, n_snap - 1))
        col = cmap(frac)
        ax.plot(x, Sx, color=col, label=fr"$t = {tt:.1f}\,\Omega^{{-1}}$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\Sigma(x,t)$")
    #ax.grid(True, alpha=0.3)
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig, ax