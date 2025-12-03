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

from matplotlib.gridspec import GridSpec

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

def plot_sigma_max_and_snapshots(
    branches: Dict[str, Dict[str, Optional[Path | str]]],
    k_phys: float,
    snapshot_label: str,
    n_snap: int = 4,
    figsize=(9, 4),
):
    """
    Three-panel summary figure:

      * Left: max_x Sigma(x,t) vs t for all branches (native + Dedalus),
              plus a single EVP prediction.
      * Right-top: Sigma(x,t) snapshots for the selected branch (native),
                   from t=0 to the last time where BOTH native and Dedalus
                   still have finite fields.
      * Right-bottom: same, but for Dedalus.

    Parameters
    ----------
    branches : dict
        Mapping label -> dict with keys (any subset may be None):
            {
              "run_native":  Path or str or None,
              "run_dedalus": Path or str or None,
            }

    k_phys : float
        Physical wavenumber of the eigenmode (for EVP).

    snapshot_label : str
        Key in `branches` that selects which branch to use for the
        right-hand snapshot panels (e.g. "128").

    n_snap : int
        Number of snapshots to show for each solver.

    figsize : tuple
        Overall figure size.
    """
    labels = list(branches.keys())
    if not labels:
        raise ValueError("No branches provided.")

    if snapshot_label not in branches:
        raise ValueError(f"snapshot_label={snapshot_label!r} not found in branches.")

    # ------------------ figure & layout ------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 2,
        width_ratios=[1.1, 1.4],
        height_ratios=[1.0, 1.0],
        wspace=0.25,
        hspace=0.20,
        figure=fig,
    )

    ax_left = fig.add_subplot(gs[:, 0])      # spans both rows
    ax_top  = fig.add_subplot(gs[0, 1])      # native snapshots
    ax_bot  = fig.add_subplot(gs[1, 1])      # Dedalus snapshots

    from pypalettes import load_cmap
    cmap = load_cmap("Callanthias_australis") 
    palette = cmap(np.linspace(0, 1, 4))


    # Optional: manual colors by label (you can customize)
    color_by_label: Dict[str, str] = {
         "32": palette[0],
         "64": palette[1],
         "128": palette[2],
         "254": palette[3],
    }

    # ---------- LEFT PANEL: max_x Sigma vs time + EVP ----------
    T_ref = None
    Sig0_ref = None
    dSig0_ref = None
    cfg_ref = None

    for i, label in enumerate(labels):
        runs = branches[label]
        col = color_by_label.get(label, f"C{i}")

        run_native = runs.get("run_native")
        run_ded = runs.get("run_dedalus")

        # native
        T_nat, Smax_nat = np.array([]), np.array([])
        if run_native is not None:
            T_nat, Smax_nat = max_sigma_series(run_native)
            if T_nat.size > 0:
                ax_left.plot(
                    T_nat, Smax_nat,
                    linestyle="-", color=col,
                    label=f"{label} (native)",
                )

        # Dedalus
        if run_ded is not None:
            T_ded, Smax_ded = max_sigma_series(run_ded)
            if T_ded.size > 0:
                ax_left.plot(
                    T_ded, Smax_ded,
                    linestyle="--", color=col,
                    label=f"{label} (Dedalus)",
                )

        # EVP reference: use the first branch that has data
        if (T_ref is None) and (T_nat.size > 0 or run_ded is not None):
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
                print(f"[plot_sigma_max_and_snapshots] EVP ref failed for {label}: {e}")
                T_ref = None
                cfg_ref = None

    # Single EVP line
    if (T_ref is not None) and (cfg_ref is not None) and (dSig0_ref is not None):
        try:
            gamma = evp_gamma(cfg_ref, k_phys)
            t0 = float(T_ref[0])
            Sig_lin = Sig0_ref + dSig0_ref * np.exp(gamma * (T_ref - t0))

            ax_left.plot(
                T_ref, Sig_lin,
                linestyle=":", linewidth=1.5, color="k",
                label=rf"EVP ($\gamma$={gamma:.3g})",
            )
        except Exception as e:
            print(f"[plot_sigma_max_and_snapshots] EVP overlay failed: {e}")

    if T_ref is not None:
        ax_left.set_xlim(T_ref[0], T_ref[-1])
    #ax_left.set_yscale("log")
    ax_left.set_xlabel(r"$t\Omega^{-1}$")
    ax_left.set_ylabel(r"$\max_x \Sigma(x,t)$")
    ax_left.legend(frameon=False, fontsize=10, ncols = 2, loc = "upper left")
   # ax_left.set_title(r"Nonlinear eigenmode growth and breakdown")

    # ---------- RIGHT PANELS: snapshots for snapshot_label ----------
    runs_snap = branches[snapshot_label]
    run_nat = runs_snap.get("run_native")
    run_ded = runs_snap.get("run_dedalus")

    if run_nat is None or run_ded is None:
        raise ValueError(f"Need both native and Dedalus runs for {snapshot_label!r}.")

    # Load native
    Nx_nat, Lx_nat, files_nat, _ = load_nonlinear_run(run_nat)
    T_nat, Sig_nat = load_nonlinear_Sigma_series(files_nat)
    with np.load(files_nat[0]) as Z0:
        x_nat = np.asarray(Z0["x"])

    # Load Dedalus
    Nx_ded, Lx_ded, files_ded, _ = load_nonlinear_run(run_ded)
    T_ded, Sig_ded = load_nonlinear_Sigma_series(files_ded)
    with np.load(files_ded[0]) as Z1:
        x_ded = np.asarray(Z1["x"])

    # Mask out NaN/Inf and determine last common "good" time
    good_nat = np.all(np.isfinite(Sig_nat), axis=1)
    good_ded = np.all(np.isfinite(Sig_ded), axis=1)

    if not np.any(good_nat) or not np.any(good_ded):
        raise ValueError("No finite snapshots to compare for the selected branch.")

    T_nat_good = T_nat[good_nat]
    T_ded_good = T_ded[good_ded]

    t0_common = max(T_nat_good[0], T_ded_good[0])
    #t_last_common = min(T_nat_good[-1], T_ded_good[-1])
    # second to last good
    t_last_common = min(T_nat_good[-4], T_ded_good[-4])

    # Restrict to common window
    mask_nat_window = (T_nat >= t0_common) & (T_nat <= t_last_common) & good_nat
    mask_ded_window = (T_ded >= t0_common) & (T_ded <= t_last_common) & good_ded

    T_nat_w = T_nat[mask_nat_window]
    Sig_nat_w = Sig_nat[mask_nat_window]
    T_ded_w = T_ded[mask_ded_window]
    Sig_ded_w = Sig_ded[mask_ded_window]

    if T_nat_w.size == 0 or T_ded_w.size == 0:
        raise ValueError("No overlapping, finite-time window between native and Dedalus.")

    # Pick snapshot indices evenly in this window
    idx_nat = np.linspace(0, len(T_nat_w) - 1, n_snap).astype(int)
    idx_ded = np.linspace(0, len(T_ded_w) - 1, n_snap).astype(int)

    T_snap_nat = T_nat_w[idx_nat]
    Sig_snap_nat = Sig_nat_w[idx_nat]
    T_snap_ded = T_ded_w[idx_ded]
    Sig_snap_ded = Sig_ded_w[idx_ded]

    # Native snapshots (top right)
    #cmap = plt.get_cmap("viridis")
    cmap = load_cmap("Hiroshige") 
    for j, (tt, Sx) in enumerate(zip(T_snap_nat, Sig_snap_nat)):
        frac = 0.1 + 0.8 * (j / max(1, n_snap - 1))
        col = cmap(frac)
        ax_top.plot(x_nat, Sx, color=col, label=fr"$t = {tt:.1f}\,\Omega^{{-1}}$")

    ax_top.set_ylabel(r"$\Sigma(x,t)$")
    #ax_top.set_title(rf"Native, $N_x={Nx_nat}$")
    ax_top.tick_params(labelbottom=False)  # x-label only on bottom panel
    ax_top.legend(frameon=True, fontsize=8, loc="upper right", ncols = 2)
    ax_top.set(xlim = (x_nat[0], x_nat[-1]))
    ax_top.text(-0.05, 2.6, r"Native", fontsize = 12)

    # Dedalus snapshots (bottom right)
    for j, (tt, Sx) in enumerate(zip(T_snap_ded, Sig_snap_ded)):
        frac = 0.1 + 0.8 * (j / max(1, n_snap - 1))
        col = cmap(frac)
        ax_bot.plot(x_ded, Sx, color=col, label=fr"$t = {tt:.1f}\,\Omega^{{-1}}$")

    ax_bot.set_xlabel(r"$x$")
    ax_bot.set_ylabel(r"$\Sigma(x,t)$")
    ax_bot.set(xlim = (x_ded[0], x_ded[-1]))
    #ax_bot.set_title(rf"Dedalus, $N_x={Nx_ded}$")
    #ax_bot.legend(frameon=False, fontsize=7, loc="upper center")
    ax_bot.text(-0.05, 2.7, r"Dedalus", fontsize = 12)


    #fig.tight_layout()
    return fig, (ax_left, ax_top, ax_bot)