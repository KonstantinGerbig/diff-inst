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
from diffinst.linear_ops import evp_solve_at_k

from matplotlib.gridspec import GridSpec

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .. import Config

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
    ax.set_xlabel(r"$t[\Omega^{-1}]$")
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
        ax.plot(x, Sx, color=col, label=fr"$t = {tt:.1f}\,[\Omega^{{-1}}]$")

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
    cmap = load_cmap("CarolMan") 
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
    #ax_left.set(xlim =(0,15), ylim = (1,5))
    ax_left.set_xlabel(r"$t[\Omega^{-1}]$")
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
        ax_top.plot(x_nat, Sx, color=col, label=fr"$t = {tt:.1f}\,[\Omega^{{-1}}]$")

    ax_top.set_ylabel(r"$\Sigma(x,t)$")
    #ax_top.set_title(rf"Native, $N_x={Nx_nat}$")
    ax_top.tick_params(labelbottom=False)  # x-label only on bottom panel
    ax_top.legend(frameon=True, fontsize=8, loc="upper right", ncols = 2)
    ax_top.set(xlim = (x_nat[0], x_nat[-1]))
    ax_top.text(-0.062, 2.2, r"Native", fontsize = 12)

    # Dedalus snapshots (bottom right)
    for j, (tt, Sx) in enumerate(zip(T_snap_ded, Sig_snap_ded)):
        frac = 0.1 + 0.8 * (j / max(1, n_snap - 1))
        col = cmap(frac)
        ax_bot.plot(x_ded, Sx, color=col, label=fr"$t = {tt:.1f}\,[\Omega^{{-1}}]$")

    ax_bot.set_xlabel(r"$x$")
    ax_bot.set_ylabel(r"$\Sigma(x,t)$")
    ax_bot.set(xlim = (x_ded[0], x_ded[-1]))
    #ax_bot.set_title(rf"Dedalus, $N_x={Nx_ded}$")
    #ax_bot.legend(frameon=False, fontsize=7, loc="upper center")
    ax_bot.text(-0.062, 2.2, r"Dedalus", fontsize = 12)


    fig.tight_layout()
    return fig, (ax_left, ax_top, ax_bot)



def plot_noise_two_res_summary(
    run_low: Path | str,
    run_mid: Path | str,
    run_high: Path | str,
    label_low: str = r"$N_x^\mathrm{(low)}$",
    label_mid: str = r"$N_x^\mathrm{(mid)}$",
    label_high: str = r"$N_x^\mathrm{(high)}$",
    k_ref: float | None = None,
    kmin: float = 10.0,
    kmax: float = 5e3,
    n_snap: int = 4,
    figsize=(10, 3.5),
):
    """
    Summary figure for a noise run at two different resolutions (both native):

      - Left panel: max_x Sigma(x,t) vs t for low and high resolution,
                    plus an optional EVP envelope using gamma(k_ref).
      - Middle panel: Sigma(x,t) snapshots (low resolution).
      - Right panel: Sigma(x,t) snapshots (high resolution).

    Parameters
    ----------
    run_low, run_high : path-like
        Nonlinear noise run directories (native backend) for low/high Nx.

    label_low, label_high : str
        Legend labels for the two resolutions.

    k_ref : float or None
        If provided, compute gamma(k_ref) from EVP and overlay linear-growth
        envelope for max_x Sigma using the low-res time series.

    n_snap : int
        Number of snapshots to show in each snapshot panel.

    figsize : tuple
        Overall figure size.
    """
    run_low = Path(run_low)
    run_mid = Path(run_mid)
    run_high = Path(run_high)

    # ---------- load low-res ----------
    Nx_lo, Lx_lo, files_lo, man_lo = load_nonlinear_run(run_low)
    T_lo, Sig_lo = load_nonlinear_Sigma_series(files_lo)
    if T_lo.size == 0:
        raise ValueError(f"No data in low-res run {run_low}")

    with np.load(files_lo[0]) as Z0:
        x_lo = np.asarray(Z0["x"])

    Smax_lo = np.nanmax(Sig_lo, axis=1)

    # ---------- load mid-res ----------
    Nx_mid, Lx_mid, files_mid, man_mid = load_nonlinear_run(run_mid)
    T_mid, Sig_mid = load_nonlinear_Sigma_series(files_mid)
    if T_mid.size == 0:
        raise ValueError(f"No data in mid-res run {run_mid}")

    with np.load(files_mid[0]) as Z1:
        x_mid = np.asarray(Z1["x"])

    Smax_mid = np.nanmax(Sig_mid, axis=1)

    # ---------- load high-res ----------
    Nx_hi, Lx_hi, files_hi, man_hi = load_nonlinear_run(run_high)
    T_hi, Sig_hi = load_nonlinear_Sigma_series(files_hi)
    if T_hi.size == 0:
        raise ValueError(f"No data in high-res run {run_high}")

    with np.load(files_hi[0]) as Z2:
        x_hi = np.asarray(Z2["x"])

    Smax_hi = np.nanmax(Sig_hi, axis=1)

    # ---------- figure layout ----------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1, 3,
        width_ratios=[1.1, 1.0, 1.0],
        wspace=0.3,
        figure=fig,
    )

    ax_left  = fig.add_subplot(gs[0, 0])
    ax_mid   = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    from pypalettes import load_cmap
    cmap = load_cmap("CarolMan") 
    palette = cmap(np.linspace(0, 1, 5))
    color1 = palette[0]
    color2 = palette[1]
    color3 = palette[2]

    # ---------- left: max Σ vs t (+ EVP envelope) ----------
    ax_left.plot(T_lo, Smax_lo, color=color1, lw=2, label=label_low)
    ax_left.plot(T_mid, Smax_mid, color=color2, lw=2, label=label_mid)
    ax_left.plot(T_hi, Smax_hi, color=color3, lw=2, label=label_high)

    cfg = load_config_from_run(run_low)
    Sig0 = float(getattr(cfg, "sig_0", getattr(cfg, "S0", 1.0)))
    dSig0 = float(Smax_lo[0] - Sig0)
    
    # EVP envelope using low-res config/time series
    if k_ref is not None:
        try:
            w, _ = evp_solve_at_k(cfg, float(k_ref))
            gamma = float(w[0].real)
            t0 = float(T_lo[0])
            Sig_lin = Sig0 + dSig0 * np.exp(gamma * (T_lo - t0))

            ax_left.plot(
                T_lo, Sig_lin,
                color="k", ls=":", lw=1.5,
                label=rf"EVP, $\gamma(k_\mathrm{{ref}})={gamma:.3g}$",
            )
        except Exception as e:
            print(f"[plot_noise_two_res_summary] EVP overlay failed: {e}")

    else: 
        # EVP sweep fastest growing mode

        nk = 200
        ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        growth = np.empty_like(ks)
        for i, k in enumerate(ks):
            w, _ = evp_solve_at_k(cfg, float(k))
            growth[i] = w[0].real

        imax = int(np.argmax(growth))
        k_max = float(ks[imax])
        gamma_max = float(growth[imax])
        t0 = float(T_lo[0])
        Sig_lin = Sig0 + dSig0 * np.exp(gamma_max * (T_lo - t0))

        #ax_left.plot(
        #        T_lo, Sig_lin,
        #        color="k", ls=":", lw=1.5,
        #        label=rf"EVP, $\gamma_\mathrm{{max}}={gamma_max:.3g}$",
        #    )

    ax_left.set_yscale("log")
    ax_left.set_xlabel(r"$t[\Omega^{-1}]$")
    ax_left.set_ylabel(r"$\max_x \Sigma(x,t)$")
    #ax_left.set_title("Noise-driven growth (two resolutions)")
    ax_left.legend(frameon=False, fontsize=9)
    ax_left.set(xlim = (T_lo[0], T_lo[-1]))

    # ---------- choose snapshot times (finite) ----------
    def _choose_snapshots(T, Sig, n):
        finite = np.all(np.isfinite(Sig), axis=1)
        T_f = T[finite]
        Sig_f = Sig[finite]
        if T_f.size == 0:
            return np.array([]), np.empty((0,) + Sig.shape[1:])
        idx = np.linspace(0, len(T_f) - 1, n).astype(int)
        return T_f[idx], Sig_f[idx]

    T_snap_lo, Sig_snap_lo = _choose_snapshots(T_lo, Sig_lo, n_snap)
    T_snap_hi, Sig_snap_hi = _choose_snapshots(T_hi, Sig_hi, n_snap)

    #cmap = plt.get_cmap("viridis")
    cmap = load_cmap("Hiroshige") 

    # ---------- middle: low-res snapshots ----------
    for j, (tt, Sx) in enumerate(zip(T_snap_lo, Sig_snap_lo)):
        frac = 0.1 + 0.8 * (j / max(1, n_snap - 1))
        col = cmap(frac)
        ax_mid.plot(x_lo, Sx, color=col, lw=1.5,
                    label=fr"$t={tt:.1f}\,[\Omega^{{-1}}]$")

    ax_mid.set_xlabel(r"$x$")
    ax_mid.set_ylabel(r"$\Sigma(x,t)$")
    ax_mid.set_title(rf"{label_low}")
    ax_mid.legend(frameon=True, fontsize=8, ncols = 2)
    ax_mid.set(xlim = (x_lo[0], x_lo[-1]))

    # ---------- right: high-res snapshots ----------
    for j, (tt, Sx) in enumerate(zip(T_snap_hi, Sig_snap_hi)):
        frac = 0.1 + 0.8 * (j / max(1, n_snap - 1))
        col = cmap(frac)
        ax_right.plot(x_hi, Sx, color=col, lw=1.5,
                      label=fr"$t={tt:.1f}\,[\Omega^{{-1}}]$")

    ax_right.set_xlabel(r"$x$")
    ax_right.set_ylabel(r"$\Sigma(x,t)$")
    ax_right.set_title(rf"{label_high}")
    #ax_right.legend(frameon=False, fontsize=7)
    ax_right.set(xlim = (x_hi[0], x_hi[-1]))

    fig.tight_layout()
    return fig, (ax_left, ax_mid, ax_right)







def plot_noise_dominant_mode_vs_theory(
    run_low: Path | str,
    run_mid: Path | str,
    run_high: Path | str,
    cfg_path: Path | str,
    label_low: str = r"$N_x^\mathrm{(low)}$",
    label_mid: str = r"$N_x^\mathrm{(mid)}$",
    label_high: str = r"$N_x^\mathrm{(high)}$",
    kmin: float = 10.0,
    kmax: float = 5e3,
    nk: int = 120,
    amp_floor: float = 1e-6,
    figsize=(6.0, 4.0),
):
    """
    Main panel: k_dom(t) for low/high-res noise runs + horizontal line at k_max.
    Inset: EVP growth curve γ(k) with k_max highlighted.

    Parameters
    ----------
    run_low, run_high : path-like
        Nonlinear noise run directories (native) for low/high Nx.

    cfg_path : path-like
        YAML config for the EVP sweep (same physics as the runs).

    kmin, kmax : float
        Min/max k for the EVP sweep (log-spaced).

    nk : int
        Number of k points in the EVP sweep.

    amp_floor : float
        If the maximum Fourier amplitude at a time step is below amp_floor,
        that time is treated as 'no dominant mode' (value set to NaN).

    figsize : tuple
        Figure size.
    """
    run_low = Path(run_low)
    run_mid = Path(run_mid)
    run_high = Path(run_high)
    cfg_path = Path(cfg_path)
    cfg = Config.from_yaml(cfg_path)

    # ---------- EVP sweep ----------
    ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    growth = np.empty_like(ks)
    for i, k in enumerate(ks):
        w, _ = evp_solve_at_k(cfg, float(k))
        growth[i] = w[0].real

    imax = int(np.argmax(growth))
    k_max = float(ks[imax])
    gamma_max = float(growth[imax])

    # ---------- helper: dominant mode from a run ----------
    def _dominant_k_series(run_dir: Path | str):
        Nx, Lx, files, _ = load_nonlinear_run(run_dir)
        T, Sig = load_nonlinear_Sigma_series(files)
        if T.size == 0:
            raise ValueError(f"No data in {run_dir}")

        dx = Lx / Nx
        k_grid = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)

        k_dom = np.full_like(T, np.nan, dtype=float)
        for i_t, Sx in enumerate(Sig):
            s = Sx - np.mean(Sx)
            ak = np.fft.rfft(s) / s.size
            amps = 2.0 * np.abs(ak[1:])  # skip k=0
            if amps.size == 0:
                continue
            a_max = np.max(amps)
            if a_max < amp_floor:
                continue
            j = int(np.argmax(amps)) + 1
            k_dom[i_t] = k_grid[j]
        return T, k_dom

    from pypalettes import load_cmap
    cmap = load_cmap("CarolMan")
    palette = cmap(np.linspace(0, 1, 5))
    color1 = palette[0]
    color2 = palette[1]
    color3 = palette[2]

    color4 = palette[2]

    T_lo, kdom_lo = _dominant_k_series(run_low)
    T_mid, kdom_mid = _dominant_k_series(run_mid)
    T_hi, kdom_hi = _dominant_k_series(run_high)

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # main panel: k_dom(t)
    ax.plot(
        T_lo, kdom_lo,
        color=color1, lw=1.5, marker="o", ms=2,
        label=label_low,
    )
    ax.plot(
        T_mid, kdom_mid,
        color=color2, lw=1.5, marker="o", ms=2,
        label=label_mid,
    )
    ax.plot(
        T_hi, kdom_hi,
        color=color3, lw=1.5, marker="s", ms=2,
        label=label_high,
    )
    ax.axhline(
        k_max, color="k", ls="--", lw=1.5,
        label=rf"EVP $K_\mathrm{{max}}\approx {k_max:.1f}$",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"$t[\Omega^{-1}]$")
    ax.set_ylabel(r"$K_\mathrm{dom}(t)$")
    ax.set_xlim(T_lo[0], T_lo[-1])
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    # ---------- inset: γ(k) vs k ----------
    ax_in = inset_axes(
        ax,
        width="55%", height="35%",
        loc="center right",
        borderpad=1.0,
    )
    ax_in.plot(ks, growth, color="k", lw=1.5)
    ax_in.set(xlim = (ks[0], ks[-1]))
    ax_in.axvline(k_max, color="k", ls="--", lw=1.0)
    ax_in.set_xscale("log")
    ax_in.set_yscale("log")
    ax_in.set_xlabel(r"$K$", fontsize=8)
    ax_in.set_ylabel(r"$\gamma(K)$", fontsize=8)
    ax_in.tick_params(labelsize=7)

    fig.tight_layout()
    return fig, (ax, ax_in)




def plot_piecewise_noise_comparison(
    run_powerlaw: str | Path,
    run_piecewise: str | Path,
    label_powerlaw: str = r"power-law $D(\Sigma)$",
    label_piecewise: str = r"piecewise $D(\Sigma)$",
    n_snap: int = 6,
    figsize: tuple[float, float] = (9.0, 3.0),
):
    """
    Compare a noise run with the original power-law closure against
    a run with piecewise-saturated diffusion.

    Layout:
        - Left panel: max_x Sigma(x,t) vs t (log y) for both runs.
        - Right panel: Sigma(x,t) snapshots for the *piecewise* run only.

    Parameters
    ----------
    run_powerlaw, run_piecewise : path-like
        Nonlinear native runs started from identical noise IC but with
        different diffusion closures.

    label_powerlaw, label_piecewise : str
        Legend / title labels.

    n_snap : int
        Number of snapshots shown for the piecewise run.

    figsize : (float, float)
        Overall figure size.
    """
    run_powerlaw = Path(run_powerlaw)
    run_piecewise = Path(run_piecewise)

    # ----- power-law run (may blow up early) -----
    Nx_pl, Lx_pl, files_pl, man_pl = load_nonlinear_run(run_powerlaw)
    T_pl, Sig_pl = load_nonlinear_Sigma_series(files_pl)
    if T_pl.size == 0:
        raise ValueError(f"No data in power-law run {run_powerlaw}")
    Smax_pl = np.nanmax(Sig_pl, axis=1)

    # ----- piecewise run (saturating) -----
    Nx_pw, Lx_pw, files_pw, man_pw = load_nonlinear_run(run_piecewise)
    T_pw, Sig_pw = load_nonlinear_Sigma_series(files_pw)
    if T_pw.size == 0:
        raise ValueError(f"No data in piecewise run {run_piecewise}")
    with np.load(files_pw[0]) as Z1:
        x_pw = np.asarray(Z1["x"])
    Smax_pw = np.nanmax(Sig_pw, axis=1)

    # ----- figure layout: 1×2 -----
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1, 2,
        width_ratios=[1.1, 1.0],
        wspace=0.35,
        figure=fig,
    )

    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # simple color palette
    try:
        from pypalettes import load_cmap
        cmap_main = load_cmap("CarolMan")
        colors = cmap_main(np.linspace(0, 1, 4))
        c_pl, c_pw = colors[0], colors[2]
    except Exception:
        c_pl, c_pw = "C0", "C1"

    # ----- left: max Σ(t) for both runs -----
    ax_left.plot(T_pl, Smax_pl, color=c_pl, lw=2, label=label_powerlaw)
    ax_left.plot(T_pw, Smax_pw, color=c_pw, lw=2, label=label_piecewise)

    ax_left.set_yscale("log")
    ax_left.set_xlabel(r"$t[\Omega^{-1}]$")
    ax_left.set_ylabel(r"$\max_x \Sigma(x,t)$")
    ax_left.set_xlim(T_pl[0], max(T_pl[-1], T_pw[-1]))
    ax_left.legend(frameon=False, fontsize=9)

    # ----- helper: choose snapshots from the *piecewise* run -----
    def _choose_snapshots(T, Sig, n):
        finite = np.all(np.isfinite(Sig), axis=1)
        T_f = T[finite]
        Sig_f = Sig[finite]
        if T_f.size == 0:
            return np.array([]), np.empty((0,) + Sig.shape[1:])
        idx = np.linspace(0, len(T_f) - 1, n).astype(int)
        return T_f[idx], Sig_f[idx]

    T_snap_pw, Sig_snap_pw = _choose_snapshots(T_pw, Sig_pw, n_snap)

    # snapshot colormap
    try:
        from pypalettes import load_cmap
        cmap_snap = load_cmap("Hiroshige")
    except Exception:
        cmap_snap = plt.get_cmap("viridis")

    # ----- right: piecewise snapshots -----
    for j, (tt, Sx) in enumerate(zip(T_snap_pw, Sig_snap_pw)):
        frac = 0.1 + 0.8 * (j / max(1, n_snap - 1))
        col = cmap_snap(frac)
        ax_right.plot(x_pw, Sx, color=col, lw=1.5,
                      label=fr"$t={tt:.1f}\,[\Omega^{{-1}}]$")

    ax_right.set_xlabel(r"$x$")
    ax_right.set_ylabel(r"$\Sigma(x,t)$")
    ax_right.set_title(label_piecewise)
    ax_right.set_xlim(x_pw[0], x_pw[-1])
    ax_right.legend(frameon=True, fontsize=8, ncols=2)

    fig.tight_layout()
    return fig, (ax_left, ax_right)


def plot_pw_saturation_three(
    run_powerlaw: str | Path,
    run_pw1: str | Path,
    run_pw2: str | Path,
    label_powerlaw: str = r"power-law $D(\Sigma)$",
    label_pw1: str = r"piecewise $D(\Sigma)$, $\Sigma_{\rm sat}=1.5\,\Sigma_0$",
    label_pw2: str = r"piecewise $D(\Sigma)$, $\Sigma_{\rm sat}=2\,\Sigma_0$",
    figsize: tuple[float, float] = (5.0, 3.0),
):
    """
    Single-panel comparison of max_x Sigma(t) for three runs:
    - a pure power-law D(Sigma),
    - two piecewise closures with different saturation amplitudes.

    Parameters
    ----------
    run_powerlaw, run_pw1, run_pw2 : path-like
        Nonlinear run directories (native backend).

    label_powerlaw, label_pw1, label_pw2 : str
        Legend labels for the three curves.

    figsize : (float, float)
        Figure size passed to matplotlib.
    """
    def _load_max_series(run_dir: str | Path):
        run_dir = Path(run_dir)
        Nx, Lx, files, man = load_nonlinear_run(run_dir)
        T, Sig = load_nonlinear_Sigma_series(files)
        if T.size == 0:
            raise ValueError(f"No data in run {run_dir}")
        Smax = np.nanmax(Sig, axis=1)
        return T, Smax

    # load all three
    T_pl, Smax_pl   = _load_max_series(run_powerlaw)
    T_pw1, Smax_pw1 = _load_max_series(run_pw1)
    T_pw2, Smax_pw2 = _load_max_series(run_pw2)

    # figure
    fig, ax = plt.subplots(figsize=figsize)

    # colors
    try:
        from pypalettes import load_cmap
        cmap = load_cmap("AsteroidCity1")
        palette = cmap(np.linspace(0, 1, 5))
        c1, c2, c3 = palette[0], palette[2], palette[4]
    except Exception:
        # fallback
        colors = plt.get_cmap("tab10").colors
        c1, c2, c3 = colors[0], colors[1], colors[2]

    ax.plot(T_pl,  Smax_pl,   color=c1, lw=2.0, label=label_powerlaw)
    ax.plot(T_pw1, Smax_pw1,  color=c2, lw=2.0, label=label_pw1)
    ax.plot(T_pw2, Smax_pw2,  color=c3, lw=2.0, label=label_pw2)


    # axis formatting
    ax.set_yscale("log")
    t_min = min(T_pl[0],  T_pw1[0],  T_pw2[0])
    t_max = max(T_pl[-1], T_pw1[-1], T_pw2[-1])
    #t_min = T_pw2[0]; t_max = T_pw2[-1]
    ax.set_xlim(t_min, t_max)

    ax.set_xlabel(r"$t[\Omega^{-1}]$")
    ax.set_ylabel(r"$\max_x \Sigma(x,t)$")

    ax.legend(frameon=False, fontsize=9, loc = "lower right")

    fig.tight_layout()
    return fig, ax