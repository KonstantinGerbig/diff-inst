from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..analysis_api import (
    load_linear_amplitude,
    load_nonlinear_amplitude,
    load_config_from_run,
    evp_gamma,
    safe_load_amplitude,
    load_mode_series_linear,
    load_mode_series_nonlinear,
)


def plot_linear_robustness_with_hodographs(
    branches: Dict[str, Dict[str, Optional[Path | str]]],
    k_phys: float,
    amp_floor_frac: float = 0.05,   # mask very early times in hodograph
    use_real_parts: bool = True,
    figsize=(14, 3.5),
):
    """
    Make a 1×N panel figure. Each panel = one branch (e.g. stable, diff-slope,
    visc-slope, overstable). Main axes: normalized |Sigma_k(t)|. Inset: hodograph
    (vy_k vs vx_k) for one representative run.

    Parameters
    ----------
    branches : dict
        Mapping branch_name -> dict with possible keys:

        {
          "run_lin_native": Path or str or None,
          "run_nl_native": Path or str or None,
          "run_nl_dedalus": Path or str or None,
        }

        The function is robust to missing entries or missing runs.

    k_phys : float
        Physical wavenumber used for the eigenmode IC / amplitude extraction.

    amp_floor_frac : float
        For the hodograph: we only plot times where |v_k| >= amp_floor_frac * max|v_k|,
        to avoid noisy early points.

    use_real_parts : bool
        If True: plot Re(vx_k), Re(vy_k). If False: Im parts.

    figsize : tuple
        Overall figure size.
    """
    # Canonical order if available; otherwise just keep whatever was passed.
    canonical_order = ["stable", "diff-slope", "visc-slope", "overstable"]
    branch_order = [b for b in canonical_order if b in branches] or list(branches.keys())

    n_branch = len(branch_order)
    if n_branch == 0:
        raise ValueError("No branches provided to plot_linear_robustness_with_hodographs.")

    fig, axes = plt.subplots(1, n_branch, figsize=figsize, sharey=True)
    if n_branch == 1:
        axes = [axes]

    # line styles per solver (same in every panel)
    styles = {
        "lin_native": dict(ls="-",  color="C0", label="linear TD (native)"),
        "nl_native":  dict(ls="--", color="C1", label="nonlinear TD (native)"),
        "nl_ded":     dict(ls=":",  color="C2", label="nonlinear TD (Dedalus)"),
        "evp":        dict(ls="-",  color="k",  label="EVP"),
    }

    for ax, name in zip(axes, branch_order):
        runs = branches[name]

        # ---- Load amplitudes safely ----
        lin_native = safe_load_amplitude(
            load_linear_amplitude, runs.get("run_lin_native"), k_phys
        )
        nl_native = safe_load_amplitude(
            load_nonlinear_amplitude, runs.get("run_nl_native"), k_phys
        )
        nl_ded = safe_load_amplitude(
            load_nonlinear_amplitude, runs.get("run_nl_dedalus"), k_phys
        )

        series = []

        if lin_native is not None:
            T, A = lin_native
            series.append(("lin_native", T, A))
        if nl_native is not None:
            T, A = nl_native
            series.append(("nl_native", T, A))
        if nl_ded is not None:
            T, A = nl_ded
            series.append(("nl_ded", T, A))

        # EVP using the first available config (any solver)
        cfg = None
        for key in ("run_lin_native", "run_nl_native", "run_nl_dedalus"):
            rd = runs.get(key)
            if rd is None:
                continue
            rd = Path(rd)
            if not rd.exists():
                continue
            try:
                cfg = load_config_from_run(rd)
                break
            except Exception:
                continue

        if cfg is not None and series:
            gamma = evp_gamma(cfg, k_phys)
            T_ref = series[0][1]
            t0 = T_ref[0]
            A_evp = np.exp(gamma * (T_ref - t0))
            series.append(("evp", T_ref, A_evp))

        if not series:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue

        # ---- Plot growth curves, normalized by initial amplitude ----
        for kind, T, A in series:
            if len(A) == 0:
                continue
            A0 = float(A[0])
            y = A / A0 if A0 != 0.0 else A
            style = styles.get(kind, dict(ls="-", color="k"))
            label = style.get("label", kind.replace("_", " "))
            ax.plot(T, y, style["ls"], color=style["color"], label=label)

        ax.set_yscale("log")
        ax.set_title(name)
        #ax.grid(True, which="both", alpha=0.3)

        # y-label only on left-most panel
        if ax is axes[0]:
            ax.set_ylabel(r"$|\Sigma_k|/|\Sigma_k(t_0)|$")
        ax.set_xlabel("t")

        # only first panel gets legend
        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=8)

        # ---- Hodograph inset (pick one representative run) ----
        inset = inset_axes(ax, width="45%", height="45%", loc="lower right")

        # preference order: nonlinear native, then linear, then Dedalus
        hod_data = None
        if runs.get("run_nl_native") is not None:
            try:
                hod_data = ("nl_native",) + load_mode_series_nonlinear(runs["run_nl_native"], k_phys)
            except Exception as e:
                print(f"[hodograph] nonlinear native failed for {name}: {e}")
                hod_data = None
        if hod_data is None and runs.get("run_lin_native") is not None:
            try:
                hod_data = ("lin_native",) + load_mode_series_linear(runs["run_lin_native"], k_phys)
            except Exception as e:
                print(f"[hodograph] linear failed for {name}: {e}")
                hod_data = None
        if hod_data is None and runs.get("run_nl_dedalus") is not None:
            try:
                hod_data = ("nl_ded",) + load_mode_series_nonlinear(runs["run_nl_dedalus"], k_phys)
            except Exception as e:
                print(f"[hodograph] Dedalus failed for {name}: {e}")
                hod_data = None

        if hod_data is not None:
            kind_h, T_h, vx_k, vy_k = hod_data

            # mask out early tiny amplitudes
            r_full = np.sqrt(np.abs(vx_k)**2 + np.abs(vy_k)**2)
            if r_full.size > 0:
                r_max = np.max(r_full)
                mask = r_full >= amp_floor_frac * r_max
                if not np.any(mask):
                    mask[:] = True
                vx_k = vx_k[mask]
                vy_k = vy_k[mask]

                # normalize by max radius so all insets have similar scale
                r = np.sqrt(np.abs(vx_k)**2 + np.abs(vy_k)**2) + 1e-30
                vx_k = vx_k / np.max(r)
                vy_k = vy_k / np.max(r)

                X = vx_k.real if use_real_parts else vx_k.imag
                Y = vy_k.real if use_real_parts else vy_k.imag

                style = styles.get(kind_h, dict(color="C1"))
                inset.plot(X, Y, lw=1.5, color=style["color"])

                # mark start / end
                inset.scatter(X[0],  Y[0],  s=30, facecolors="none",
                              edgecolors=style["color"], zorder=3)
                inset.scatter(X[-1], Y[-1], s=30, facecolors=style["color"],
                              edgecolors="k", zorder=3)

                inset.set_xlabel(r"$\Re[\hat{v}_x]$", fontsize=7)
                inset.set_ylabel(r"$\Re[\hat{v}_y]$", fontsize=7)
                inset.tick_params(labelsize=6)
                inset.set_aspect("equal", adjustable="datalim")
            else:
                inset.text(0.5, 0.5, "no hodograph", ha="center",
                           va="center", transform=inset.transAxes, fontsize=7)
        else:
            inset.text(0.5, 0.5, "no hodograph", ha="center",
                       va="center", transform=inset.transAxes, fontsize=7)

    #fig.suptitle(f"Linear-regime robustness and hodographs at k={k_phys}", y=0.97)
    fig.tight_layout()#(rect=[0, 0, 1, 0.95])
    return fig, axes