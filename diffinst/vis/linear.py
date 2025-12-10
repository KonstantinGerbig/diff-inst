from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from .. import Config
from ..linear_ops import evp_solve_at_k


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
    amp_floor_frac: float = 0.05,   # floor for hodograph masking (overstable only)
    use_real_parts: bool = True,
    figsize=(13, 3.5),
):
    """
    1×N panel figure. Each panel = one branch (stable, diff-slope, visc-slope,
    overstable). Main axes: normalized |Sigma_k(t)|. Insets:
      - for non-overstable branches: instantaneous growth rate γ_num(t)
        compared to EVP γ;
      - for 'overstable': hodograph (Re[v_yk] vs Re[v_xk]) from nonlinear TD.

    Parameters
    ----------
    branches : dict
        Mapping branch_name -> dict with possible keys:

        {
          "run_lin_native": Path or str or None,
          "run_nl_native": Path or str or None,
          "run_nl_dedalus": Path or str or None,
        }

    k_phys : float
        Physical wavenumber used for the eigenmode IC / amplitude extraction.
    """
    # Canonical order if present
    canonical_order = ["stable", "diff-slope", "visc-slope", "overstable"]
    branch_order = [b for b in canonical_order if b in branches] or list(branches.keys())

    n_branch = len(branch_order)
    if n_branch == 0:
        raise ValueError("No branches provided to plot_linear_robustness_with_hodographs.")

    fig, axes = plt.subplots(1, n_branch, figsize=figsize, sharey=True)
    if n_branch == 1:
        axes = [axes]

    from pypalettes import load_cmap
    cmap = load_cmap("Juarez") 
    palette = cmap(np.linspace(0, 1, 3))

    sigma_hat_color = palette[0]
    inset_gamma_color = palette[1]
    hodograph_color = palette[2]

    styles = {
        "lin_native": dict(ls="--",  color=sigma_hat_color, label="linear TD "),
        "nl_native":  dict(ls="-", color=sigma_hat_color, label="IVP (nonlinear)"),
        "nl_ded":     dict(ls=":",  color=sigma_hat_color, label="nonlinear TD (Dedalus)"),
        "evp":        dict(ls="-",  color="k",  label="EVP"),
    }

    

    for ax, name in zip(axes, branch_order):
        runs = branches[name]

        # ---------- Load amplitudes ----------
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

        # Pick a config for EVP
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

        gamma = None
        if cfg is not None and series:
            try:
                gamma = evp_gamma(cfg, k_phys)
                T_ref = series[0][1]
                t0 = T_ref[0]
                A_evp = np.exp(gamma * (T_ref - t0))
                series.append(("evp", T_ref, A_evp))
            except Exception as e:
                print(f"[plot_linear_robustness] EVP failed for branch {name}: {e}")
                gamma = None

        if not series:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue

        # ---------- Main growth curves ----------
        for kind, T, A in series:
            if len(A) == 0:
                continue
            A0 = float(A[0])
            y = A / A0 if A0 != 0.0 else A
            style = styles.get(kind, dict(ls="-", color="k"))
            label = style.get("label", kind.replace("_", " "))
            ax.plot(T, y, style["ls"], color=style["color"], label=label, lw = 2)
            
            ax.set(xlim = (T[0], T[-1]))


        ax.set_yscale("log")
            
        #ax.set_title(name)
        #ax.grid(True, which="both", alpha=0.2)

        if ax is axes[0]:
            ax.set_ylabel(r"$|\Sigma_k|/|\Sigma_k(t_0)|$")
        ax.set_xlabel(r"$t[\Omega^{-1}]$")

        if ax is axes[0]:
            ax.legend(frameon=False, fontsize=10, loc="lower left")

        if ax is axes[0]:
            inset = inset_axes(
                ax,
                width="48%", height="48%",
                loc="upper right",
            )
            
        else:
            inset = inset_axes(
                ax,
                width="48%", height="48%",
                #bbox_to_anchor=(0.515, 0.09, 0.48, 0.48),   # ← shift upward
                #bbox_transform=ax.transAxes,
                loc="lower right",
            )

        
        # ---------- Insets ----------
        if name != "overstable":
            # ---- γ_num(t) inset ----
            

            # Prefer nonlinear native for γ_num, fall back to linear
            gamma_src = nl_native or lin_native
            if (gamma_src is not None) and (gamma is not None):
                T_g, A_g = gamma_src
                mask = A_g > 0
                T_g = T_g[mask]
                A_g = A_g[mask]

                if len(T_g) >= 3:
                    lnA = np.log(A_g)
                    gamma_num = (lnA[2:] - lnA[:-2]) / (T_g[2:] - T_g[:-2])
                    T_mid = T_g[1:-1]

                    inset.plot(T_mid, gamma_num, color=inset_gamma_color, lw=2)
                    inset.axhline(gamma, color="k", ls="-", lw=1.0)

                    inset.set(xlim = (T_mid[0], T_mid[-1]))

                    # auto y-limits with some padding
                    g_min = min(np.min(gamma_num), gamma)
                    g_max = max(np.max(gamma_num), gamma)
                    pad = 0.15 * (g_max - g_min if g_max != g_min else abs(gamma) + 1e-3)
                    inset.set_ylim(g_min - pad, g_max + pad)

                    inset.tick_params(labelsize=9)
                    # no axis labels in inset to avoid overlap
                    #inset.set_xticklabels([])
                    #inset.set_yticklabels([])
                    inset.set_ylabel(r"$\gamma_{\rm num}[\Omega]$", fontsize=10)
                    inset.set_xlabel(r"$t[\Omega^{-1}]$", fontsize=10)
                else:
                    inset.text(0.5, 0.5, "too short", ha="center", va="center",
                               transform=inset.transAxes, fontsize=6)
            else:
                inset.text(0.5, 0.5, "no γ", ha="center", va="center",
                           transform=inset.transAxes, fontsize=6)

        else:
            # ---- Hodograph inset only for overstable branch ----

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
                r_full = np.sqrt(np.abs(vx_k)**2 + np.abs(vy_k)**2)
                if r_full.size > 0:
                    r_max = np.max(r_full)
                    mask = r_full >= amp_floor_frac * r_max
                    if not np.any(mask):
                        mask[:] = True
                    vx_k = vx_k[mask]
                    vy_k = vy_k[mask]

                    X = vx_k.real if use_real_parts else vx_k.imag
                    Y = vy_k.real if use_real_parts else vy_k.imag

                    style = styles.get(kind_h, dict(color="C1"))
                    #color = style["color"]
                    color = hodograph_color
                    inset.plot(X, Y, lw=2, color=color)
                    inset.scatter(X[0],  Y[0],  s=25, facecolors="none",
                                  edgecolors=color, zorder=3)
                    inset.scatter(X[-1], Y[-1], s=25, facecolors=color,
                                  edgecolors="k", zorder=3)

                    inset.tick_params(labelsize=8)
                    #inset.set_xticklabels([])
                    #inset.set_yticklabels([])
                    inset.set_aspect("equal", adjustable="datalim")
                    inset.set_xlabel(r"$\Re[\hat{v}_x]$", fontsize=10)
                    inset.set_ylabel(r"$\Re[\hat{v}_y]$", fontsize=10)
                else:
                    inset.text(0.5, 0.5, "no hodograph", ha="center",
                               va="center", transform=inset.transAxes, fontsize=6)
            else:
                inset.text(0.5, 0.5, "no hodograph", ha="center",
                           va="center", transform=inset.transAxes, fontsize=6)

        if ax is not axes[0]:
            inset.xaxis.set_ticks_position('top')
            inset.xaxis.set_label_position('top')
            inset.tick_params(axis='x', labeltop=True, labelbottom=False)
        
        axes[0].set_title("(a) stable")
        axes[1].set_title(r"(b) $D$-slope driven instability ")
        axes[2].set_title(r"(c) $\nu$-slope driven instability")
        axes[3].set_title("(d) Overstability")

    fig.tight_layout()
    return fig, axes


def plot_diffusion_slope_gas_vs_dust(
    linear_runs: Dict[str, Path | str],
    k_phys: float,
    normalize: bool = True,
    figsize=(6, 4),
):
    """
    Plot |Sigma_k|(t) for several diffusion-slope setups (dust-only vs various
    gas-inclusive configs), using linear native TD runs.

    Parameters
    ----------
    linear_runs : dict
        Mapping label -> run_dir (Path or str). Any runs that fail to load
        will be skipped.

    k_phys : float
        Physical wavenumber at which to extract the amplitude.

    normalize : bool
        If True, normalize each curve by its initial amplitude so the slopes
        directly reflect growth rates.

    figsize : tuple
        Matplotlib figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    any_plotted = False

    for label, run_dir in linear_runs.items():
        # robust loading via analysis_api helper
        data = safe_load_amplitude(load_linear_amplitude, run_dir, k_phys)
        if data is None:
            print(f"[plot_diffusion_slope_gas_vs_dust] skipping {label!r}: could not load amplitude.")
            continue

        T, A = data
        if A.size == 0:
            print(f"[plot_diffusion_slope_gas_vs_dust] skipping {label!r}: empty amplitude array.")
            continue

        y = A.astype(float)
        if normalize:
            A0 = float(y[0])
            if A0 != 0.0:
                y = y / A0

        ax.plot(T, y, label=label)

        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.set_yscale("log")
        ax.set_xlabel(r"$t\,\Omega^{-1}$")
        ylabel = r"$|\Sigma_k|/|\Sigma_k(t_0)|$" if normalize else r"$|\Sigma_k|$"
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig, ax


def plot_diffusion_slope_family(
    configs: Dict[str, Path | str],
    runs_nl: Dict[str, Path | str],
    k_phys: float,
    kmin: float = 10.0,
    kmax: float = 1e4,
    nk: int = 200,
    labels: Optional[Dict[str, str]] = None,
    figsize=(8.0, 3.5),
):
    """
    Two-panel figure for a family of diffusion-slope experiments.

    Left panel: linear growth rate γ(k) from EVP for each config
                (γ = Re[ω_0(k)] of the most unstable eigenvalue).

    Right panel: nonlinear IVP evolution of |Σ_k(t)| / |Σ_k(t_0)| at a fixed k_phys
                 for each run, with the corresponding EVP growth curve
                 exp[γ(k_phys) (t - t_0)] overlaid in the same color.

    Parameters
    ----------
    configs : dict
        key -> path to experiment YAML (used for EVP sweeps).
    runs_nl : dict
        key -> nonlinear run directory (used for IVP amplitudes).
    k_phys : float
        Physical wavenumber for the time-domain amplitude extraction.
    kmin, kmax : float
        k-range (physical) for the EVP sweep.
    nk : int
        Number of k points (log-spaced) for the EVP sweep.
    labels : dict, optional
        key -> label string for legend; defaults to key.
    figsize : tuple
        Figure size.
    """

    keys = list(configs.keys())
    if not keys:
        raise ValueError("No configs supplied to plot_diffusion_slope_family.")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

    ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    # store cfg and color for reuse between panels
    cfg_by_key: Dict[str, Config] = {}
    
    from pypalettes import load_cmap
    cmap = load_cmap("Callanthias_australis") 
    palette = cmap(np.linspace(0, 1, 4))

    color_by_key: Dict[str, str] = {
        "dust":palette[0],
        "fid":palette[1],
        "high_eps":palette[2],
        "low_nu_g":palette[3],
    }

    

    axL.axvline(100, color = "black", lw = 1, ls = "--", label = r"$k = 100$")

    # ---- left panel: γ(k) via EVP ----
    for i, key in enumerate(keys):
        cfg_path = Path(configs[key])
        if not cfg_path.exists():
            print(f"[plot_diffusion_slope_family] config missing for {key}: {cfg_path}")
            continue
        

        cfg = Config.from_yaml(cfg_path)
        cfg_by_key[key] = cfg

        gamma_k = np.empty_like(ks, dtype=float)
        for j, k in enumerate(ks):
            w, _ = evp_solve_at_k(cfg, float(k))
            gamma_k[j] = w[0].real

        label = labels.get(key, key) if labels is not None else key
        color = color_by_key.get(key, None)
        line = axL.plot(ks, gamma_k, lw=2, label=label, color = color)[0]
        

    
    axL.set(ylim = (1e-4,1), xlim = (ks[0], ks[-1]))
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel(r"$k$")
    axL.set_ylabel(r"$\gamma(k)$")
    axL.legend(frameon=False, loc = "upper left")

    # ---- right panel: nonlinear IVP amplitude + EVP prediction at k_phys ----
    for key in keys:
        run_dir = runs_nl.get(key)
        if run_dir is None:
            continue

        # nonlinear amplitude |Σ_k(t)|
        loaded = safe_load_amplitude(load_nonlinear_amplitude, run_dir, k_phys)
        if loaded is None:
            print(f"[plot_diffusion_slope_family] no nonlinear data for {key}: {run_dir}")
            continue
        T, A = loaded
        if len(A) == 0:
            continue

        A0 = float(A[0])
        y = A / A0 if A0 != 0.0 else A

        color = color_by_key.get(key, None)
        label = labels.get(key, key) if labels is not None else key
        axR.plot(T, y, lw=2, color=color, label=label)

        # EVP prediction at k_phys, same cfg + color
        cfg = cfg_by_key.get(key)
        if cfg is not None:
            try:
                gamma_phys = evp_gamma(cfg, k_phys)
                y_th = np.exp(gamma_phys * (T - T[0]))
                axR.plot(T, y_th, lw=1.5, ls="--", color=color)
            except Exception as e:
                print(f"[plot_diffusion_slope_family] EVP at k={k_phys} failed for {key}: {e}")

    axR.set(xlim = (T[0], T[-1]))
    axR.set_yscale("log")
    axR.set_xlabel(r"$t\,[\Omega^{-1}]$")
    axR.set_ylabel(r"$|\Sigma_k|/|\Sigma_k(t_0)|$")
    #axR.legend(frameon=False)

    fig.tight_layout()
    return fig, (axL, axR)