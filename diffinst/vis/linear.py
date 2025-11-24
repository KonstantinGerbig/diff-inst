# diffinst/viz/linear.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..analysis_api import (
    load_linear_amplitude,
    load_nonlinear_amplitude,
    load_config_from_run,
    evp_gamma,
    load_nonlinear_run,
    load_nonlinear_field_series,
    nearest_k_index,
    safe_load_amplitude,
)


def plot_linear_robustness(
    branches: Dict[str, Dict[str, Optional[Path | str]]],
    k_phys: float,
    figsize=(10, 8),
):
    """
    Multi-panel figure: for each branch, plot normalized |Sigma_k|(t) from

      - native linear TD
      - native nonlinear TD (small amplitude)
      - Dedalus nonlinear TD (if present)
      - EVP prediction from eigenvalue gamma

    Parameters
    ----------
    branches : dict
        Mapping branch_name -> dict with keys (any subset may be None):

            {
              "run_lin_native": Path or None,
              "run_nl_native": Path or None,
              "run_nl_dedalus": Path or None,
            }

    k_phys : float
        Physical wavenumber used for the eigenmode IC / amplitude extraction.
    """
    branch_names = list(branches.keys())
    n_branch = len(branch_names)
    if n_branch == 0:
        raise ValueError("No branches provided to plot_linear_robustness.")

    ncols = 2
    nrows = int(np.ceil(n_branch / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])

    for i, name in enumerate(branch_names):
        ax = axes_flat[i]
        runs = branches[name]

        # --- load amplitudes safely via analysis_api helper
        lin_native = safe_load_amplitude(
            load_linear_amplitude,
            runs.get("run_lin_native"),
            k_phys,
            label=f"{name} / linear TD (native)",
        )
        nl_native = safe_load_amplitude(
            load_nonlinear_amplitude,
            runs.get("run_nl_native"),
            k_phys,
            label=f"{name} / nonlinear TD (native)",
        )
        nl_ded = safe_load_amplitude(
            load_nonlinear_amplitude,
            runs.get("run_nl_dedalus"),
            k_phys,
            label=f"{name} / nonlinear TD (Dedalus)",
        )

        series = []

        if lin_native is not None:
            T, A = lin_native
            series.append((T, A, "linear TD (native)", "-"))

        if nl_native is not None:
            T, A = nl_native
            series.append((T, A, "nonlinear TD (native, small amp)", "--"))

        if nl_ded is not None:
            T, A = nl_ded
            series.append((T, A, "nonlinear TD (Dedalus)", ":"))

        # EVP line: only if we have something to compare against
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
            try:
                gamma = evp_gamma(cfg, k_phys)
                T_ref = series[0][0]
                t0 = T_ref[0]
                A_evp = np.exp(gamma * (T_ref - t0))
                series.append((T_ref, A_evp, f"EVP: γ={gamma:.3g}", "k-"))
            except Exception as e:
                print(f"[plot_linear_robustness] EVP failed for branch {name}: {e}")

        if not series:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue

        # normalize by initial amplitude
        for T, A, label, ls in series:
            if len(A) == 0:
                continue
            A0 = float(A[0])
            y = A / A0 if A0 != 0.0 else A
            ax.plot(T, y, ls, label=label)

        ax.set_yscale("log")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)

        if i % ncols == 0:
            ax.set_ylabel(r"$|\Sigma_k|/|\Sigma_k(t_0)|$")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("t")

        if i == 0:
            ax.legend(frameon=False, fontsize=8)

    # hide unused axes if any
    for j in range(n_branch, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Linear-regime robustness at k={k_phys}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes