# Diffusive Instability (dust–gas) — clean numerical stack

Goal: a fast, clear, reproducible **1D** stack for the dust–gas diffusive instability  
(axes: **EVP → linear time-domain → nonlinear saturation**), with switchable solvers:

- Native 1D pseudo-spectral IMEX (NumPy / FFT)
- Optional Dedalus backend for cross-checks

All runs are configured via YAML + a small set of CLI overrides.

---

## Status / Capabilities

### Core infrastructure

- ✅ **Repo scaffolded**: `Config` dataclass → grid handling → streaming I/O (`StreamWriter`) → runners in `scripts/`
- ✅ **Config system**: `defaults.yaml` + `experiments/*.yaml`, merged into an immutable `Config`
- ✅ **Streaming I/O**:
  - `run.json` manifest per run (stores config path, Nx, Lx, backend, dt, tstop,…)
  - `metrics.jsonl` with time-series diagnostics (e.g. `mode1_amp`, `mass`)
  - `checkpoints/chk_****.npz` with `t, x, Sigma, vx, vy, uy`
- ✅ **Tests**: basic unit tests pass (`pytest`)

### Linear theory

- ✅ **EVP (dispersion relation)**:
  - Direct 4×4 eigenvalue problem implemented in `linear_ops.evp_solve_at_k`
  - CLI sweep script (in `scripts/`) to scan over physical wavenumber `k` and tabulate growth rates
- ✅ **Linear time-domain solver (native)**:
  - IMEX pseudo-spectral integrator for the linearized system
  - Uses the same grid and closures as the nonlinear solver (1D periodic Real FFT)
- ✅ **Consistency checks**:
  - Notebook / analysis helpers to compare:
    - EVP growth rate `γ(k)`
    - Linear TD amplitude evolution `|Σ_k(t)|`
    - Nonlinear runs in the small-amplitude regime

### Nonlinear time-domain

- ✅ **Native nonlinear solver** (`diffinst/solvers/native_nonlinear.py`):
  - 1D pseudo-spectral IMEX with:
    - Dust Σ, vx, vy
    - Gas uy (axisymmetric closure, always present)
  - Explicit part: advection + all variable-coefficient fluxes
  - Implicit part: frozen Laplacian terms for dust/gas viscosity
  - Writes standard `run.json` + metrics + checkpoints
- ✅ **Initial condition options (native)**:
  - `seed_mode="eigen"`: seed from EVP eigenvector at given `k_phys`
  - `seed_mode="cos"`: simple cosine perturbation on Σ
  - `seed_mode="noise"`: Σ = S₀ + Gaussian noise with controlled physical amplitude
  - `--init-from path.npz`: load a precomputed IC (`Sigma, vx, vy, uy`) from disk
  - Amplitudes:
    - `--amp` + `--amp-physical` / fractional
    - `--amp-metric=max|rms` for eigenmode normalization
- ✅ **Grid override (native)**:
  - `Nx` can be overridden at run time via CLI (used to build a compatible `Grid1D` and operators)
  - Manifests / loaders respect the effective `Nx` used in a given run
- ✅ **Analysis API** (`diffinst/analysis_api.py`):
  - `load_manifest`, `load_config_from_run`, `list_checkpoints`, `load_metrics`
  - `load_linear_run`, `load_linear_Sigma_series`
  - `load_nonlinear_run`, `load_nonlinear_Sigma_series`
  - Resolution-independent mode amplitude helpers:
    - `nearest_k_index(Lx, Nx, k_phys)`
    - `amplitude_at_k_from_sigma` (returns physical cosine amplitude Aₖ)
    - `load_linear_amplitude`, `load_nonlinear_amplitude`
  - IC helpers: `save_ic_npz`, `load_ic_npz`
  - EVP helper: `evp_gamma(cfg, k_phys)`

### Dedalus backend

- ✅ **Dedalus nonlinear backend** (`diffinst/solvers/dedalus_backend.py`):
  - 1D `RealFourier` domain in Dedalus (x only; y is a dummy coord)
  - Implements the same dust–gas equations as the native solver
  - Supports:
    - Eigenmode seeding (EVP-based)
    - Cosine seeding
    - External IC from `.npz`
  - Uses fixed RK443 timestep with user-supplied `dt`
  - Writes checkpoints compatible with the analysis API and the native runs
  - Logs progress and aborts cleanly on NaNs / Infs
- ⚠️ **Current behavior**:
  - Good agreement with native solver & EVP in the **linear / early nonlinear** regime
  - At very large amplitudes, runs can still produce overflows / NaNs  
    (we’ve experimented with floors and clamped closures but haven’t fully “hardened”
    the high-amplitude regime yet)

### IC generation & eigenmode experiments

- ✅ **Eigenmode IC generator** (helper script, called from notebook / CLI):
  - Uses a chosen experiment config + `k_phys` + `Nx` override to:
    - Solve EVP
    - Build a real eigenmode in x
    - Normalize it to a desired physical `amp_phys` in Σ
    - Save to `runs/ic_k*_eigen.npz` with `Sigma, vx, vy, uy` and metadata
  - This IC is then reused for:
    - Linear TD runs (native)
    - Nonlinear native runs
    - Nonlinear Dedalus runs
  - → “Equal ICs @ k = …” test: EVP vs linear TD vs nonlinear TD all consistent in growth regime

---

## What’s still to do

### Nonlinear saturation / robustness

- 🚧 **High-amplitude nonlinear regime**:
  - Right now, once the mode becomes very large, both native and Dedalus can still hit:
    - Large gradients in Σ, vx, vy
    - Overflows inside nonlinear flux terms
    - FFT warnings when spectra get extremely steep
  - We’ve added:
    - Hard floors on Σ inside the closures
    - Optional clamping of Σ in the native stepper
  - Still open:
    - Decide on a principled regularization strategy (e.g. physical hyper-diffusion, slope limiting, or a stronger floor)
    - Possibly use adaptive dt (CFL-style) in the native solver for very nonlinear runs
    - Document the regime where the solver is “trusted” vs “formally unstable but informative”

### Convergence & verification

- 🚧 **Convergence harness**:
  - Systematic scripts to:
    - Sweep `Nx` and `dt` and compare:
      - Growth rate vs EVP
      - Mode shapes vs EVP eigenvectors
      - Invariants (mass conservation)
    - Compare native vs Dedalus integrators at matched parameters
  - Add regression tests that:
    - Check small-amplitude nonlinear runs reproduce linear growth rates
    - Check mass conservation to a specified tolerance

### Plotting / paper-figure layer

- 🚧 **Paper-quality figure scripts**:
  - High-level plotting routines for:
    - `|Σ_k(t)|` for multiple runs on the same axes (linear TD vs nonlinear vs EVP)
    - Σ(x,t) “waterfall” or stacked snapshots (already prototyped in notebooks)
    - k-spectra from noise runs as a function of time
    - Parameter sweeps (e.g. β_diff, D₀, ν₀) summarized across runs
  - Wrap the notebook snippets into **reusable functions** under a `figures/` or `scripts/fig_*.py` namespace

### Documentation / ergonomics

- 🚧 **CLI & config docs**:
  - Document scripts such as:
    - `scripts/run_linear.py`
    - `scripts/run_nonlinear.py`
    - IC generation helper
  - Clearly separate:
    - **Physical parameters**: stored in YAML (`experiments/*.yaml`, `defaults.yaml`)
    - **Numerical parameters**: `Nx`, `dt`, `stop_time`, `save_stride`, `seed_mode`, etc.,
      which can be overridden on the CLI
  - Add small “recipes”:
    - “EVP sweep & compare to linear TD”
    - “Nonlinear eigenmode run from the same IC”
    - “Noise run: spectrum evolution & mode selection”

- 🚧 **Dedalus environment docs**:
  - Provide a short recipe for setting up a `dedalus3` conda env and installing this package there
  - Clarify that Dedalus is **optional** and used mainly for cross-checks

---

## Install (recommended: venv)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .  # editable install of this repo
pip install scipy matplotlib pytest