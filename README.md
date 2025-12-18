
---

# Diffusive Instabilities in Dusty Disks — numerical companion repo

Fast, reproducible **1D axisymmetric** experiments for the diffusive instability in dusty disks, including an **incompressible, viscous gas** that responds **azimuthally** and couples to the dust via **drag backreaction**.

This repo supports the workflow used in the paper:

**EVP (dispersion relation) → linear time-domain validation → nonlinear evolution / breakdown → saturating closure experiments**

Two interchangeable backends:

- **Native**: 1D pseudo-spectral **IMEX** solver (NumPy + FFT) for speed and clarity  
- **Dedalus** (optional): independent implementation for cross-checks  

All runs are configured via **YAML** (+ a small set of CLI overrides). Every run writes a manifest + streaming diagnostics + checkpoints for analysis and plotting.

---

## What model is being solved?

This code integrates the paper’s 1D axisymmetric dust–gas system (dust: `Σ, v_x, v_y`; gas: `u_y`), with periodic boundary conditions in `x`.

### Closure highlight (important)

The dust “pressure” is implemented via the closure

\[
c_d^2 = D/t_s,
\]

so the effective dust pressure inherits the same density dependence as the turbulent diffusivity \(D(\Sigma)\). This is a **closure choice** intended to represent unresolved velocity dispersion associated with turbulent mixing; it is not meant as a fundamental equation of state. The repo includes both pure power-law closures and a piecewise saturating closure used to eliminate nonlinear blow-up in 1D.

---

## Repo structure (high level)

diffinst/
  solvers/
    native_*                # pseudo-spectral IMEX solvers (linear + nonlinear)
    dedalus_backend.py      # optional Dedalus implementation
  vis/
    linear.py               # visualization scripts for linear experiments
    nonlinear.py            # visualization scripts for nonlinear experiments
  linear_ops.py             # EVP / dispersion relation utilities
  analysis_api.py           # run loading, metrics, mode amplitudes
  config.py                 # Config dataclass + YAML merge / validation
  io.py                     # StreamWriter + run directory layout
  io_utils.py
  fields.py
  grid.py
  runtime.py
  operators.py
  nonlinear_terms.py

defaults.yaml               # baseline parameters
experiments/*.yaml          # paper-style configs
scripts/                    # runnable entry points
notebooks/                  # jupyter notebooks
figures/                    # figure outputs
runs/                       # output (ignored by git)

---

## Output format (reproducibility)

Each run creates a self-contained directory under `runs/`:

- `run.json` — immutable manifest (resolved config + numerics: `Nx`, `Lx`, backend, `dt`, `tstop`, seed, etc.)
- `metrics.jsonl` — streaming time-series diagnostics (e.g. mass, selected mode amplitudes)
- `checkpoints/chk_****.npz` — snapshots containing `t, x, Sigma, vx, vy, uy`

The analysis API can load and compare runs independent of resolution/backend.

---

## Overview of capabilities

### Linear theory (EVP)

- 4×4 eigenvalue problem for the dispersion relation  
- CLI sweeps over physical wavenumber `k` to obtain growth rates `γ(k)`  
- EVP eigenvectors used to generate matched initial conditions for TD solvers  

### Linear time-domain (native)

- IMEX pseudo-spectral integrator for the linearized system  
- Consistency checks: EVP growth rates vs TD amplitude evolution  

### Nonlinear time-domain (native)

- 1D pseudo-spectral IMEX integrator for the full nonlinear equations  
  - Explicit: advection + variable-coefficient fluxes + drag + nonlinear dust-pressure terms  
  - Implicit: constant-coefficient Laplacian viscosity terms (dust/gas)  
- Seeding options:
  - `seed_mode="eigen"` (EVP-based IC at chosen `k`)
  - `seed_mode="cos"`
  - `seed_mode="noise"`
  - `--init-from path.npz` (load `Sigma, vx, vy, uy`)
- Closure relations:
  - Power-law closures `D, ν ∝ Σ^β`  
  - Piecewise saturating closure (negative slope active only over a finite density interval)  

### Dedalus backend (optional cross-check)

- Same equations implemented in Dedalus
- Supports eigen, cosine, noise and external IC seeding
- Currently does not support piecewise closure
- Writes checkpoints compatible with the same analysis tools as the native solver


---

## Quickstart

### Install (recommended: venv)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
pip install numpy scipy matplotlib pyyaml pytest
```

### Optional: Dedalus

Dedalus is optional and used for cross-checks only. If you want it, set up a dedalus3 environment per Dedalus documentation, then install this repo inside that env.

## Running experiments via Scripts

Exact script names may vary slightly; the pattern is: pick an experiments/*.yaml and override numerics (Nx, dt, tstop, backend, seed) on the CLI.

### 1) EVP sweep: growth rate vs wavenumber

```bash
python scripts/run_evp.py \
  --config experiments/diffinst.yaml \
  --kmin 10 --kmax 400 --nk 200
```

Outputs growth rates γ(k) and (optionally) eigenvector information.

### 2) Linear TD: validate EVP growth at a single k

```bash
python scripts/run_linear.py \
  --config experiments/diffinst.yaml \
  --k 100 --Nx 128 --dt 1e-3 --tstop 5.0 \
  --seed eigen
```

Compare the measured mode amplitude growth to EVP’s γ(k).

### 3) Make initial condition

```bash
python -m scripts.make_ic_eigen \
  --config experiments/unstable_baseline.yaml \
  --k 100.0 \
  --amp 1e-6 \
  --Nx 256 \
  --exact-fit-harm 2
```


### 3) Nonlinear eigenmode run

```bash
python scripts/run_nonlinear.py \
  --config experiments/diffinst.yaml \
  --k 100 --Nx 128 --dt 1e-3 --tstop 5.0 \
  --seed eigen --amp 0.1
```


### 4) Noise run: mode selection

```bash
python scripts/run_nonlinear.py \
  --config experiments/diffinst.yaml \
  --seed noise --Nx 512 --dt 5e-4 --tstop 5.0 \
  --noise-amp 1e-2
```

### 5) Piecewise saturating closure

```bash
python scripts/run_nonlinear.py \
  --config experiments/diffinst.yaml \
  --seed noise --closure piecewise \
  --Sigma_sat_over_Sigma0 1.5
```


## Analysis utilities

The recommended way to compare runs is via the lightweight analysis API:
- Load manifests, metrics, checkpoints
- Compute resolution-independent Fourier mode amplitudes at a target physical wavenumber
- Compare native vs Dedalus runs for matched ICs

Typical workflows:
- EVP γ(k) vs linear TD amplitude growth
- Noise runs: dominant k(t) vs fastest-growing EVP mode
- Nonlinear: track max Σ(t), spike formation, and (with piecewise closure) saturation behavior

## Notes / limitations (expected)
- With pure power-law closures and sufficiently negative diffusion slope, the 1D nonlinear system can exhibit finite-time steepening / collapse into narrow spikes. This is a model behavior, not primarily a numerical failure.
- Dust viscosity regularizes the linear small-scale behavior but does not generically provide nonlinear regularization against the dominant gradient-amplifying term in 1D.
- The Dedalus backend is for cross-checks; the native solver is the primary “fast iteration” workhorse.

## Citation

If you use this code, please cite the associated paper and (optionally) the repository. A Zenodo DOI can be added once archived.



## Still to discuss

### Core infrastructure

- ✅ **Repo scaffolded**: `Config` dataclass → grid handling → streaming I/O (`StreamWriter`) → runners in `scripts/`
- ✅ **Config system**: `defaults.yaml` + `experiments/*.yaml`, merged into an immutable `Config`
- ✅ **Streaming I/O**:
  - `run.json` manifest per run (stores config path, Nx, Lx, backend, dt, tstop,…)
  - `metrics.jsonl` with time-series diagnostics (e.g. `mode1_amp`, `mass`)
  - `checkpoints/chk_****.npz` with `t, x, Sigma, vx, vy, uy`
- ✅ **Tests**: basic unit tests pass (`pytest`)


### Nonlinear time-domain

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
