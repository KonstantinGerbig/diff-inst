
---

# Diffusive Instabilities in Dusty Disks — numerical companion repo

Fast, reproducible **1D axisymmetric** experiments for the diffusive instability in dusty disks ([Gerbig et al. 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad1114), Gerbig & Lin, submitted), including an **incompressible, viscous gas** that responds **azimuthally** and couples to the dust via **drag backreaction**. 

This repo supports the workflow used in the paper:

**EVP (dispersion relation) → linear time-domain validation → nonlinear evolution / breakdown → saturating closure experiments**

Two interchangeable backends:

- **Native**: 1D pseudo-spectral **IMEX** solver (NumPy + FFT) for speed and clarity  
- **Dedalus** (optional): independent implementation for cross-checks  

All runs are configured via **YAML** (+ a small set of CLI overrides). Every run writes a manifest + streaming diagnostics + checkpoints for analysis and plotting.

---

## What model is being solved?

This code integrates the paper’s 1D axisymmetric dust–gas system (dust: $\Sigma, v_x, v_y$; gas: $u_y$), with periodic boundary conditions in $x$.

### Closure highlight

The dust “pressure” is implemented via the closure 
$$c_\text{s}^2 = \frac{D}{t_\text{s}}$$
so the effective dust pressure inherits the same density dependence as the turbulent diffusivity 
$$D(\Sigma) = \left(\frac{\Sigma}{\Sigma_0}\right)^{\beta_\text{diff}}.$$
This is a **closure choice** intended to represent unresolved velocity dispersion associated with turbulent mixing; it is not meant as a fundamental equation of state. Roughly speaking, for small stopping times and for $\beta_\text{diff} < -2$, this pressure closure leads to diffusive instability. The repo includes both pure power-law closures and a piecewise saturating closure used to eliminate nonlinear blow-up in 1D.

---


## Repo structure

```bash
src/diffinst/
  solvers/
    native_*                # pseudo-spectral IMEX solvers (linear + nonlinear)
    dedalus_backend.py      # optional Dedalus implementation
  vis/
    linear.py               # visualization scripts for linear experiments
    nonlinear.py            # visualization scripts for nonlinear experiments
  linear_ops.py             # EVP / dispersion relation utilities
  analysis_api.py           # run loading, metrics, mode amplitudes
  config.py                 # Config dataclass + YAML merge / validation
  io_utils.py               # StreamWriter + run directory layout
  fields.py                 # defines fields sigma, vx, vy, uy
  grid.py                   # Fourier grid
  runtime.py                # Run handling
  operators.py              # Pseudospectral operators (fft, ifft, derivatives etc)
  nonlinear_terms.py        # equations that are solved + closures

defaults.yaml               # baseline parameters
experiments/*.yaml          # contains configs files
scripts/                    # runnable entry points
tests/                      # includes a few tests (WIP)
notebooks/                  # jupyter notebooks
figures/                    # figure outputs
runs/                       # output (ignored by git)
```

---

## Output format

Each run creates a self-contained directory under `runs/`:

- `run.json` — immutable manifest (resolved config + numerics: `Nx`, `Lx`, backend, `dt`, `tstop`, seed, etc.)
- `metrics.jsonl` — streaming time-series diagnostics (e.g. mass, selected mode amplitudes)
- `checkpoints/chk_****.npz` — snapshots containing `t, x, Sigma, vx, vy, uy`

The analysis API can load and compare runs independent of resolution/backend.

---

## Overview of capabilities

### Linear theory (EVP)

- 4×4 eigenvalue problem for the dispersion relation  
- CLI sweeps over physical wavenumber `k` to obtain growth rates `gamma(k)`  
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
  - Power-law closures $D, \nu \propto \Sigma^\beta$  
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

The code base includes serveral useful scripts for solving eigen value problem and running time domain setups. The general pattern is: pick an experiments/*.yaml and override numerics (Nx, dt, tstop, backend, seed) on the CLI. Scripts can be run from the command line (see examples below) or from notebooks (see the jupyter notebooks in the notebook folder).

### 1) EVP sweep: growth rate vs wavenumber

```bash
python scripts/run_evp.py \
  --config experiments/diffinst.yaml \
  --kmin 10 --kmax 400 --nk 200
```

Outputs growth rates $\gamma(k)$ and (optionally) eigenvector information.

### 2) Make eigenvalue initial condition

```bash
python -m scripts.make_ic_eigen \
  --config experiments/diffinst.yaml \
  --k 100.0 \
  --amp 1e-6 \
  --Nx 256 \
  --exact-fit-harm 2 \
  --out runs/ic_eigen.npz
```

### 3) Linear TD: validate EVP growth at a single k

```bash
python scripts/run_linear.py \
  --config experiments/diffinst.yaml \
  --k 100 --Nx 256 --dt 1e-3 --tstop 5.0 \
  --init-from runs/ic_eigen.npz
```

Compare the measured mode amplitude growth to EVP’s $\gamma(k)$.


### 3) Nonlinear eigenmode run

```bash
python scripts/run_nonlinear.py \
  --config experiments/diffinst.yaml \
  --k 100 --Nx 256 --dt 1e-3 --tstop 5.0 \
  --init-from runs/ic_eigen.npz
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
  --config experiments/diffinst_noise_piecewise.yaml \
  --seed noise --closure piecewise \
  --Sigma_sat_over_Sigma0 1.5
```

---

## Analysis utilities

The recommended way to compare runs is via the lightweight **Analysis API** (`diffinst/analysis_api.py`):
- Load manifests, metrics, checkpoints
  - `load_manifest`, `load_config_from_run`, `list_checkpoints`, `load_metrics`
  - `load_linear_run`, `load_linear_Sigma_series`
  - `load_nonlinear_run`, `load_nonlinear_Sigma_series`
- Compute resolution-independent Fourier mode amplitudes at a target physical wavenumber
  - `nearest_k_index(Lx, Nx, k_phys)`
  - `amplitude_at_k_from_sigma` (returns physical cosine amplitude Aₖ)
  - `load_linear_amplitude`, `load_nonlinear_amplitude`
- IC helpers: `save_ic_npz`, `load_ic_npz`
- EVP helper: `evp_gamma(cfg, k_phys)`

Example workflows:
- EVP $\gamma(k)$ vs TD amplitude growth
- Noise runs: dominant $k(t)$ vs fastest-growing EVP mode
- Nonlinear: track max $\Sigma(t)$, spike formation, and (with piecewise closure) saturation behavior

---

## Code units and parameters

The code uses dimensionless units where orbital frequency and characteristic scale height are set to $\Omega = 1$ and $H=1$ respectively. The background dust surface density is set to $\Sigma_0 = 1$. All parameters can be edited via yaml configuration files. Physical parameters are
- `D_0`: (unperturbed) diffusion coefficient
- `beta_diff`: diffusion slope ($D = D_0 (\Sigma/\Sigma_0)^{\beta_\mathrm{diff}}$ for the standard closure)
- `nu_0` and `beta_visc`: (unperturbed) dust viscosity coefficient and viscosity slope ($\nu = \nu_0 (\Sigma/\Sigma_0)^{\beta_\mathrm{visc}}$)
- `ts`: dust stopping time
- `q`: shear parameter

The model also includes an incompressible gas:
- `eps`: dust-to-gas ratio
- `nu_g`: constant gas viscosity
- `enable_gas`: toggles inclusion of the gas equation. If `enable_gas: false`, the model assumes drag against a static background. 

The grid is also set up via the yaml file:
- `Nx`: number of grid cells
- `Lx`: domain size in code units

For eigenvalue tests, it is useful that the domain size is an exact multiple of the target wavelength. To avoid having to calculate the correct Lx by hand, this can be asserted in the yaml:
- `exact_fit: enable`: flag to force an exact fit to the given wavenumber, overrides any set `Lx`
- `exact_fit: K_target`: target wavenumber. Note, that this only sets up the grid. It does not set the initial condition.  
- `exact_fit: harmonics`: number of wavelengths to fit into the domain

Run control parameters:
- `stop_time`
- `max_dt`: max time step
- `min_dt`: min time step
- `save_stride`
- `log_stride`
- `solver: backend`: options include `native` and `dedalus`

Note that grid and runtime parameters are ignored when the code is in evp mode. 

---

## Pytests



---

## Additional notes and limitations

- **Eigenmode IC generator** (helper script, called from notebook / CLI, see above):
  - Uses a chosen experiment config + `k_phys` + `Nx` override to:
    - Solve EVP
    - Build a real eigenmode in x
    - Normalize it to a desired physical `amp_phys` in Σ
    - Save to `runs/ic_k*_eigen.npz` with `Sigma, vx, vy, uy` and metadata
  - This IC can be reused for any run
- **Grid override**:
  - `Nx` can be overridden at run time via CLI (used to build a compatible `Grid1D` and operators)
  - Manifests / loaders respect the effective `Nx` used in a given run
- With pure power-law closures and sufficiently negative diffusion slope, the 1D nonlinear system can exhibit finite-time steepening / collapse into narrow spikes. This is a model behavior, not primarily a numerical failure. This can be prevented by modifying the closure relation, such as the implemented piecwise closure.
- Dust viscosity regularizes the linear small-scale behavior but does not generically provide nonlinear regularization against the dominant gradient-amplifying term in 1D.
- The Dedalus backend is for cross-checks; the native solver is the primary “fast iteration” workhorse.

---

## Citation

If you use this code, please cite the associated paper and (optionally) the repository.