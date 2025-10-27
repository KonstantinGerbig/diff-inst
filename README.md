# Diffusive Instability (dust–gas) — clean numerical stack

One goal: a fast, clear, reproducible 1D stack for the diffusive instability model
(axes: EVP, linear time-domain, nonlinear saturation), with switchable solvers:
our native 1D pseudo-spectral IMEX **and** a Dedalus backend.

## status

- ✅ Repo scaffolded: config → grid → streaming I/O → runners
- ✅ Tests pass (`pytest`)
- ✅ Dry-run works (writes manifest, metrics, one checkpoint)
- ✅ EVP (direct 4×4) implemented with CLI sweep
- 🚧 Linear time-domain harness (IMEX) — next step
- 🚧 Nonlinear core + convergence metrics — after linear harness
- 🚧 Paper figure scripts — after linear/nonlinear are in

## install (recommended: venv)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .  # editable install
pip install scipy matplotlib pytest