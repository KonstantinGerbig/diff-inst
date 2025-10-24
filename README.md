# diff-instability (1D dust–gas) — clean stack

This repo starts with a dry scaffold: configs, grid, I/O, and a no-op runner that writes a single checkpoint. No physics yet. The goal is to ensure the plumbing is correct before we add equations.

## Quick start
```bash
pytest -q
python scripts/run_linear.py --config experiments/baseline.yaml --outdir runs/linear --dry-run