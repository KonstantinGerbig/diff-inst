import json
import numpy as np
import pytest
from pathlib import Path

from diffinst import Config
from diffinst.runtime import run_linear_native
from diffinst.linear_ops import evp_solve_at_k


def _first_and_last_with_keys(metrics_path: Path, keys):
    lines = metrics_path.read_text().strip().splitlines()
    rows = [json.loads(ln) for ln in lines]
    filt = [r for r in rows if all(k in r for k in keys)]
    assert len(filt) >= 2, "Not enough metrics rows with required keys"
    return filt[0], filt[-1]


def test_linear_growth_matches_evp(tmp_path: Path):
    cfg = Config.from_yaml("experiments/unstable_baseline.yaml", "defaults.yaml")
    outdir = tmp_path / "lin"
    outdir.mkdir(parents=True, exist_ok=True)

    info = run_linear_native(
        cfg=cfg,
        outdir=outdir,
        stop_time=2.0,
        dt=5e-3,
        save_stride=10,
        k_target=50.0,
        amp=1e-6,             # tiny to stay linear
        seed=0,
        amp_is_physical=True,
        amp_metric="max",
    )
    assert info["t_final"] == pytest.approx(2.0)

    first, last = _first_and_last_with_keys(outdir / "metrics.jsonl", ["k_track", "amp_S_k", "t"])
    t0 = float(first["t"]);  a0 = float(first["amp_S_k"])
    tf = float(last["t"]);   af = float(last["amp_S_k"])
    assert a0 > 0 and af > 0 and tf > t0

    k_track = float(first["k_track"])
    w, _ = evp_solve_at_k(cfg, k_track)
    gamma = float(w[0].real)

    gamma_num = (np.log(af) - np.log(a0)) / (tf - t0)
    rel_err = abs(gamma_num - gamma) / max(1e-12, abs(gamma))
    assert rel_err < 0.1