import numpy as np
from diffinst import Config
from diffinst.linear_ops import evp_solve_at_k

def test_evp_stable_defaults():
    cfg = Config.from_yaml("experiments/baseline.yaml", "defaults.yaml")  # stable
    ks = np.logspace(0, 2, 5)
    for k in ks:
        w, _ = evp_solve_at_k(cfg, float(k))
        assert np.all(np.isfinite(w))
        # dominant growth should be non-positive for stable defaults
        assert w[0].real <= 1e-10