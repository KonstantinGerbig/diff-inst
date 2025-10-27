from diffinst import Config

def test_config_loads():
    cfg = Config.from_yaml("experiments/baseline.yaml", "defaults.yaml")
    assert cfg.Nx > 0
    assert cfg.max_dt >= cfg.min_dt > 0.0