from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

@dataclass(frozen=True)
class Config:
    Omega: float = 1.0
    q: float = 1.5
    sig_0: float = 1.0
    tstop: float = 0.1
    D_0: float = 1e-5
    nu_0: float = 1e-5
    beta_diff: float = 0
    beta_visc: float = 0
    nu_g: float = 1e-3
    eps: float = 1.0
    Lx: float = 0.2
    Nx: int = 128
    exact_fit: Dict[str, Any] = field(default_factory=lambda: {"enable": False, "K_target": 100.0, "harmonics": 2})
    stop_time: float = 40.0
    max_dt: float = 1e-2
    min_dt: float = 5e-4
    save_stride: int = 50
    log_stride: int = 50
    enable_gas: bool = True
    enable_dedalus_evp: bool = True
    solver: Dict[str, Any] = field(default_factory=lambda: {"backend": "native"})
    source_file: str = ""

    @staticmethod
    def from_yaml(path: str | Path, defaults_path: str | Path = "defaults.yaml") -> "Config":
        path = Path(path)
        defaults_path = Path(defaults_path)
        with defaults_path.open("r") as f:
            defaults = yaml.safe_load(f) or {}
        with path.open("r") as f:
            user = yaml.safe_load(f) or {}
        merged = {**defaults, **user}
        # nested dicts
        for key in ("exact_fit", "solver"):
            if key in defaults:
                merged[key] = {**defaults[key], **(user.get(key, {}) or {})}
        cfg = Config(**merged, source_file=str(path))
        _validate(cfg)
        return cfg

def _validate(cfg: Config) -> None:
    assert cfg.Nx > 0 and int(cfg.Nx) == cfg.Nx
    assert cfg.Lx > 0.0
    assert cfg.max_dt >= cfg.min_dt > 0.0
    for val in [cfg.Omega, cfg.sig_0, cfg.tstop, cfg.D_0, cfg.nu_0, cfg.nu_g, cfg.eps]:
        assert isinstance(val, (int, float))
    assert cfg.solver["backend"] in ("native", "dedalus")