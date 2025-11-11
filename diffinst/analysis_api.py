# diffinst/analysis_api.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import Tuple, List, Dict, Iterable, Optional
from . import Config
from .linear_ops import evp_solve_at_k

# ---------- manifests / basics ----------

def load_manifest(run_dir: Path | str) -> dict:
    run_dir = Path(run_dir)
    return json.loads((run_dir / "run.json").read_text())

def load_config_from_run(run_dir: Path | str) -> Config:
    cfg_path = load_manifest(run_dir)["config"]
    return Config.from_yaml(cfg_path)

def list_checkpoints(run_dir: Path | str) -> List[Path]:
    run_dir = Path(run_dir)
    return sorted((run_dir / "checkpoints").glob("chk_*.npz"))

def load_metrics(run_dir: Path | str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return times and dict of metric arrays (columns auto-discovered)."""
    run_dir = Path(run_dir)
    mfile = run_dir / "metrics.jsonl"
    if not mfile.exists():
        return np.array([]), {}
    cols: Dict[str, List[float]] = {}
    t = []
    with mfile.open() as f:
        for line in f:
            row = json.loads(line)
            t.append(float(row.get("t", 0.0)))
            for k, v in row.items():
                if k == "t":
                    continue
                cols.setdefault(k, []).append(float(v) if isinstance(v, (int, float)) else np.nan)
    t_arr = np.array(t, dtype=float)
    cols_arr = {k: np.array(v, dtype=float) for k, v in cols.items()}
    return t_arr, cols_arr

# ---------- linear run loaders ----------

def load_linear_run(run_dir: Path | str):
    """Return Nx, Lx, checkpoint files, k-array, t_metrics, amp_S_k if present, manifest."""
    man = load_manifest(run_dir)
    Nx = int(man["Nx"]); Lx = float(man["Lx"])
    files = list_checkpoints(run_dir)
    k = None
    if files:
        with np.load(files[0]) as Z0:
            k = np.array(Z0["k"])
    t_m, cols = load_metrics(run_dir)
    a_track = cols.get("amp_S_k", None)
    return Nx, Lx, files, k, t_m, a_track, man

def reconstruct_Sigma_from_linear_checkpoint(fn: Path | str, Nx: int, Sigma0: float) -> Tuple[float, np.ndarray]:
    """Return (t, Sigma(x))."""
    with np.load(fn) as Z:
        t = float(Z["t"])
        Xhat = Z["Xhat"]  # shape (Nk, 4) -> [S_hat, vx_hat, vy_hat, uy_hat]
        S_hat = Xhat[:, 0]
    s_x = np.fft.irfft(S_hat, n=Nx)       # perturbation
    Sigma = Sigma0 + s_x
    return t, Sigma

def load_linear_Sigma_series(files: Iterable[Path], Nx: int, Sigma0: float) -> Tuple[np.ndarray, np.ndarray]:
    T, Sig = [], []
    for fn in files:
        t, S = reconstruct_Sigma_from_linear_checkpoint(fn, Nx, Sigma0)
        T.append(t); Sig.append(S)
    return np.array(T), np.array(Sig)

# ---------- nonlinear run loaders ----------

def load_nonlinear_run(run_dir: Path | str) -> Tuple[int, float, List[Path], dict]:
    """Return Nx, Lx, files, manifest."""
    man = load_manifest(run_dir)
    Nx = int(man["Nx"]); Lx = float(man["Lx"])
    files = list_checkpoints(run_dir)
    return Nx, Lx, files, man

def load_nonlinear_Sigma_series(files: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    T, Sig = [], []
    for fn in files:
        with np.load(fn) as Z:
            T.append(float(Z["t"]))
            Sig.append(np.asarray(Z["Sigma"]))
    return np.array(T), np.array(Sig)

# ---------- k-utilities ----------

def nearest_k_index(Lx: float, Nx: int, k_phys: float) -> int:
    """Index into rfft bins for given physical wavenumber."""
    ks = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=Lx / Nx)
    return int(np.argmin(np.abs(ks - float(k_phys))))

def amplitude_at_k_from_sigma(Sigma: np.ndarray, k_idx: int) -> float:
    """Return |FFT(Sigma - mean)[k_idx]|."""
    return float(np.abs(np.fft.rfft(Sigma - Sigma.mean())[k_idx]))

def amplitude_series_from_sigma(T: np.ndarray, Sigma_series: np.ndarray, k_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    A = np.array([amplitude_at_k_from_sigma(S, k_idx) for S in Sigma_series])
    return T, A

def amplitude_series_from_linear(files: Iterable[Path], Nx: int, Sigma0: float, k_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    T, A = [], []
    for fn in files:
        with np.load(fn) as Z:
            T.append(float(Z["t"]))
            S_hat = Z["Xhat"][:, 0]
            s_x = np.fft.irfft(S_hat, n=Nx)
            A.append(float(np.abs(np.fft.rfft(s_x)[k_idx])))
    return np.array(T), np.array(A)

# ---------- IC I/O helpers ----------

def load_ic_npz(path: Path | str) -> Dict[str, np.ndarray]:
    """Expected keys: Sigma, vx, vy, uy; optional meta."""
    Z = np.load(path)
    out = {k: np.asarray(Z[k]) for k in ("Sigma", "vx", "vy", "uy")}
    if "meta" in Z.files:
        out["meta"] = Z["meta"].item() if Z["meta"].dtype == object else Z["meta"]
    return out

def save_ic_npz(path: Path | str, Sigma: np.ndarray, vx: np.ndarray, vy: np.ndarray, uy: np.ndarray, meta: Optional[dict]=None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if meta is None:
        np.savez_compressed(path, Sigma=Sigma, vx=vx, vy=vy, uy=uy)
    else:
        np.savez_compressed(path, Sigma=Sigma, vx=vx, vy=vy, uy=uy, meta=np.array(meta, dtype=object))

# ---------- EVP convenience ----------

def evp_gamma(cfg: Config, k_phys: float) -> float:
    w, _ = evp_solve_at_k(cfg, float(k_phys))
    return float(w[0].real)

# ---------- one-liners for notebooks ----------

def load_linear_amplitude(run_dir: Path | str, k_phys: float) -> Tuple[np.ndarray, np.ndarray]:
    Nx, Lx, files, _, _, _, _ = load_linear_run(run_dir)
    cfg = load_config_from_run(run_dir)
    S0 = cfg.sig_0
    k_idx = nearest_k_index(Lx, Nx, k_phys)
    return amplitude_series_from_linear(files, Nx, S0, k_idx)

def load_nonlinear_amplitude(run_dir: Path | str, k_phys: float) -> Tuple[np.ndarray, np.ndarray]:
    Nx, Lx, files, _ = load_nonlinear_run(run_dir)
    T, Sigma = load_nonlinear_Sigma_series(files)
    k_idx = nearest_k_index(Lx, Nx, k_phys)
    return amplitude_series_from_sigma(T, Sigma, k_idx)