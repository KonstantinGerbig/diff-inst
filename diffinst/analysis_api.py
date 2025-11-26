# diffinst/analysis_api.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import Tuple, List, Dict, Iterable, Optional, Callable
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
    """
    Resolution-independent amplitude at mode k_idx.

    Returns the physical cosine amplitude A_k such that
    Sigma(x) ~ ... + A_k cos(kx + phi), independent of Nx.
    """
    s = Sigma - Sigma.mean()
    ak = np.fft.rfft(s)
    N = s.size
    if k_idx == 0:
        # DC mode: no factor 2
        return float(np.abs(ak[0]) / N)
    else:
        # k>0: A = 2|ak[k]|/N
        return float(2.0 * np.abs(ak[k_idx]) / N)

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
            # s_x is the perturbation; reconstruct full Sigma if you prefer:
            Sigma = Sigma0 + s_x
            A.append(amplitude_at_k_from_sigma(Sigma, k_idx))
    return np.array(T), np.array(A)

# ---------- IC I/O helpers ----------

def load_ic_npz(path: Path | str):
    Z = np.load(path, allow_pickle=True)
    out = {}

    out["Sigma"] = np.asarray(Z["Sigma"])
    out["vx"]    = np.asarray(Z["vx"])
    out["vy"]    = np.asarray(Z["vy"])

    if "uy" in Z.files:
        out["uy"] = np.asarray(Z["uy"])
    else:
        # Dust-only IC → linear solver still wants 4 components
        out["uy"] = np.zeros_like(out["Sigma"])

    if "meta" in Z.files:
        meta = Z["meta"]
        out["meta"] = meta.item() if isinstance(meta, np.ndarray) else meta

    return out

def save_ic_npz(
    path: Path | str,
    Sigma: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    uy: Optional[np.ndarray] = None,
    meta: Optional[dict] = None,
) -> None:
    """
    Save an initial condition to a compressed .npz file.

    For dust-only setups, uy may be passed as None; we then store
    an array of zeros with the same shape as Sigma.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if uy is None:
        uy = np.zeros_like(Sigma)

    if meta is None:
        np.savez_compressed(path, Sigma=Sigma, vx=vx, vy=vy, uy=uy)
    else:
        np.savez_compressed(
            path,
            Sigma=Sigma,
            vx=vx,
            vy=vy,
            uy=uy,
            meta=np.array(meta, dtype=object),
        )

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

# ---------- safe loader helpers for amplitudes ----------

def safe_load_amplitude(loader_func, run_dir: Path | str | None, k_phys: float):
    """
    Wrapper around load_linear_amplitude / load_nonlinear_amplitude that
    returns None if the run directory is missing or the loader fails.
    """
    if run_dir is None:
        return None
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    try:
        return loader_func(run_dir, k_phys)
    except Exception as e:
        print(f"[safe_load_amplitude] failed for {run_dir}: {e}")
        return None


# ---------- mode-series loaders for hodographs ----------

def _k_index_for_run(run_dir: Path | str, k_phys: float) -> int:
    """
    Helper: given a run directory and physical k, return the rfft-bin index.
    Works for both linear and nonlinear runs (uses Nx, Lx from manifest).
    """
    run_dir = Path(run_dir)
    man = load_manifest(run_dir)
    Nx = int(man["Nx"]); Lx = float(man["Lx"])
    return nearest_k_index(Lx, Nx, k_phys)


def load_mode_series_linear(run_dir: Path | str, k_phys: float):
    """
    For a linear TD run, return:
        T, vx_k(T), vy_k(T)
    where vx_k, vy_k are the complex Fourier coefficients of mode k_phys.
    """
    run_dir = Path(run_dir)
    Nx, Lx, files, _, _, _, _ = load_linear_run(run_dir)
    k_idx = nearest_k_index(Lx, Nx, k_phys)

    T_list, vxk_list, vyk_list = [], [], []
    for fn in files:
        with np.load(fn) as Z:
            T_list.append(float(Z["t"]))
            Xhat = Z["Xhat"]  # shape (Nk, 4)
            vx_hat = Xhat[:, 1]
            vy_hat = Xhat[:, 2]
            vxk_list.append(vx_hat[k_idx])
            vyk_list.append(vy_hat[k_idx])

    return np.array(T_list), np.array(vxk_list), np.array(vyk_list)


def load_mode_series_nonlinear(run_dir: Path | str, k_phys: float):
    """
    For a nonlinear TD run (native or Dedalus), return:
        T, vx_k(T), vy_k(T)
    where vx_k, vy_k are the complex Fourier coefficients of mode k_phys
    computed from x-space fields.
    """
    run_dir = Path(run_dir)
    Nx, Lx, files, _ = load_nonlinear_run(run_dir)
    k_idx = nearest_k_index(Lx, Nx, k_phys)

    T_list, vxk_list, vyk_list = [], [], []
    for fn in files:
        with np.load(fn) as Z:
            T_list.append(float(Z["t"]))
            vx = np.asarray(Z["vx"])
            vy = np.asarray(Z["vy"])
            vx_hat = np.fft.rfft(vx)
            vy_hat = np.fft.rfft(vy)
            vxk_list.append(vx_hat[k_idx])
            vyk_list.append(vy_hat[k_idx])

    return np.array(T_list), np.array(vxk_list), np.array(vyk_list)