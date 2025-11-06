import numpy as np
from pathlib import Path
from dataclasses import replace

from diffinst import Config
from diffinst.runtime import run_nonlinear_native
from diffinst.linear_ops import evp_solve_at_k



def _fft_mode_amplitude(sig: np.ndarray, k_index: int) -> float:
    ak = np.fft.rfft(sig)[k_index]
    return float(abs(ak))


def test_nonlinear_pure_diffusion_mode_decay(tmp_path: Path):
    base = Config.from_yaml("experiments/baseline.yaml", "defaults.yaml")

    # Create a modified copy (frozen dataclass -> use replace).
    # Only pass fields that exist in Config.__init__.
    cfg = replace(
        base,
        Nx=256,
        Lx=2.0 * np.pi,
        Omega=0.0,   # no Coriolis/shear (shear coupling vanishes with Omega=0)
        D_0=1e-3,    # constant diffusion
        nu_0=0.0,    # no viscosity
    )

    outdir = tmp_path / "nl"
    outdir.mkdir(parents=True, exist_ok=True)

    init_k = 3
    amp_phys = 1e-6      # very small so pressure effects stay negligible
    stop_time = 0.05         # short run; diffusion dominates
    dt = 1e-3
    save_stride = 20

    info = run_nonlinear_native(
        cfg=cfg,
        outdir=outdir,
        stop_time=stop_time,
        dt=dt,
        save_stride=save_stride,
        init_k=init_k,
        amp=amp_phys,     # physical amplitude of (Sigma - mean)
        seed=0,
    )
    assert info["t_final"] > 0.0

    # Load first and last checkpoints
    chk_dir = outdir / "checkpoints"
    files = sorted(chk_dir.glob("chk_*.npz"))
    assert len(files) >= 2

    with np.load(files[0]) as Z0:
        x = Z0["x"]
        Sigma0 = Z0["Sigma"]
        t0 = float(Z0["t"])

    with np.load(files[-1]) as Z1:
        Sigma1 = Z1["Sigma"]
        t1 = float(Z1["t"])

    # Mass conservation
    assert abs(Sigma1.mean() - Sigma0.mean()) < 1e-10

    # Mode amplitude decay: A(t) ~ A0 * exp(-D * k^2 * (t1 - t0))
    k_phys = init_k * (2.0 * np.pi / float(cfg.Lx))
    A0 = _fft_mode_amplitude(Sigma0 - Sigma0.mean(), init_k)
    A1 = _fft_mode_amplitude(Sigma1 - Sigma1.mean(), init_k)
    assert A0 > 0 and A1 > 0


    # ... after loading Sigma0, Sigma1 and computing k_phys, A0, A1, t0, t1

    # Predict with EVP (dominant eigenvalue at k_phys for the same cfg)
    w, _ = evp_solve_at_k(cfg, k_phys)
    gamma = float(w[0].real)     # linear growth/decay rate for Σ at this k
    A_pred = A0 * np.exp(gamma * (t1 - t0))

    rel_err = abs(A1 - A_pred) / max(1e-16, abs(A_pred))
    assert rel_err < 0.15