# diffinst/operators.py
from __future__ import annotations
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq



class PSOperators:

    """
    Real-to-complex 1D spectral operators with 2/3 dealias and
    diagonal Laplacian in k-space. Periodic domain.
    """

    def __init__(self, nx: int, Lx: float, dealias: bool = True):
        self.nx = int(nx)
        self.Lx = float(Lx)
        self.dk = 2.0 * np.pi / Lx
        self.k = rfftfreq(self.nx, d=Lx / nx) * 2.0 * np.pi  # shape (nx//2+1,)
        self.dealias = dealias
        if dealias:
            kmax = self.k.max()
            cutoff = (2.0/3.0) * kmax
            self.mask = (np.abs(self.k) <= cutoff).astype(np.float64)
        else:
            self.mask = np.ones_like(self.k)

        # preallocate work arrays if desired
        self._ik = 1j * self.k
        self._k2 = self.k**2

    # transforms
    def fft(self, f_x: np.ndarray) -> np.ndarray:
        return rfft(f_x)

    def ifft(self, f_k: np.ndarray) -> np.ndarray:
        return irfft(f_k, n=self.nx)

    def dealias_k(self, f_k: np.ndarray) -> np.ndarray:
        if self.dealias:
            return f_k * self.mask
        return f_k

    # derivatives (apply in spectral space)
    def dx(self, f_x: np.ndarray) -> np.ndarray:
        fk = self.fft(f_x)
        fk *= self._ik
        fk = self.dealias_k(fk)
        return self.ifft(fk)

    def lap(self, f_x: np.ndarray) -> np.ndarray:
        fk = self.fft(f_x)
        fk *= -self._k2
        fk = self.dealias_k(fk)
        return self.ifft(fk)

    # linear implicit operator helpers: (I - a*dt*L)^{-1} application
    def invert_I_minus_a_dt_nu_lap(self, f_x: np.ndarray, a_dt_nu: float) -> np.ndarray:
        fk = self.fft(f_x)
        denom = 1.0 + a_dt_nu * self._k2
        fk /= denom
        fk = self.dealias_k(fk)
        return self.ifft(fk)

