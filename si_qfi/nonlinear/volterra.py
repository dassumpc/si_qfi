"""
si_qfi.nonlinear.volterra
=========================
Volterra series nonlinear model (real-axis mode only).

Implements a third-order Volterra series truncation operating on the full real
RF waveform. Supports three parameterisation options (PRD §5.4):

Option A — Diagonal kernel (memory polynomial on real axis):
    y[n] = Σ_{p=1,2,3,...} Σ_m a_{pm} · x[n-m]^p
    Both odd and even orders present. Coefficients supplied directly or fit
    from P1dB, IP3 (odd-symmetric, h₂=0), and optionally IP2 (even-order).

Option B — Full kernel (h₁ + h₃, h₂=0 for odd-symmetric amplifiers):
    y[n] = Σ_τ h₁[τ]·x[n-τ]
           + Σ_{τ₁,τ₂,τ₃} h₃[τ₁,τ₂,τ₃]·x[n-τ₁]·x[n-τ₂]·x[n-τ₃]
    h₃ is the triangular (symmetric) kernel of shape (M+1, M+1, M+1).
    Note: full 3-D kernel storage is O(M³) — keep M small (≤ 20).

Option C — Describing function parameterisation (default):
    h₁ is supplied from the SI transfer function (set externally by the engine).
    h₃ is a diagonal approximation derived from P1dB and IP3 (odd-symmetric).
    This is the practical default when only P1dB/IP3 measurements are available.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.signal import fftconvolve

from .base import NonlinearNode


class VolterraModel(NonlinearNode):
    """
    Third-order Volterra series model for real RF waveform propagation.

    Parameters
    ----------
    h1 : np.ndarray, float64, shape (L1,)
        First-order (linear) kernel — impulse response. Usually set from the
        SI transfer function by the simulation engine. If None, h1 = delta[0]
        (no linear filtering within the NL block itself; the linear channel is
        applied by the engine before and after).
    option : str
        'diagonal'    — diagonal kernel; requires coefficients.
        'full_kernel' — full h3 tensor; requires h3.
        'describing'  — h1 from SI + h3 derived from p1db_amplitude, ip3_amplitude.
    coefficients : np.ndarray, shape (n_orders, M+1), optional
        For option='diagonal'. Row per order (1,2,3,...), column per memory tap.
    orders : list of int, optional
        Orders included in diagonal kernel. Default [1, 2, 3].
    h3 : np.ndarray, float64, shape (M+1, M+1, M+1), optional
        Full symmetric third-order kernel for option='full_kernel'.
    p1db_amplitude : float, optional
        For option='describing': P1dB input amplitude (same units as waveform).
    ip3_amplitude : float, optional
        For option='describing': IP3 input amplitude.
    small_signal_gain : float
        Small-signal linear gain (used in describing parameterisation).
    memory_depth : int
        Memory depth M (samples) for diagonal / describing options.
    """

    def __init__(
        self,
        h1: Optional[np.ndarray] = None,
        option: str = "describing",
        coefficients: Optional[np.ndarray] = None,
        orders: Optional[list] = None,
        h3: Optional[np.ndarray] = None,
        p1db_amplitude: Optional[float] = None,
        ip3_amplitude: Optional[float] = None,
        small_signal_gain: float = 1.0,
        memory_depth: int = 5,
    ) -> None:
        self._h1 = np.asarray(h1, dtype=float) if h1 is not None else np.array([1.0])
        self._option = option
        self._M = int(memory_depth)
        self._gain = float(small_signal_gain)

        if option == "diagonal":
            if coefficients is None:
                raise ValueError("option='diagonal' requires coefficients.")
            self._coeff = np.asarray(coefficients, dtype=float)
            self._orders = list(orders) if orders is not None else list(range(1, self._coeff.shape[0] + 1))

        elif option == "full_kernel":
            if h3 is None:
                raise ValueError("option='full_kernel' requires h3.")
            self._h3 = np.asarray(h3, dtype=float)
            if self._h3.ndim != 3:
                raise ValueError("h3 must be a 3-D array (M+1, M+1, M+1).")

        elif option == "describing":
            if p1db_amplitude is None or ip3_amplitude is None:
                raise ValueError(
                    "option='describing' requires p1db_amplitude and ip3_amplitude."
                )
            self._p1db = float(p1db_amplitude)
            self._ip3 = float(ip3_amplitude)
            self._coeff, self._orders = self._build_describing_coeff()

        else:
            raise ValueError(f"Unknown option '{option}'. Use 'diagonal', 'full_kernel', or 'describing'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_describing_coeff(self):
        """
        Build diagonal kernel coefficients from P1dB and IP3 using the
        cubic describing function result (PRD §5.1).

        For the real-axis cubic: f(x) = G₀·x + c·x³
            c = -G₀ / A_IP3²          (from IP3 definition)
        Only odd orders (1, 3); even orders assumed zero (odd-symmetric amplifier).
        All memory taps beyond m=0 initialised to zero.
        """
        M = self._M
        coeff = np.zeros((2, M + 1), dtype=float)
        coeff[0, 0] = self._gain                             # k=1, m=0: linear gain
        coeff[1, 0] = -self._gain / (self._ip3 ** 2)        # k=3, m=0: cubic compression
        return coeff, [1, 3]

    def _apply_diagonal(self, x: np.ndarray) -> np.ndarray:
        """Diagonal kernel: y[n] = Σ_p Σ_m a_{pm} · x[n-m]^p"""
        N = len(x)
        out = np.zeros(N, dtype=float)
        for row, p in enumerate(self._orders):
            for m in range(self._M + 1):
                a = self._coeff[row, m]
                if a == 0.0:
                    continue
                if m == 0:
                    x_del = x
                else:
                    x_del = np.concatenate([np.zeros(m), x[:-m]])
                out += a * (x_del ** p)
        return out

    def _apply_full_kernel(self, x: np.ndarray) -> np.ndarray:
        """
        Full third-order Volterra: y = h1*x + triple sum of h3.
        Efficient implementation: for each (τ₁, τ₂), convolve h3[:, τ₁, τ₂] with x,
        then multiply by x[n-τ₁] · x[n-τ₂] and accumulate.
        Still O(N·M²) — keep M small.
        """
        N = len(x)
        M1 = len(self._h1)
        # Linear term
        y = fftconvolve(x, self._h1)[:N]

        # Third-order term
        M = self._h3.shape[0] - 1
        x_pad = np.concatenate([np.zeros(M), x])   # pad for delayed access

        for tau1 in range(M + 1):
            x1 = x_pad[M - tau1: M - tau1 + N]
            for tau2 in range(M + 1):
                x2 = x_pad[M - tau2: M - tau2 + N]
                for tau3 in range(M + 1):
                    h = self._h3[tau1, tau2, tau3]
                    if h == 0.0:
                        continue
                    x3 = x_pad[M - tau3: M - tau3 + N]
                    y += h * x1 * x2 * x3
        return y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_h1(self, h1: np.ndarray) -> None:
        """
        Set the linear kernel from an external source (e.g. SI transfer function).
        Called by the simulation engine when option='describing'.
        """
        self._h1 = np.asarray(h1, dtype=float)

    @property
    def memory_depth(self) -> int:
        return self._M

    @property
    def supports_baseband(self) -> bool:
        return False

    @property
    def supports_real_axis(self) -> bool:
        return True

    def apply_real_axis(self, v: np.ndarray) -> np.ndarray:
        """
        Apply Volterra series to real RF waveform v(t).

        Dispatches to the appropriate internal implementation based on option.
        """
        v = np.asarray(v, dtype=float)
        if self._option in ("diagonal", "describing"):
            return self._apply_diagonal(v)
        elif self._option == "full_kernel":
            return self._apply_full_kernel(v)
        else:
            raise RuntimeError(f"Unknown option: {self._option}")

    def __repr__(self) -> str:
        return (
            f"VolterraModel(option='{self._option}', M={self._M}, "
            f"gain={self._gain:.3f})"
        )
