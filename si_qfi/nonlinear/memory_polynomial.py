"""
si_qfi.nonlinear.memory_polynomial
===================================
Memory polynomial model (complex baseband mode only).

Extends memoryless AM-AM to finite impulse-response memory depth M.
Valid when reflection delay τ is comparable to pulse width but harmonic
suppression still holds (i.e. narrowband assumption is still satisfied).

Model
-----
    ũ_out(t) = Σ_{k∈{1,3,5}} Σ_{m=0}^{M} a_{km} · ũ(t-mT) · |ũ(t-mT)|^(k-1)

where:
    k   = nonlinear order (odd only — narrowband, bandpass-filtered)
    m   = memory tap index (0 = current sample, M = oldest sample)
    T   = sample interval
    a_{km} = complex coefficient for order k, tap m

At M=0 (memoryless) this reduces to a standard odd-polynomial AM-AM with
complex coefficients, which is the complex baseband equivalent of the
real-axis Volterra diagonal kernel.

Coefficient identification
--------------------------
Option A — Direct supply: pass a coefficient matrix directly.
Option B — From P1dB / IP3: SI-QFI sets the k=1 coefficients from small-signal
           gain and the k=3 coefficients from the cubic describing-function result
           (PRD §5.1). Coefficients for m>0 taps are set to zero (memoryless seed).
           The user then optionally modifies off-diagonal taps to add memory.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import NonlinearNode


class MemoryPolynomial(NonlinearNode):
    """
    Memory polynomial nonlinear model.

    Parameters
    ----------
    coefficients : np.ndarray, complex128, shape (n_orders, M+1)
        Coefficient matrix a_{km}.
        Row index maps to odd order k via orders[row].
        Column index is memory tap m ∈ {0, ..., M}.
    orders : list of int
        Odd orders included, e.g. [1, 3, 5]. Must match number of rows in
        coefficients.
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        orders: list[int] = [1, 3, 5],
    ) -> None:
        coefficients = np.asarray(coefficients, dtype=complex)
        if coefficients.ndim != 2:
            raise ValueError("coefficients must be a 2-D array (n_orders, M+1).")
        if len(orders) != coefficients.shape[0]:
            raise ValueError(
                f"len(orders)={len(orders)} must match coefficients.shape[0]="
                f"{coefficients.shape[0]}."
            )
        for k in orders:
            if k % 2 == 0 or k < 1:
                raise ValueError(f"Order {k} is not a positive odd integer.")
        self._coefficients = coefficients
        self._orders = list(orders)
        self._M = coefficients.shape[1] - 1   # memory depth

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_p1db_ip3(
        cls,
        p1db_amplitude: float,
        ip3_amplitude: float,
        small_signal_gain: float = 1.0,
        memory_depth: int = 0,
        orders: list[int] = [1, 3, 5],
    ) -> "MemoryPolynomial":
        """
        Build a memoryless seed from P1dB and IP3, optionally extended to
        memory_depth taps (all memory taps initialised to zero).

        The k=1 coefficient (a_{1,0}) is set to the small-signal gain.
        The k=3 coefficient (a_{3,0}) is derived from the cubic describing
        function result (PRD §5.1):

            G[A] ≈ G₀ · (1 + (3c/4)·A²)
            where c = -4/(3·A_IP3²)
            → a_{3,0} = G₀ · (3c/4) = -G₀ / A_IP3²
        """
        M = memory_depth
        n_orders = len(orders)
        coeff = np.zeros((n_orders, M + 1), dtype=complex)

        a_ip3 = float(ip3_amplitude)

        for row, k in enumerate(orders):
            if k == 1:
                coeff[row, 0] = complex(small_signal_gain)
            elif k == 3:
                # Describing function coefficient for x³ is 3/4;
                # so the effective complex baseband 3rd-order coeff is:
                # a_{3,0} = (3/4) · c  where c = -4/(3·A_IP3²)
                # → a_{3,0} = -1 / A_IP3²
                coeff[row, 0] = complex(-small_signal_gain / (a_ip3**2))
            # Higher orders left at zero (no data to fit them without measurements)

        return cls(coeff, orders)

    @classmethod
    def from_saleh(
        cls,
        saleh_model,
        memory_depth: int = 0,
        amp_max: float = 1.0,
        n_fit_points: int = 64,
        orders: list[int] = [1, 3, 5],
    ) -> "MemoryPolynomial":
        """
        Fit memory polynomial coefficients to a SalehModel at M=0 (memoryless).
        Useful to start from a Saleh fit and then add memory taps.

        Fits polynomial a_{k,0} via least-squares regression on the Saleh
        AM-AM curve evaluated at n_fit_points amplitude levels.
        """
        from .saleh import SalehModel
        amp = np.linspace(0, amp_max, n_fit_points)
        # Target: complex gain F[A] = G[A] · exp(j·Φ[A]) evaluated pointwise
        g = saleh_model.gain(amp)
        phi = saleh_model.phase_shift(amp)
        # Output envelope = F[A] · A  →  output as function of input
        u_in = amp.astype(complex)
        u_out = g * np.exp(1j * phi) * u_in

        M = memory_depth
        n_orders = len(orders)
        coeff = np.zeros((n_orders, M + 1), dtype=complex)

        # Fit each order independently at m=0 via OLS
        # Build regressor matrix: columns are u_in · |u_in|^(k-1) for each k
        X = np.column_stack(
            [u_in * (amp ** (k - 1)) for k in orders]
        )
        # Least-squares: X @ c = u_out
        c, _, _, _ = np.linalg.lstsq(X, u_out, rcond=None)
        for row, _ in enumerate(orders):
            coeff[row, 0] = c[row]

        return cls(coeff, orders)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def memory_depth(self) -> int:
        return self._M

    @property
    def supports_baseband(self) -> bool:
        return True

    @property
    def supports_real_axis(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Core application
    # ------------------------------------------------------------------

    def apply_baseband(self, u: np.ndarray) -> np.ndarray:
        """
        Apply memory polynomial to complex baseband envelope ũ(t).

        ũ_out[n] = Σ_k Σ_m a_{km} · ũ[n-m] · |ũ[n-m]|^(k-1)
        """
        u = np.asarray(u, dtype=complex)
        N = len(u)
        out = np.zeros(N, dtype=complex)

        for row, k in enumerate(self._orders):
            for m in range(self._M + 1):
                a_km = self._coefficients[row, m]
                if a_km == 0:
                    continue
                # Delayed signal (zero-pad past boundaries)
                if m == 0:
                    u_del = u
                else:
                    u_del = np.concatenate([np.zeros(m, dtype=complex), u[:-m]])
                amp_del = np.abs(u_del)
                out += a_km * u_del * (amp_del ** (k - 1))

        return out

    def __repr__(self) -> str:
        return (
            f"MemoryPolynomial(orders={self._orders}, M={self._M}, "
            f"coeff_shape={self._coefficients.shape})"
        )
