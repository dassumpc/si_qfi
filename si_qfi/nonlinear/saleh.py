"""
si_qfi.nonlinear.saleh
======================
Saleh AM-AM / AM-PM model (complex baseband mode only).

Reference: Saleh, A.A.M., "Frequency-independent and frequency-dependent nonlinear
models of TWT amplifiers," IEEE Trans. Commun., 1981.

Model equations
---------------
    G[A] = α_a · A / (1 + β_a · A²)          AM-AM output amplitude
    Φ[A] = α_φ · A² / (1 + β_φ · A²)         AM-PM phase shift (radians)

    ũ_out(t) = G[|ũ_in(t)|] / |ũ_in(t)|  ·  exp(j·Φ[|ũ_in(t)|])  ·  ũ_in(t)
             = (α_a / (1 + β_a·A²)) · exp(j·Φ[A]) · ũ_in(t)

Relation to P1dB and IP3 (cubic-only approximation, §5.1 of PRD)
-----------------------------------------------------------------
For a cubic polynomial f(x) = x + c·x³ (c < 0 for compression):
    A_IP3  = sqrt(-4 / (3c))
    A_1dB  ≈ 0.383 · A_IP3        (the 9.6 dB rule)

Saleh parameters can be fit from P1dB + IP3 via convenience constructor
`SalehModel.from_p1db_ip3()`, or from measured power-in/power-out data via
`SalehModel.fit()`.

Gain convention (PRD §3.6)
---------------------------
alpha_a is the model's small-signal gain and should be ≈ 1.0: the amplifier's
actual linear gain belongs in the SI schematic (as a small-signal S-parameter
block), and this model supplies only the amplitude-dependent compression on
top of it. `from_p1db_ip3()` defaults `small_signal_gain=1.0` for this reason.
`fit()` on raw measured amp_in/amp_out data will instead recover the device's
real gain as alpha_a — if that same device is also in the schematic, this
double-counts its gain. `siq.run()` warns if alpha_a deviates from 1.0 by
more than ~3 dB.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from typing import Optional

from .base import NonlinearNode


class SalehModel(NonlinearNode):
    """
    Saleh AM-AM / AM-PM nonlinear amplifier model.

    Parameters
    ----------
    alpha_a, beta_a : float
        AM-AM Saleh parameters. alpha_a is the small-signal gain and should
        be ≈ 1.0 under the SI-QFI gain convention — see module docstring.
    alpha_phi, beta_phi : float
        AM-PM Saleh parameters. Set alpha_phi=0 to disable AM-PM.
    """

    def __init__(
        self,
        alpha_a: float,
        beta_a: float,
        alpha_phi: float = 0.0,
        beta_phi: float = 0.0,
    ) -> None:
        self.alpha_a = float(alpha_a)
        self.beta_a = float(beta_a)
        self.alpha_phi = float(alpha_phi)
        self.beta_phi = float(beta_phi)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_p1db_ip3(
        cls,
        p1db_amplitude: float,
        ip3_amplitude: float,
        small_signal_gain: float = 1.0,
        enable_am_pm: bool = False,
        am_pm_peak_deg: float = 0.0,
    ) -> "SalehModel":
        """
        Construct a SalehModel from P1dB and IP3 amplitude values.

        Uses the cubic-polynomial approximation from PRD §5.1 to derive the
        describing-function coefficient, then maps it to Saleh parameters.

        Parameters
        ----------
        p1db_amplitude : float
            Input amplitude at the 1 dB compression point (same units as waveform,
            e.g. volts peak). Must satisfy p1db_amplitude < ip3_amplitude.
        ip3_amplitude : float
            Input-referred third-order intercept amplitude (volts peak).
        small_signal_gain : float
            Linear (small-signal) voltage gain G₀ = α_a (dimensionless).
        enable_am_pm : bool
            If True, set a heuristic AM-PM based on am_pm_peak_deg at compression.
        am_pm_peak_deg : float
            Peak AM-PM phase shift (degrees) at the compression point (used only
            if enable_am_pm=True).
        """
        # From cubic model: A_IP3 = sqrt(-4/(3c))  →  c = -4/(3·A_IP3²)
        # Describing function coefficient for x³ is 3/4, so effective AM-AM is
        # G[A] ≈ G₀·(1 + (3c/4)·A²) = G₀·(1 - A²/A_IP3²)
        # Map to Saleh: α_a / (1 + β_a·A²) ≈ G₀·(1 - β_a·A²) for small β_a·A²
        # At P1dB: G[A_1dB] = G₀ · 10^(-1/20)
        # → β_a = (1 - 10^(-1/20)) / A_1dB²  ≈ 0.109 / A_1dB²
        a1db = float(p1db_amplitude)
        beta_a = (1.0 - 10 ** (-1.0 / 20.0)) / (a1db**2)
        alpha_a = float(small_signal_gain)

        alpha_phi, beta_phi = 0.0, 0.0
        if enable_am_pm:
            # Simple heuristic: Φ peaks at A_1dB
            # Φ[A_1dB] = alpha_phi·A_1dB² / (1 + beta_phi·A_1dB²) = peak_rad
            peak_rad = np.deg2rad(am_pm_peak_deg)
            # Choose beta_phi so peak is at A_1dB: beta_phi = 1/A_1dB²
            beta_phi = 1.0 / (a1db**2)
            alpha_phi = peak_rad * (1.0 + beta_phi * a1db**2) / (a1db**2)

        return cls(alpha_a, beta_a, alpha_phi, beta_phi)

    @classmethod
    def fit(
        cls,
        amp_in: np.ndarray,
        amp_out: np.ndarray,
        phase_shift_rad: Optional[np.ndarray] = None,
    ) -> "SalehModel":
        """
        Fit Saleh parameters from measured AM-AM (and optionally AM-PM) curves.

        Parameters
        ----------
        amp_in : np.ndarray, shape (N,)
            Input amplitude values (V peak or normalised).
        amp_out : np.ndarray, shape (N,)
            Corresponding output amplitude values.
        phase_shift_rad : np.ndarray, shape (N,), optional
            Output phase shift in radians vs. input amplitude.
            If None, AM-PM is disabled (alpha_phi = 0).

        Returns
        -------
        SalehModel with fitted parameters.
        """
        amp_in = np.asarray(amp_in, dtype=float)
        amp_out = np.asarray(amp_out, dtype=float)

        def saleh_amam(a, alpha_a, beta_a):
            return alpha_a * a / (1.0 + beta_a * a**2)

        p0 = [amp_out[0] / amp_in[0] if amp_in[0] != 0 else 1.0, 1.0]
        popt_a, _ = curve_fit(saleh_amam, amp_in, amp_out, p0=p0, maxfev=10000)
        alpha_a, beta_a = popt_a

        alpha_phi, beta_phi = 0.0, 0.0
        if phase_shift_rad is not None:
            phase_shift_rad = np.asarray(phase_shift_rad, dtype=float)

            def saleh_ampm(a, alpha_phi, beta_phi):
                return alpha_phi * a**2 / (1.0 + beta_phi * a**2)

            p0_phi = [phase_shift_rad[-1] / amp_in[-1] ** 2, 1.0]
            popt_phi, _ = curve_fit(
                saleh_ampm, amp_in, phase_shift_rad, p0=p0_phi, maxfev=10000
            )
            alpha_phi, beta_phi = popt_phi

        return cls(alpha_a, beta_a, alpha_phi, beta_phi)

    # ------------------------------------------------------------------
    # Core model evaluation
    # ------------------------------------------------------------------

    def gain(self, amplitude: np.ndarray) -> np.ndarray:
        """AM-AM complex gain G[A] (real, dimensionless)."""
        a = np.asarray(amplitude, dtype=float)
        return self.alpha_a / (1.0 + self.beta_a * a**2)

    def phase_shift(self, amplitude: np.ndarray) -> np.ndarray:
        """AM-PM phase shift Φ[A] in radians."""
        a = np.asarray(amplitude, dtype=float)
        if self.alpha_phi == 0.0:
            return np.zeros_like(a)
        return self.alpha_phi * a**2 / (1.0 + self.beta_phi * a**2)

    def compression_point_amplitude(self) -> float:
        """
        Estimate the 1 dB compression point amplitude from Saleh parameters.
        Solves G[A_1dB] = α_a · 10^(-1/20) for A_1dB.
        Returns NaN if no compression point exists (linear model).
        """
        # G[A] = α_a/(1+β_a·A²) = α_a·10^(-1/20)
        # → 1/(1+β_a·A²) = 10^(-1/20)
        # → A² = (10^(1/20) - 1) / β_a
        if self.beta_a <= 0:
            return float("nan")
        a2 = (10 ** (1.0 / 20.0) - 1.0) / self.beta_a
        return float(np.sqrt(a2))

    def am_pm_significant(self, threshold_deg: float = 1.0) -> bool:
        """
        Return True if AM-PM phase shift exceeds threshold_deg at the
        compression point. Used by diagnostics to warn about neglected AM-PM.
        """
        if self.alpha_phi == 0.0:
            return False
        a1db = self.compression_point_amplitude()
        if np.isnan(a1db):
            return False
        phi_at_comp = np.rad2deg(self.phase_shift(np.array([a1db]))[0])
        return abs(phi_at_comp) > threshold_deg

    # ------------------------------------------------------------------
    # NonlinearNode interface
    # ------------------------------------------------------------------

    @property
    def supports_baseband(self) -> bool:
        return True

    @property
    def supports_real_axis(self) -> bool:
        return False

    @property
    def small_signal_gain(self) -> float:
        """G[A] as A → 0, i.e. alpha_a. Should be ≈1.0 — see module docstring."""
        return self.alpha_a

    def apply_baseband(self, u: np.ndarray) -> np.ndarray:
        """
        Apply Saleh AM-AM/AM-PM to complex baseband envelope ũ(t).

        ũ_out(t) = G[A(t)] · exp(j·Φ[A(t)]) · ũ(t) / A(t)

        where A(t) = |ũ(t)|.  Handles A(t) = 0 safely.
        """
        u = np.asarray(u, dtype=complex)
        amp = np.abs(u)
        g = self.gain(amp)
        phi = self.phase_shift(amp)
        # Avoid division by zero for zero-amplitude samples
        safe_amp = np.where(amp > 0, amp, 1.0)
        u_norm = u / safe_amp          # unit phasor, or 0/1 = 0 where amp=0
        return g * np.exp(1j * phi) * u_norm * amp

    def __repr__(self) -> str:
        return (
            f"SalehModel(α_a={self.alpha_a:.4f}, β_a={self.beta_a:.4f}, "
            f"α_φ={self.alpha_phi:.4f}, β_φ={self.beta_phi:.4f})"
        )
