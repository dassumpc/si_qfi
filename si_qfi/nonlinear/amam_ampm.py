"""
si_qfi.nonlinear.amam_ampm
==========================
Tabulated AM-AM / AM-PM nonlinear model (complex baseband mode only).

Takes measured or simulated amplitude-in / amplitude-out and amplitude-in /
phase-shift-rad tables and interpolates via cubic spline. No parametric fit
required — this is the most faithful representation of a real measured amplifier.

The describing function justification (PRD §5.1) guarantees that this model
is exact under the narrowband + harmonic-filtering assumptions.

Gain convention (PRD §3.6)
---------------------------
The amp_out/amp_in curve should approach unity slope as amp_in → 0: the
device's linear gain belongs in the SI schematic (small-signal S-parameter
block), and this table should describe only the amplitude-dependent
deviation from that response. A curve taken directly from a full device
measurement (gain included) will instead have a small-signal slope equal to
the device's real gain — if that device is also in the schematic, this
double-counts its gain. `siq.run()` warns if small_signal_gain deviates from
1.0 by more than ~3 dB.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Optional

from .base import NonlinearNode


class TabulatedAMAM(NonlinearNode):
    """
    AM-AM / AM-PM model defined by measured amplitude tables.

    Parameters
    ----------
    amp_in : array-like, shape (N,)
        Input amplitude values (volts peak or normalised). Must be monotonically
        increasing and start at or near zero.
    amp_out : array-like, shape (N,)
        Output amplitude values corresponding to amp_in.
    phase_shift_rad : array-like, shape (N,), optional
        Phase shift (radians) as a function of input amplitude.
        If None, AM-PM is disabled.
    extrapolate : str
        How to handle input amplitudes outside the tabulated range.
        'clip'   — clamp to the last tabulated value (default, safe).
        'linear' — linear extrapolation beyond the table edges.
        'error'  — raise ValueError for out-of-range inputs.
    """

    def __init__(
        self,
        amp_in: np.ndarray,
        amp_out: np.ndarray,
        phase_shift_rad: Optional[np.ndarray] = None,
        extrapolate: str = "clip",
    ) -> None:
        amp_in = np.asarray(amp_in, dtype=float)
        amp_out = np.asarray(amp_out, dtype=float)

        if amp_in.ndim != 1 or amp_out.ndim != 1:
            raise ValueError("amp_in and amp_out must be 1-D arrays.")
        if len(amp_in) != len(amp_out):
            raise ValueError("amp_in and amp_out must have the same length.")
        if not np.all(np.diff(amp_in) > 0):
            raise ValueError("amp_in must be strictly monotonically increasing.")
        if extrapolate not in ("clip", "linear", "error"):
            raise ValueError("extrapolate must be 'clip', 'linear', or 'error'.")

        self._amp_in = amp_in
        self._amp_out = amp_out
        self._extrapolate = extrapolate

        # Fit AM-AM spline: maps input amplitude → output amplitude
        self._amam_spline = CubicSpline(amp_in, amp_out, extrapolate=(extrapolate == "linear"))

        # Fit AM-PM spline if provided
        self._ampm_spline: Optional[CubicSpline] = None
        if phase_shift_rad is not None:
            phase_shift_rad = np.asarray(phase_shift_rad, dtype=float)
            if len(phase_shift_rad) != len(amp_in):
                raise ValueError("phase_shift_rad must have same length as amp_in.")
            self._ampm_spline = CubicSpline(
                amp_in, phase_shift_rad, extrapolate=(extrapolate == "linear")
            )

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def output_amplitude(self, amp: np.ndarray) -> np.ndarray:
        """Evaluate AM-AM: output amplitude given input amplitude."""
        amp = np.asarray(amp, dtype=float)
        if self._extrapolate == "error":
            if np.any(amp < self._amp_in[0]) or np.any(amp > self._amp_in[-1]):
                raise ValueError(
                    f"Input amplitude out of tabulated range "
                    f"[{self._amp_in[0]:.4g}, {self._amp_in[-1]:.4g}]."
                )
        elif self._extrapolate == "clip":
            amp = np.clip(amp, self._amp_in[0], self._amp_in[-1])
        return self._amam_spline(amp)

    def phase_shift(self, amp: np.ndarray) -> np.ndarray:
        """Evaluate AM-PM: phase shift (radians) given input amplitude."""
        amp = np.asarray(amp, dtype=float)
        if self._ampm_spline is None:
            return np.zeros_like(amp)
        if self._extrapolate == "clip":
            amp = np.clip(amp, self._amp_in[0], self._amp_in[-1])
        return self._ampm_spline(amp)

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
        """
        AM-AM slope at the lowest tabulated amplitude, as an estimate of the
        curve's linear gain. Should be ≈1.0 — see module docstring.
        """
        return float(self._amam_spline(self._amp_in[0], 1))

    def apply_baseband(self, u: np.ndarray) -> np.ndarray:
        """
        Apply tabulated AM-AM/AM-PM to complex baseband envelope ũ(t).

        ũ_out(t) = A_out(A(t)) / A(t) · exp(j·Φ(A(t))) · ũ(t)
        """
        u = np.asarray(u, dtype=complex)
        amp_in = np.abs(u)
        amp_out = self.output_amplitude(amp_in)
        phi = self.phase_shift(amp_in)

        safe_amp = np.where(amp_in > 0, amp_in, 1.0)
        u_norm = u / safe_amp
        # Where amp_in == 0, output is 0 regardless
        scale = np.where(amp_in > 0, amp_out * np.exp(1j * phi), 0.0 + 0.0j)
        return scale * u_norm

    def __repr__(self) -> str:
        has_ampm = self._ampm_spline is not None
        return (
            f"TabulatedAMAM(N={len(self._amp_in)}, AM-PM={'yes' if has_ampm else 'no'}, "
            f"range=[{self._amp_in[0]:.3g}, {self._amp_in[-1]:.3g}])"
        )
