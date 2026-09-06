"""
si_qfi.nonlinear.tabulated
===========================
Generic table-based AM-AM / AM-PM nonlinear model: interpolate a caller-
supplied amplitude map directly, rather than fitting a parametric shape
(Saleh's bounded rational, Volterra's polynomial).

Motivation: some real devices have a compression/response curve that doesn't
match either existing model's functional form. The concrete case that
motivated this: an acousto-optic modulator's RF-power-to-diffracted-light-
amplitude response is sinusoidal (`A_out = sin(kappa*A_in)` for a Bragg
cell), and is genuinely non-monotonic past its first diffraction maximum --
neither Saleh's rational G[A] (bounded, monotonic-until-overdriven) nor a
truncated Volterra polynomial (unbounded) represent that shape. Rather than
add device-specific physics to si_qfi, this model lets the caller supply
*any* amplitude-in -> amplitude-out (and optional phase) table and have it
interpolated -- the AOM case, or any other device with a measured/derived
curve, becomes one more table, not new si_qfi code.

Table representation -- output amplitude, not gain
----------------------------------------------------
Stores `(amplitude, output_amplitude)` pairs and interpolates the *output
amplitude* directly (np.interp), not a gain ratio `output/input`. This
sidesteps the ratio's singularity at A=0 and is literally "the AM-AM map":
a direct amplitude-in -> amplitude-out lookup, well-defined everywhere the
table covers. Both arrays must start at exactly (0.0, 0.0) -- zero drive
must produce zero output for any physical AM-AM curve, and fixing this
convention removes the need to special-case interpolation near the origin.

One class, both modes
-----------------------
Unlike SalehModel/SalehRealAxisModel, a single TabulatedModel supports both
complex_baseband and real_axis:
  - Baseband: amp = |u|; interpolate output_amplitude and (if given)
    phase_rad at amp; same construction as SalehModel.apply_baseband (unit
    phasor * interpolated output amplitude * exp(j*phase)).
  - Real-axis: apply to sign(v) * interp(|v|, amplitude, output_amplitude)
    -- the standard odd-symmetry convention every real AM-AM datasheet curve
    implies (matches how SalehRealAxisModel gets oddness from G[A] depending
    only on A**2). The phase table is simply unused here -- there is no
    separate envelope phase to modulate on the real axis (same reasoning
    SalehRealAxisModel's own docstring gives), not an error condition, which
    is why one class covers both modes instead of Saleh's two-class split.

Extrapolation vs. non-monotonicity -- a deliberately different warning than
Saleh/Volterra's max_monotonic_amplitude
-----------------------------------------------------------------------------
np.interp already clamps values outside [amplitude[0], amplitude[-1]] to the
boundary output values. apply_baseband/apply_real_axis warn when the input
peak exceeds amplitude[-1] (the table's calibrated range), since beyond that
the flat-clamped extrapolation has no physical basis. Crucially, this is NOT
a monotonicity check: unlike Saleh/Volterra's max_monotonic_amplitude (which
flags turning-over output as a sign of overdriving a model whose real-world
counterpart wouldn't behave that way), a table is explicitly allowed to be
non-monotonic *within* its own range -- the AOM's sin(kappa*A) turning over
past the first diffraction maximum is a legitimate, intentional table shape,
not a bug signal. No monotonicity check is performed within the table range.

Gain convention (PRD Sec 3.6)
-------------------------------
small_signal_gain is estimated from the table's *second* point
(output_amplitude[1]/amplitude[1]), since the first point is fixed at (0,0)
and can't itself supply a ratio. Returns None if the table has fewer than 2
points. Feeds nonlinear/registry.py's existing gain-convention warning
unchanged -- a table whose near-origin slope deviates materially from unity
likely double-counts a device's linear gain that's also in the SI schematic,
same reasoning as Saleh/Volterra.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .base import NonlinearNode


class TabulatedModel(NonlinearNode):
    """
    Generic table-based AM-AM / AM-PM nonlinear model (both simulation modes).

    Parameters
    ----------
    amplitude : array-like, ascending, must start at exactly 0.0
        Input amplitudes (same units as the waveform, e.g. volts peak).
    output_amplitude : array-like, same length as amplitude, must start at
        exactly 0.0
        Corresponding output amplitudes. Need not be monotonic (e.g. an
        AOM's sin(kappa*A) diffraction response past its first maximum).
    phase_rad : array-like, same length, optional
        AM-PM phase shift (radians) at each amplitude point. Baseband mode
        only -- ignored in real-axis mode. Omit for no AM-PM.
    """

    def __init__(
        self,
        amplitude: np.ndarray,
        output_amplitude: np.ndarray,
        phase_rad: Optional[np.ndarray] = None,
    ) -> None:
        amp = np.asarray(amplitude, dtype=float)
        out = np.asarray(output_amplitude, dtype=float)
        if amp.ndim != 1 or out.ndim != 1:
            raise ValueError("amplitude and output_amplitude must be 1-D arrays.")
        if len(amp) != len(out):
            raise ValueError(
                f"amplitude (len {len(amp)}) and output_amplitude "
                f"(len {len(out)}) must have the same length."
            )
        if len(amp) < 2:
            raise ValueError("amplitude/output_amplitude need at least 2 points.")
        if amp[0] != 0.0 or out[0] != 0.0:
            raise ValueError(
                "TabulatedModel requires the table to start at exactly "
                "(0.0, 0.0) -- zero drive must produce zero output. Got "
                f"(amplitude[0]={amp[0]!r}, output_amplitude[0]={out[0]!r})."
            )
        if np.any(np.diff(amp) <= 0):
            raise ValueError("amplitude must be strictly ascending.")

        self._amplitude = amp
        self._output_amplitude = out
        if phase_rad is not None:
            phase = np.asarray(phase_rad, dtype=float)
            if len(phase) != len(amp):
                raise ValueError(
                    f"phase_rad (len {len(phase)}) must have the same length "
                    f"as amplitude (len {len(amp)})."
                )
            self._phase_rad = phase
        else:
            self._phase_rad = None

    # ------------------------------------------------------------------
    # NonlinearNode interface
    # ------------------------------------------------------------------

    @property
    def supports_baseband(self) -> bool:
        return True

    @property
    def supports_real_axis(self) -> bool:
        return True

    @property
    def small_signal_gain(self) -> Optional[float]:
        """
        Estimated from the table's second point (output_amplitude[1]/
        amplitude[1]) -- see module docstring. None if that point is at
        amplitude 0 (shouldn't happen given the strictly-ascending check in
        __init__, but guarded defensively).
        """
        if self._amplitude[1] == 0.0:
            return None
        return float(self._output_amplitude[1] / self._amplitude[1])

    def _warn_if_out_of_range(self, amp: np.ndarray) -> None:
        if len(amp) == 0:
            return
        table_max = float(self._amplitude[-1])
        peak = float(np.max(amp))
        if peak > table_max:
            warnings.warn(
                f"SI-QFI: TabulatedModel input peak amplitude ({peak:.4g}) "
                f"exceeds the table's calibrated range (max amplitude "
                f"{table_max:.4g}). Values beyond this are flat-clamped to "
                f"the table's last output_amplitude entry, which has no "
                f"physical basis past that point -- extend the table if "
                f"operation out here needs to be trusted.",
                stacklevel=3,
            )

    def apply_baseband(self, u: np.ndarray) -> np.ndarray:
        """
        Apply the tabulated AM-AM/AM-PM map to a complex baseband envelope.

        ũ_out(t) = interp(A(t)) * exp(j*phase_interp(A(t))) * ũ(t)/A(t)

        where A(t) = |ũ(t)|. Handles A(t) = 0 safely.
        """
        u = np.asarray(u, dtype=complex)
        amp = np.abs(u)
        self._warn_if_out_of_range(amp)
        out_amp = np.interp(amp, self._amplitude, self._output_amplitude)
        if self._phase_rad is not None:
            phase = np.interp(amp, self._amplitude, self._phase_rad)
        else:
            phase = np.zeros_like(amp)
        safe_amp = np.where(amp > 0, amp, 1.0)
        u_norm = u / safe_amp
        return out_amp * np.exp(1j * phase) * u_norm

    def apply_real_axis(self, v: np.ndarray) -> np.ndarray:
        """
        Apply sign(v) * interp(|v|) to the real RF waveform -- the standard
        odd-symmetry convention for a real AM-AM curve (see module
        docstring). The AM-PM (phase_rad) table, if any, is unused here.
        """
        v = np.asarray(v, dtype=float)
        abs_v = np.abs(v)
        self._warn_if_out_of_range(abs_v)
        out_amp = np.interp(abs_v, self._amplitude, self._output_amplitude)
        return np.sign(v) * out_amp

    def __repr__(self) -> str:
        return (
            f"TabulatedModel(n_points={len(self._amplitude)}, "
            f"range=[0, {self._amplitude[-1]:.4g}], "
            f"am_pm={'yes' if self._phase_rad is not None else 'no'})"
        )
