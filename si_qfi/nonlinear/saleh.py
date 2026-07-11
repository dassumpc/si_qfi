"""
si_qfi.nonlinear.saleh
======================
Classic 2-parameter Saleh AM-AM / AM-PM model, plus its real-axis counterpart.

Reference: Saleh, A.A.M., "Frequency-independent and frequency-dependent nonlinear
models of TWT amplifiers," IEEE Trans. Commun., 1981.

Two classes live here:
  - SalehModel: complex baseband mode. G[A] scales the envelope's magnitude;
    A = |ũ(t)|. Includes AM-PM (phase vs. amplitude).
  - SalehRealAxisModel(SalehModel): real-axis mode. The SAME bounded rational
    G[A] curve, but applied directly to the instantaneous real waveform x(t)
    (A = x(t) itself, signed) rather than to an envelope magnitude -- see
    "Real-axis variant" below. No AM-PM (there is no separate envelope phase
    to modulate when operating directly on the real waveform).

Model equations
---------------
    G[A] = α_a / (1 + β_a·A²)                AM-AM gain (dimensionless)
    Φ[A] = α_φ · A² / (1 + β_φ · A²)         AM-PM phase shift (radians, baseband only)

    Baseband:   ũ_out(t) = G[|ũ_in(t)|] · exp(j·Φ[|ũ_in(t)|]) · ũ_in(t)
    Real axis:  y(t)     = G[x(t)] · x(t)                        (x signed; G[A] only
                                                                    uses A², so this is
                                                                    well-defined and odd)

Fitting from OP1dB/OIP3 -- EXACTLY ONE, not both (output-referred -- see gain
convention below)
------------------------------------------------------------------------------
This classic form has exactly one free shape parameter (β_a) besides the gain
(α_a) -- enough to hit exactly ONE calibration point, not two independently.
`SalehModel.from_op1db_oip3()` therefore requires EXACTLY one of
op1db_amplitude/oip3_amplitude (raises if given neither OR both):
  - oip3_amplitude: β_a fixed by the ASYMPTOTIC two-tone IP3 definition. The
    factor relating β_a to A_IP3 is DOMAIN-DEPENDENT (see "Real-axis variant"
    below for why) -- each class defines its own `_OIP3_BETA_FACTOR`:
      * SalehModel (baseband/envelope): β_a = 1 / A_IP3²
      * SalehRealAxisModel: β_a = (4/3) / A_IP3²  (matches VolterraModel)
  - op1db_amplitude: β_a solved EXACTLY from the 1dB-compression-point
    definition G[A_1dB] = α_a·10^(-1/20) (not the linearized/approximate
    "9.6dB rule"), domain-independent (this equation is just a statement
    about the G[A] curve itself, no two-tone/harmonic physics involved):
        β_a = (10^(1/20) - 1) / A_1dB²

(An earlier version of this module added a second free parameter, γ_a, to a
denominator A⁴ term specifically so BOTH points could be hit exactly at once
-- removed. Fitting both independently needs a second free parameter no
matter how it's added, and that generality wasn't worth the extra surface
area for this codebase right now -- see feedback_lean_scope_no_speculative_
fitting memory. Only one of OP1dB/OIP3 may be specified.)

OIP3-only implies a specific OP1dB (and vice versa) -- NOT a free choice
--------------------------------------------------------------------------
Since β_a has only one degree of freedom, fitting from OIP3 alone
*determines* where this model's actual 1dB compression point falls -- it is
not independently choosable. Solving `compression_point_amplitude()`
(β_a = _OIP3_BETA_FACTOR/A_IP3²) gives, in closed form (α_a=1):

    A_1dB,in = A_IP3 · sqrt((10^(1/20)-1) / _OIP3_BETA_FACTOR)
    OP1dB    = A_1dB,in · 10^(-1/20)                                (output-referred)
    OP1dB / OIP3 = 10^(-1/20) · sqrt((10^(1/20)-1) / _OIP3_BETA_FACTOR)

For SalehModel (baseband, factor 1.0): OP1dB/OIP3 ≈ 10**(-10.14/20) --
i.e. OP1dB sits **~10.1 dB below OIP3** (output-referred, in dB). This is a
DIFFERENT number from a plain cubic Volterra polynomial's OIP3-implied OP1dB
(~10.6 dB below -- see nonlinear/volterra.py module docstring) precisely
because Saleh's rational G[A] and a truncated cubic are different SHAPES of
nonlinearity that only agree asymptotically (same leading-order IP3
behavior), not at the compression point itself. Verified in
tests/test_nonlinear.py.

For SalehRealAxisModel (factor 4/3): the analogous *raw* pointwise
evaluation gives a DIFFERENT (larger) gap that does NOT match the physical
single-tone measurement -- see "Real-axis variant" below for why, and why
tests/test_nonlinear.py verifies SalehRealAxisModel's actual compression
point via simulated single-tone drive (FFT-extracted fundamental) rather
than via a closed-form/raw-evaluation shortcut.

Real-axis variant (SalehRealAxisModel)
---------------------------------------
Applying G[A] directly to the instantaneous real waveform x(t) (rather than
to an envelope magnitude) means a two-tone real bandpass input generates
genuine harmonic/intermodulation content via ordinary waveform distortion --
the same mechanism VolterraModel's polynomial exploits. Because G[A] only
uses A², applying it to a signed x(t) is well-defined and automatically
odd-symmetric (G[x]·x is an odd function of x), matching a real amplifier's
AM-AM curve.

The two-tone IP3 crossing condition differs from the baseband/envelope case
by a factor of 4/3, because the real bandpass cubic expansion
(cos³θ = (3/4)cosθ + (1/4)cos3θ) splits energy between the fundamental and
third harmonic/IM3 directly on the real axis, whereas the complex baseband
envelope model already represents only the in-band (fundamental) content --
its own (3/4) reduction factor exactly cancels the (4/3) that appears in the
real-axis derivation.

IMPORTANT: this 4/3 factor is verified (tests/test_nonlinear.py) to make
SalehRealAxisModel's TRUE single-tone-driven fundamental response (i.e. the
physically meaningful thing a spectrum analyzer would measure -- extracted
via FFT from an actual simulated sinusoid through apply_real_axis()) track
the baseband SalehModel's gain(A) curve closely (same OIP3, both alpha_a=1)
-- NOT by making apply_real_axis() applied to a raw CONSTANT value match
baseband's gain(A)*A exactly. Those are different things: G[A]·A evaluated
at a constant A is a quasi-static/DC-like curve with no direct physical
meaning for a real (AC) waveform, since it doesn't separate the fundamental
from harmonic content the way an actual sinusoidal drive + Fourier analysis
does. For the baseband envelope model this distinction doesn't arise (G[A]·A
*is* the fundamental response by construction -- see PRD §5.1). For the
real-axis model it does, and only the FFT/single-tone-based comparison is a
meaningful equivalence check.

Gain convention (PRD §3.6)
---------------------------
alpha_a is the model's small-signal gain and should be ≈ 1.0: the amplifier's
actual linear gain belongs in the SI schematic (as a small-signal S-parameter
block), and this model supplies only the amplitude-dependent compression on
top of it. `from_op1db_oip3()` takes no gain argument at all and always
builds alpha_a=1.0 -- a purely output-referred nonlinearity, with no
gain-driven input/output conversion to reason about. (The general
constructor, `SalehModel(alpha_a, beta_a, ...)`, still accepts an arbitrary
alpha_a for the rare case in PRD §3.6 -- "When this convention does not
apply" -- where a device has no separate linear representation and its full
response, gain included, must live in the nonlinear model itself.)
`siq.run()` warns if alpha_a deviates from 1.0 by more than ~3 dB.

All nonlinearity specs in this codebase (op1db_amplitude, oip3_amplitude) are
OUTPUT-referred by convention -- i.e. the actual output amplitude at that
point, not the input amplitude that produced it. Since from_op1db_oip3()
always uses alpha_a=1.0, output-referred and input-referred amplitudes differ
only by the compression itself, never by a separate gain factor: OIP3 needs
no conversion at all (oip3_in = oip3_amplitude exactly); OP1dB still needs
the 1dB-compression-ratio conversion (op1db_in = op1db_amplitude /
10^(-1/20)), since that's a statement about compression, not gain.

max_monotonic_amplitude (bug fixed vs. the removed γ_a-extended version)
--------------------------------------------------------------------------
Even this classic 2-parameter form has a genuine breakdown amplitude: raw
output y(A) = α_a·A/(1+β_a·A²) is NOT monotonically increasing for all A --
it peaks at A = 1/sqrt(β_a) and DECLINES beyond that (never true for a real
amplifier short of hard clipping), even though gain itself compresses
monotonically the whole time (gain-expansion never happens for β_a > 0). An
earlier version of this module (when γ_a existed) incorrectly short-circuited
`max_monotonic_amplitude` to infinity whenever γ_a >= 0 -- which included
every single-point (OIP3-only or OP1dB-only) fit -- missing this raw-output
turnover criterion entirely (it only checked gain-expansion/pole criteria,
mirroring VolterraModel's *two*-criterion check but dropping one of the two).
Fixed here: max_monotonic_amplitude = 1/sqrt(β_a) for β_a > 0 (the ordinary
case), inf for β_a == 0, or 1/sqrt(-β_a) (denominator pole) for β_a < 0 (only
reachable via direct construction, never from_op1db_oip3()).
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional

from .base import NonlinearNode

_ONE_DB_RATIO = 10 ** (-1.0 / 20.0)   # linear amplitude ratio at -1dB


class SalehModel(NonlinearNode):
    """
    Saleh AM-AM / AM-PM nonlinear amplifier model (complex baseband mode).

    Parameters
    ----------
    alpha_a, beta_a : float
        AM-AM Saleh parameters. alpha_a is the small-signal gain and should
        be ≈ 1.0 under the SI-QFI gain convention — see module docstring.
    alpha_phi, beta_phi : float
        AM-PM Saleh parameters. Set alpha_phi=0 to disable AM-PM.
    """

    # Two-tone IP3 -> beta_a factor. Domain-dependent -- see module
    # docstring's "Real-axis variant" section. SalehRealAxisModel overrides
    # this to 4/3.
    _OIP3_BETA_FACTOR = 1.0

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
    def from_op1db_oip3(
        cls,
        op1db_amplitude: Optional[float] = None,
        oip3_amplitude: Optional[float] = None,
        enable_am_pm: bool = False,
        am_pm_peak_deg: float = 0.0,
    ) -> "SalehModel":
        """
        Construct a SalehModel (or SalehRealAxisModel, via cls) from an
        OUTPUT-referred P1dB or IP3 amplitude value -- EXACTLY one, not both
        (see module docstring for the full derivation, the domain-dependent
        _OIP3_BETA_FACTOR, and the output-referred convention). No gain
        argument: alpha_a is always 1.0 (PRD §3.6 gain convention -- the
        SI schematic supplies a device's actual linear gain, so this is a
        purely output-referred nonlinearity with no separate gain factor to
        convert through).

        Parameters
        ----------
        op1db_amplitude : float, optional
            OUTPUT-referred amplitude at the 1 dB compression point (same
            units as waveform, e.g. volts peak) -- i.e. the actual
            (compressed) output amplitude at that point, not the input
            amplitude that produced it. Since alpha_a=1.0, the corresponding
            input-referred amplitude (op1db_in, what the fit below actually
            solves against) is op1db_amplitude / 10^(-1/20) -- larger than
            op1db_amplitude itself, since driving the (unity-gain) model at
            op1db_in is what produces the 1dB-compressed output
            op1db_amplitude. Mutually exclusive with oip3_amplitude.
        oip3_amplitude : float, optional
            OUTPUT-referred third-order intercept amplitude (OIP3, volts
            peak). Since alpha_a=1.0, this needs no conversion at all --
            oip3_in (input-referred) equals oip3_amplitude exactly. Mutually
            exclusive with op1db_amplitude. Note: this alone determines
            (does not leave free) where the model's actual OP1dB falls --
            see module docstring's "OIP3-only implies a specific OP1dB"
            section.
        enable_am_pm : bool
            If True, set a heuristic AM-PM based on am_pm_peak_deg at
            whichever of op1db_amplitude/oip3_amplitude was supplied.
            Not supported by SalehRealAxisModel (overridden there to raise).
        am_pm_peak_deg : float
            Peak AM-PM phase shift (degrees) at that amplitude (used only
            if enable_am_pm=True).
        """
        if op1db_amplitude is None and oip3_amplitude is None:
            raise ValueError(
                "from_op1db_oip3() requires exactly one of op1db_amplitude "
                "or oip3_amplitude -- neither was given."
            )
        if op1db_amplitude is not None and oip3_amplitude is not None:
            raise ValueError(
                "from_op1db_oip3() accepts exactly one of op1db_amplitude / "
                "oip3_amplitude, not both -- this classic 2-parameter Saleh "
                "form has only one free shape parameter (beta_a), which can "
                "be calibrated from ONE point only. Fitting both exactly "
                "would need a second free parameter, which this codebase "
                "deliberately does not support (see module docstring)."
            )
        alpha_a = 1.0

        if oip3_amplitude is not None:
            a_ref = float(oip3_amplitude)   # oip3_in == oip3_amplitude (alpha_a=1)
            beta_a = cls._OIP3_BETA_FACTOR / (a_ref ** 2)
        else:
            # OP1dB only: exact single-point solve of the rational
            # equation (not the linearized "9.6dB rule" approximation).
            # Domain-independent -- see module docstring.
            a_ref = float(op1db_amplitude) / _ONE_DB_RATIO   # op1db_in
            beta_a = (10 ** (1.0 / 20.0) - 1.0) / (a_ref ** 2)

        alpha_phi, beta_phi = 0.0, 0.0
        if enable_am_pm:
            peak_rad = np.deg2rad(am_pm_peak_deg)
            beta_phi = 1.0 / (a_ref ** 2)
            alpha_phi = peak_rad * (1.0 + beta_phi * a_ref ** 2) / (a_ref ** 2)

        return cls(alpha_a, beta_a, alpha_phi, beta_phi)

    # ------------------------------------------------------------------
    # Core model evaluation
    # ------------------------------------------------------------------

    def gain(self, amplitude: np.ndarray) -> np.ndarray:
        """AM-AM gain G[A] (real, dimensionless). Only uses A², so this is
        well-defined for signed A too (see SalehRealAxisModel)."""
        a = np.asarray(amplitude, dtype=float)
        return self.alpha_a / (1.0 + self.beta_a * a ** 2)

    def phase_shift(self, amplitude: np.ndarray) -> np.ndarray:
        """AM-PM phase shift Φ[A] in radians."""
        a = np.asarray(amplitude, dtype=float)
        if self.alpha_phi == 0.0:
            return np.zeros_like(a)
        return self.alpha_phi * a**2 / (1.0 + self.beta_phi * a**2)

    def compression_point_amplitude(self) -> float:
        """
        The ACTUAL 1 dB compression point (input-referred) amplitude implied
        by beta_a -- solves G[A_1dB] = alpha_a·10^(-1/20) for A_1dB exactly.
        Returns NaN if no real positive solution exists (beta_a <= 0).
        """
        if self.beta_a <= 0:
            return float("nan")
        target = 10 ** (1.0 / 20.0) - 1.0   # 1+beta_a*A^2 - 1
        return float(np.sqrt(target / self.beta_a))

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

    @property
    def max_monotonic_amplitude(self) -> float:
        """
        Largest input amplitude for which this model still behaves like a
        plausible compressive amplifier -- see module docstring's
        "max_monotonic_amplitude" section for the derivation and the bug
        this fixes relative to the removed gamma_a-extended version.
        """
        if self.beta_a > 0:
            return float(1.0 / np.sqrt(self.beta_a))
        if self.beta_a == 0:
            return float("inf")
        return float(1.0 / np.sqrt(-self.beta_a))   # denominator pole (beta_a < 0)

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
        self._warn_if_overdriven(amp)
        g = self.gain(amp)
        phi = self.phase_shift(amp)
        # Avoid division by zero for zero-amplitude samples
        safe_amp = np.where(amp > 0, amp, 1.0)
        u_norm = u / safe_amp          # unit phasor, or 0/1 = 0 where amp=0
        return g * np.exp(1j * phi) * u_norm * amp

    def _warn_if_overdriven(self, amp: np.ndarray) -> None:
        """Warn if amp's peak exceeds max_monotonic_amplitude -- see its
        docstring for what that means physically."""
        if len(amp) == 0:
            return
        max_valid = self.max_monotonic_amplitude
        if not np.isfinite(max_valid):
            return
        peak = float(np.max(amp))
        if peak > max_valid:
            warnings.warn(
                f"SI-QFI: {type(self).__name__} input peak amplitude ({peak:.4g}) "
                f"exceeds the amplitude ({max_valid:.4g}) beyond which this model's "
                f"raw output stops increasing with more drive (output turns over "
                f"and declines -- never true for a real amplifier short of hard "
                f"clipping). This happens for ANY beta_a > 0 once driven far "
                f"enough past the amplitude used to calibrate "
                f"op1db_amplitude/oip3_amplitude -- a single rational shape "
                f"parameter has no independent control over how gently it "
                f"saturates.",
                stacklevel=3,
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(α_a={self.alpha_a:.4f}, β_a={self.beta_a:.4f}, "
            f"α_φ={self.alpha_phi:.4f}, β_φ={self.beta_phi:.4f})"
        )


class SalehRealAxisModel(SalehModel):
    """
    Real-axis variant of the Saleh model: the same bounded rational G[A]
    curve, applied directly to the instantaneous real waveform x(t) instead
    of to a complex envelope's magnitude -- see module docstring's
    "Real-axis variant" section for the derivation (in particular, why
    _OIP3_BETA_FACTOR = 4/3 here, unlike the baseband case's 1.0).

    No AM-PM: there is no separate envelope phase to modulate when operating
    directly on a real, signed waveform, so alpha_phi/beta_phi stay 0 and
    enable_am_pm=True raises in from_op1db_oip3().
    """

    _OIP3_BETA_FACTOR = 4.0 / 3.0

    @classmethod
    def from_op1db_oip3(
        cls,
        op1db_amplitude: Optional[float] = None,
        oip3_amplitude: Optional[float] = None,
        enable_am_pm: bool = False,
        am_pm_peak_deg: float = 0.0,
    ) -> "SalehRealAxisModel":
        if enable_am_pm:
            raise ValueError(
                "SalehRealAxisModel has no AM-PM mechanism -- there is no "
                "separate envelope/phase representation on the real axis "
                "(see module docstring). enable_am_pm is not supported here."
            )
        return super().from_op1db_oip3(
            op1db_amplitude=op1db_amplitude,
            oip3_amplitude=oip3_amplitude,
            enable_am_pm=False,
            am_pm_peak_deg=am_pm_peak_deg,
        )

    @property
    def supports_baseband(self) -> bool:
        return False

    @property
    def supports_real_axis(self) -> bool:
        return True

    def apply_baseband(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} does not support complex baseband mode."
        )

    def apply_real_axis(self, v: np.ndarray) -> np.ndarray:
        """
        Apply G[x(t)]·x(t) directly to the real RF waveform v(t). G[A] only
        uses A² (see gain()), so this is well-defined for signed v and
        automatically odd-symmetric.
        """
        v = np.asarray(v, dtype=float)
        self._warn_if_overdriven(np.abs(v))
        return v * self.gain(v)
