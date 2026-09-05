"""
si_qfi.source.waveform
======================
SourceWaveform wraps a SignalIntegrity Waveform envelope with a carrier frequency.

The SI Waveform already encodes duration and sample rate; only the carrier frequency
is needed as an additional parameter.

In complex baseband mode the envelope is used directly as ũ(t).
In real-axis mode it is modulated onto the carrier to produce the full RF waveform:
    v(t) = Re{ envelope(t) · exp(j·2π·f_carrier·t) }
at the schematic's own native sample rate (see rf_waveform_at() and
transfer_function.native_sample_rate() — PRD §3.3): real-axis mode resamples
the envelope to match the schematic, rather than the transfer function being
interpolated onto this waveform's own fs.

# SI API notes (verified against the installed SignalIntegrity source):
#   SignalIntegrity.Lib.TimeDomain.Waveform.Waveform
#   .TimeDescriptor()  ->  METHOD (not an attribute!), returns .td
#                          -> SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor
#       .Fs            ->  sample rate (samples/sec)
#       .K             ->  number of samples
#       .H             ->  time of first sample (sec)
#   .Values()          ->  list of waveform samples (Waveform is itself a
#                          list subclass; .td is the TimeDescriptor attribute
#                          directly, equivalent to calling .TimeDescriptor()).
# -------------------
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


def source_from_envelope_array(shape: np.ndarray, fs: float, carrier_ghz: float) -> "SourceWaveform":
    """
    Build a SourceWaveform directly from a numpy envelope array (as
    returned by build_gaussian_envelope()/build_drag_envelope() below) --
    the SI Waveform/TimeDescriptor construction every demo/test in this
    codebase was hand-duplicating identically.
    """
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


@dataclass
class SourceWaveform:
    """
    Drive waveform for the SI-QFI simulation.

    Parameters
    ----------
    carrier_freq_ghz : float
        Carrier (qubit drive) frequency in GHz.
    envelope : SignalIntegrity Waveform
        Complex baseband pulse envelope. The Waveform's own TimeDescriptor
        defines the sample rate and duration — do not pass those separately.
        For complex baseband mode: values are complex (I + jQ).
        For real-axis mode: values are real; SI-QFI modulates onto carrier.
    """

    carrier_freq_ghz: float
    envelope: Any  # SignalIntegrity.Lib.TimeDomain.Waveform.Waveform

    # Derived attributes populated in __post_init__
    _t: np.ndarray = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _fs: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        td = self.envelope.td
        self._fs = float(td.Fs)           # sample rate, Hz
        n = int(td.K)                      # number of samples
        t0 = float(td.H)                   # time of first sample, seconds
        self._t = t0 + np.arange(n) / self._fs
        raw = self.envelope.Values()
        self._values = np.asarray(raw, dtype=complex)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def carrier_freq_hz(self) -> float:
        """Carrier frequency in Hz."""
        return self.carrier_freq_ghz * 1e9

    @property
    def t(self) -> np.ndarray:
        """Time array (seconds), shape (N,)."""
        return self._t

    @property
    def fs(self) -> float:
        """Sample rate in Hz."""
        return self._fs

    @property
    def dt(self) -> float:
        """Sample interval in seconds."""
        return 1.0 / self._fs

    @property
    def n_samples(self) -> int:
        return len(self._t)

    @property
    def envelope_complex(self) -> np.ndarray:
        """
        Complex baseband envelope ũ(t), shape (N,), dtype complex128.
        If the SI Waveform was real, returns it cast to complex.
        """
        return self._values.astype(complex)

    @property
    def rf_waveform(self) -> np.ndarray:
        """
        Full real RF waveform v(t) = Re{ ũ(t) · exp(j·2π·fc·t) }, shape (N,),
        at this waveform's own sample rate (self.fs).

        NOTE: real-axis mode does not use this directly — it uses
        rf_waveform_at(fs) with the schematic's own native sample rate
        (see transfer_function.native_sample_rate() / PRD §3.3). This
        property remains useful standalone (e.g. for inspecting the pulse
        at its native resolution) and is what real-axis mode falls back to
        when the schematic's native rate happens to equal self.fs.
        """
        carrier = np.exp(1j * 2 * np.pi * self.carrier_freq_hz * self._t)
        return np.real(self._values * carrier)

    def resampled_envelope_at(self, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Complex envelope ũ(t), resampled (FFT-based, scipy.signal.resample)
        to an explicit sample rate `fs` -- WITHOUT modulating onto the
        carrier. Returns (t, envelope_resampled).

        This is the "resample" half of rf_waveform_at(), factored out so a
        caller can apply a per-realization operation to the envelope BEFORE
        modulation (e.g. engine.run()'s phase-noise injection, which needs
        to rotate ũ(t) by exp(j*phi(t)) prior to hitting the carrier --
        rf_waveform_at() itself has no hook for that). Resampling the
        complex envelope directly (rather than the already-modulated real
        RF signal) is safe/robust because the envelope is smooth/
        band-limited by construction.
        """
        if np.isclose(fs, self._fs):
            return self._t, self._values.copy()

        from scipy.signal import resample
        n_new = max(int(round(self.n_samples * fs / self._fs)), 1)
        env_resampled = resample(self._values, n_new)
        t_new = self._t[0] + np.arange(n_new) / fs
        return t_new, env_resampled

    def rf_waveform_at(self, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Real RF waveform v(t), resampled to an explicit sample rate `fs`
        rather than self.fs. Returns (t, v).

        Real-axis mode uses this with the schematic's own native sample
        rate: the schematic's transfer function is the source of truth for
        real-axis time resolution (PRD §3.3), so the drive envelope is
        resampled to match it, rather than the transfer function being
        interpolated onto this waveform's own fs.
        """
        t_new, env_resampled = self.resampled_envelope_at(fs)
        carrier = np.exp(1j * 2 * np.pi * self.carrier_freq_hz * t_new)
        return t_new, np.real(env_resampled * carrier)

    @property
    def bandwidth_hz(self) -> float:
        """
        Rough estimate of signal bandwidth: Nyquist of the envelope sample rate.
        The true occupied bandwidth depends on the pulse shape.
        """
        return self._fs / 2.0

    def narrowband_ratio(self) -> float:
        """
        bandwidth / carrier_freq. Should be << 1 for complex baseband mode to be valid.
        SI-QFI uses this to help decide whether to warn about mode selection.
        """
        return self.bandwidth_hz / self.carrier_freq_hz

    def check_sample_rate_for_real_axis(self, fs: float, harmonic_order: int = 3) -> None:
        """
        Raise ValueError if `fs` is insufficient for real-axis mode.
        Nyquist requires fs > 2 · harmonic_order · carrier_freq.

        `fs` is explicit rather than defaulting to self.fs because real-axis
        mode runs at the schematic's own native sample rate (PRD §3.3), not
        this waveform's original fs — pass
        transfer_function.native_sample_rate(...) here.
        """
        required = 2.0 * harmonic_order * self.carrier_freq_hz
        if fs < required:
            raise ValueError(
                f"Sample rate {fs/1e9:.2f} GSa/s is insufficient for real-axis "
                f"mode with harmonic order {harmonic_order}. "
                f"Need >= {required/1e9:.1f} GSa/s "
                f"(2 × {harmonic_order} × {self.carrier_freq_ghz} GHz carrier). "
                f"This is the schematic's own native sample rate (PRD §3.3) — "
                f"increase its frequency sweep resolution (CalculationProperties "
                f"EndFrequency / UserSampleRate) rather than this waveform's fs."
            )


def build_drag_envelope(
    duration_s: float,
    sigma_s: float,
    anharmonicity_hz: float,
    sample_rate_hz: float,
    pi_amp: float = 1.0,
) -> np.ndarray:
    """
    Generate a DRAG pulse complex envelope as a numpy array.

    Returns complex array (I + jQ) suitable for wrapping in a SignalIntegrity Waveform.
    The caller is responsible for creating the SI Waveform from this array.

    Parameters
    ----------
    duration_s : float
        Total pulse duration (seconds).
    sigma_s : float
        Gaussian sigma (seconds).
    anharmonicity_hz : float
        Qubit anharmonicity in Hz (negative for transmon, e.g. -200e6).
    sample_rate_hz : float
        Sample rate for the output array (Hz).
    pi_amp : float
        Peak amplitude of the Gaussian (arb. units, scales the drive strength).

    Returns
    -------
    envelope : np.ndarray, complex128, shape (N,)
        DRAG pulse complex envelope. Real part = I (Gaussian), imag = Q (derivative).

    Notes
    -----
    DRAG (Derivative Removal via Adiabatic Gate) suppresses leakage to the |2⟩ level.
    Reference: Motzoi et al., PRL 103, 110501 (2009).
        Ω_I(t) = Ω_gauss(t)
        Ω_Q(t) = -dΩ_I/dt / anharmonicity
    """
    n = int(duration_s * sample_rate_hz)
    t = np.linspace(-duration_s / 2, duration_s / 2, n, endpoint=False)
    gauss = pi_amp * np.exp(-(t**2) / (2 * sigma_s**2))
    # Remove DC offset so pulse starts and ends at zero
    gauss -= gauss[0]
    d_gauss = np.gradient(gauss, 1.0 / sample_rate_hz)
    drag_q = -d_gauss / (2 * np.pi * anharmonicity_hz)
    return gauss + 1j * drag_q


def build_gaussian_envelope(
    duration_s: float,
    sigma_s: float,
    sample_rate_hz: float,
    amp: float = 1.0,
) -> np.ndarray:
    """
    Gaussian pulse complex envelope (real-valued, zero imaginary part).
    """
    n = int(duration_s * sample_rate_hz)
    t = np.linspace(-duration_s / 2, duration_s / 2, n, endpoint=False)
    gauss = amp * np.exp(-(t**2) / (2 * sigma_s**2))
    gauss -= gauss[0]
    return gauss.astype(complex)
