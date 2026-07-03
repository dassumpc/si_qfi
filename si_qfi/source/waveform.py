"""
si_qfi.source.waveform
======================
SourceWaveform wraps a SignalIntegrity Waveform envelope with a carrier frequency.

The SI Waveform already encodes duration and sample rate; only the carrier frequency
is needed as an additional parameter.

In complex baseband mode the envelope is used directly as ũ(t).
In real-axis mode it is modulated onto the carrier to produce the full RF waveform:
    v(t) = Re{ envelope(t) · exp(j·2π·f_carrier·t) }

# --- CURSOR NOTE ---
# The SignalIntegrity Waveform type lives at:
#   SignalIntegrity.Lib.TimeDomain.Waveform.Waveform
# Key attributes used here:
#   .TimeDescriptor  ->  SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor
#       .Fs          ->  sample rate (samples/sec)
#       .K           ->  number of samples
#       .H           ->  time of first sample (sec)
#   .Values()        ->  list or np.ndarray of waveform samples
# Verify these against the current SI repo before finalising.
# -------------------
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


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
        # --- CURSOR NOTE ---
        # Replace the attribute accesses below with the correct SI Waveform API.
        # The pattern shown is based on SI repo inspection; confirm against source.
        # -------------------
        td = self.envelope.TimeDescriptor
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
        Full real RF waveform v(t) = Re{ ũ(t) · exp(j·2π·fc·t) }, shape (N,).
        Used in real-axis mode as the signal injected at the voltage source.
        """
        carrier = np.exp(1j * 2 * np.pi * self.carrier_freq_hz * self._t)
        return np.real(self._values * carrier)

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

    def check_sample_rate_for_real_axis(self, harmonic_order: int = 3) -> None:
        """
        Raise ValueError if the sample rate is insufficient for real-axis mode.
        Nyquist requires fs > 2 · harmonic_order · carrier_freq.
        """
        required = 2.0 * harmonic_order * self.carrier_freq_hz
        if self._fs < required:
            raise ValueError(
                f"Sample rate {self._fs/1e9:.2f} GSa/s is insufficient for real-axis "
                f"mode with harmonic order {harmonic_order}. "
                f"Need >= {required/1e9:.1f} GSa/s "
                f"(2 × {harmonic_order} × {self.carrier_freq_ghz} GHz carrier)."
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
