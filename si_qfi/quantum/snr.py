"""
si_qfi.quantum.snr
====================
Effective SNR of a simulated pulse's noise-free signal vs. its noise
ensemble at the qubit plane -- generalizes the bandwidth-matching
methodology developed in tests/test_engine_noise.py's
TestNoiseModeEquivalence (real_axis vs complex_baseband noise comparison)
into a reusable utility for any SimulationResult with noise enabled.

Definition: average signal power / average noise power, both computed only
over the time window where the deterministic signal is actually
significant (|v_signal|^2 > window_threshold_frac * its own peak) -- not
averaged over the full array, which is typically padded well beyond the
active pulse by convolution growth (nonlinear segmentation and/or
dispersive transfer functions) and would otherwise dilute the signal-power
estimate with near-zero-signal samples. Noise power is estimated from
v_qubit_ensemble[i] - v_nl_qubit for every realization, averaged over both
realizations and the same time window.

Why flat (in-window) weighting, not weighted by the pulse shape Omega(t):
a first-order (Magnus) sensitivity analysis shows the correct weight
function actually depends on which quadrature the noise lives in relative
to the drive -- flat (uniform) weighting is the EXACT result for noise on
the same axis as the drive (H0(t) and the perturbation commute pointwise,
so the accumulated angle error is simply the unweighted time integral of
the noise, regardless of Omega(t)'s shape), while noise on the orthogonal
quadrature picks up a filter-function weight in the accumulated ROTATION
ANGLE theta(t)=integral(Omega(t')dt') via sin(theta(t))/cos(theta(t)), NOT
Omega(t) itself (checked numerically against this codebase's own Gaussian
pi-pulse: corr(sin(theta(t)), normalized Omega(t))=0.99 but
corr(cos(theta(t)), normalized Omega(t))=-0.02, i.e. Omega(t)-weighting
captures roughly one of the three relevant terms and misses another
entirely). Since the demodulated noise here is isotropic complex Gaussian
(equal I/Q power by construction, see noise/realization.py), all three
terms matter for a fully rigorous treatment. Flat weighting was kept here
deliberately (not Omega(t)-weighted, and not the full 3-term filter-
function version) as the simpler, "good enough for a diagnostic" choice --
exact for half the noise power (I-quadrature) and a defensible zeroth-order
stand-in for the rest. gate_fidelity() already does the exact QuTiP solve
when precision actually matters; this utility is a cheap proxy, not a
replacement for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .hamiltonian import demodulate


@dataclass
class SNRResult:
    snr: float
    snr_db: float
    signal_power: float
    noise_power: float
    n_window_samples: int


def pulse_snr(
    result,
    lpf_cutoff_hz: Optional[float] = None,
    window_threshold_frac: float = 0.05,
) -> SNRResult:
    """
    Effective SNR of a noisy SimulationResult's ensemble relative to its
    own noise-free signal -- see module docstring for the definition.

    For real_axis mode, both the signal and every noise realization are
    first demodulated to a baseband-equivalent complex representation via
    quantum.demodulate() (the standard "x2 single-sideband" bandpass-to-
    envelope conversion -- correct for both the coherent signal and a
    genuinely wideband noise source, see noise/realization.py's
    "Absolute-scale history"); `lpf_cutoff_hz` is REQUIRED in that case --
    there is no signal-bandwidth-independent default this function could
    pick on its own (unlike quantum.demodulate()'s own default of
    carrier_freq_hz, which is chosen for image rejection, not SNR
    bandwidth-matching -- see tests/test_engine_noise.py's module
    docstring for why those are different goals). For complex_baseband
    mode, result.v_nl_qubit / result.v_qubit_ensemble are already in a
    consistent complex-envelope representation and are used as-is;
    `lpf_cutoff_hz` is ignored.

    Parameters
    ----------
    result : SimulationResult
        Must have noise enabled (result.noise_enabled), i.e. built via
        engine.run(..., noise={...}, n_realizations=...).
    lpf_cutoff_hz : float, optional
        Bandwidth-matching LPF cutoff (Hz) for real_axis mode's
        demodulation. Required when result.mode == "real_axis".
    window_threshold_frac : float
        Fraction of the signal's own peak power above which a time sample
        is included in the window (default 0.05, i.e. 5%).

    Returns
    -------
    SNRResult
    """
    if not result.noise_enabled or not result.v_qubit_ensemble:
        raise ValueError(
            "pulse_snr() requires a SimulationResult with noise enabled "
            "(result.noise_enabled) -- build it via engine.run(..., noise={...})."
        )

    if result.mode == "real_axis":
        if lpf_cutoff_hz is None:
            raise ValueError(
                "lpf_cutoff_hz is required for real_axis mode -- see "
                "pulse_snr()'s docstring for why no default is picked automatically."
            )
        t = np.arange(len(result.v_nl_qubit)) / result.fs
        I_sig, Q_sig = demodulate(result.v_nl_qubit, t, result.carrier_freq_hz, lpf_cutoff_hz=lpf_cutoff_hz)
        v_signal = I_sig + 1j * Q_sig
        v_ensemble = []
        for v in result.v_qubit_ensemble:
            I, Q = demodulate(v, t, result.carrier_freq_hz, lpf_cutoff_hz=lpf_cutoff_hz)
            v_ensemble.append(I + 1j * Q)
    else:
        v_signal = result.v_nl_qubit
        v_ensemble = result.v_qubit_ensemble

    sig_power_profile = np.abs(v_signal) ** 2
    peak = sig_power_profile.max()
    window = sig_power_profile > window_threshold_frac * peak
    if window.sum() == 0:
        raise ValueError(
            "pulse_snr(): no samples exceed window_threshold_frac of the signal's own peak power."
        )

    signal_power = float(np.mean(sig_power_profile[window]))

    diffs = np.array([v - v_signal for v in v_ensemble])
    noise_power_profile = np.mean(np.abs(diffs) ** 2, axis=0)
    noise_power = float(np.mean(noise_power_profile[window]))

    snr = signal_power / noise_power
    snr_db = 10.0 * np.log10(snr) if snr > 0 else float("-inf")

    return SNRResult(
        snr=snr,
        snr_db=snr_db,
        signal_power=signal_power,
        noise_power=noise_power,
        n_window_samples=int(window.sum()),
    )
