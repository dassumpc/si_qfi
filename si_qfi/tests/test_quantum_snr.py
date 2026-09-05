"""
tests/test_quantum_snr.py
==========================
Unit tests for si_qfi.quantum.snr.pulse_snr() -- verifiable without
SignalIntegrity or QuTiP installed (constructs SimulationResult directly
with synthetic arrays rather than running engine.run()).
"""
from __future__ import annotations

import numpy as np
import pytest

from si_qfi.quantum.snr import pulse_snr
from si_qfi.simulation.engine import SimulationResult


def _flat_pulse_result(n_realizations=500, amp=2.0, sigma_noise=0.1, seed=42, outside_burst=0.0):
    """
    A synthetic complex_baseband SimulationResult: a rectangular "pulse"
    (constant amplitude `amp` over the middle third of the array, ~zero
    elsewhere) plus i.i.d. Gaussian noise of known std `sigma_noise` added
    to every realization -- gives a known, closed-form expected SNR =
    amp^2 / sigma_noise^2 to check pulse_snr() against directly.

    `outside_burst`, if nonzero, adds much larger noise OUTSIDE the pulse
    window (where the signal is ~zero) -- used to confirm windowing
    actually excludes those samples from the noise-power estimate.
    """
    rng = np.random.default_rng(seed)
    N = 3000
    signal = np.zeros(N)
    lo, hi = 1000, 2000
    signal[lo:hi] = amp

    ensemble = []
    for _ in range(n_realizations):
        noise = rng.normal(0.0, sigma_noise, N)
        if outside_burst:
            noise[:lo] += rng.normal(0.0, outside_burst, lo)
            noise[hi:] += rng.normal(0.0, outside_burst, N - hi)
        ensemble.append(signal + noise)

    result = SimulationResult(
        v_nl_qubit=signal,
        v_qubit_ensemble=ensemble,
        fs=1e9,
        mode="complex_baseband",
        carrier_freq_hz=5e9,
        noise_enabled=True,
        n_realizations=n_realizations,
    )
    return result, hi - lo


class TestPulseSNR:
    def test_matches_closed_form_snr(self):
        amp, sigma_noise = 2.0, 0.1
        result, window_len = _flat_pulse_result(amp=amp, sigma_noise=sigma_noise)
        snr_result = pulse_snr(result)

        expected_snr = amp ** 2 / sigma_noise ** 2
        rel_err = abs(snr_result.snr - expected_snr) / expected_snr
        assert rel_err < 0.05, (
            f"SNR {snr_result.snr:.3e} should be near closed-form {expected_snr:.3e} (rel err {rel_err:.2%})"
        )
        assert snr_result.n_window_samples == window_len
        assert snr_result.snr_db == pytest.approx(10.0 * np.log10(snr_result.snr))

    def test_windowing_ignores_noise_outside_signal(self):
        """A huge noise burst OUTSIDE the signal-active window should not
        move the reported SNR at all -- confirms noise power is genuinely
        restricted to the window, not averaged over the full array."""
        amp, sigma_noise = 2.0, 0.1
        result_clean, _ = _flat_pulse_result(amp=amp, sigma_noise=sigma_noise, outside_burst=0.0)
        result_burst, _ = _flat_pulse_result(amp=amp, sigma_noise=sigma_noise, outside_burst=50.0)

        snr_clean = pulse_snr(result_clean).snr
        snr_burst = pulse_snr(result_burst).snr
        rel_diff = abs(snr_clean - snr_burst) / snr_clean
        assert rel_diff < 0.05, (
            f"SNR should be unaffected by out-of-window noise: clean={snr_clean:.3e}, "
            f"with burst={snr_burst:.3e} (rel diff {rel_diff:.1%})"
        )

    def test_real_axis_requires_lpf_cutoff(self):
        result, _ = _flat_pulse_result()
        result.mode = "real_axis"
        with pytest.raises(ValueError, match="lpf_cutoff_hz"):
            pulse_snr(result)

    def test_real_axis_runs_with_lpf_cutoff(self):
        """Exercises the real_axis demodulation code path end to end (not
        checking absolute-scale correctness -- that's covered exhaustively
        by tests/test_noise.py's real_axis-vs-baseband equivalence tests)."""
        fs = 20e9
        N = 8192
        t = np.arange(N) / fs
        carrier_hz = 5e9
        envelope = np.zeros(N)
        envelope[2000:6000] = 1.0
        signal = envelope * np.cos(2 * np.pi * carrier_hz * t)

        rng = np.random.default_rng(0)
        ensemble = [signal + rng.normal(0.0, 0.05, N) for _ in range(50)]

        result = SimulationResult(
            v_nl_qubit=signal,
            v_qubit_ensemble=ensemble,
            fs=fs,
            mode="real_axis",
            carrier_freq_hz=carrier_hz,
            noise_enabled=True,
            n_realizations=50,
        )
        snr_result = pulse_snr(result, lpf_cutoff_hz=200e6)
        assert np.isfinite(snr_result.snr)
        assert snr_result.snr > 0
        assert snr_result.n_window_samples > 0

    def test_rejects_noise_disabled_result(self):
        result, _ = _flat_pulse_result()
        result.noise_enabled = False
        with pytest.raises(ValueError, match="noise enabled"):
            pulse_snr(result)

    def test_rejects_empty_ensemble(self):
        result, _ = _flat_pulse_result()
        result.v_qubit_ensemble = []
        with pytest.raises(ValueError, match="noise enabled"):
            pulse_snr(result)

    def test_rejects_all_zero_signal(self):
        """No samples exceed window_threshold_frac of the peak when the
        signal is identically zero (peak itself is zero) -- should raise
        rather than silently divide by zero or return a bogus window."""
        N = 100
        signal = np.zeros(N)
        ensemble = [np.random.default_rng(1).normal(0, 0.1, N) for _ in range(10)]
        result = SimulationResult(
            v_nl_qubit=signal, v_qubit_ensemble=ensemble, fs=1e9,
            mode="complex_baseband", carrier_freq_hz=5e9, noise_enabled=True, n_realizations=10,
        )
        with pytest.raises(ValueError, match="window_threshold_frac"):
            pulse_snr(result)


if __name__ == "__main__":
    pytest.main([__file__])
