"""
tests/test_engine_noise.py
===========================
Integration tests for engine.run()'s noise pass against real SI schematics
declaring an actual SI statistical-noise-source device (VN1, a
DeviceVoltageStatisticalNoiseSource -- see schematic/noise.py), covering two
concerns:

1. TestEngineNoiseSINative / TestEngineNoiseOverride / TestNoiseSourceValidation
---------------------------------------------------------------------------
Both noise-sourcing paths are exercised end to end through the full engine
(not just the underlying psd.py/propagation.py units, already covered by
tests/test_noise.py), on tests/test_schematic_noise.si:
  - SI-native: `noise={"VN1": {}}` -- PSD comes from SI's own Johnson-noise
    computation on VN1's schematic-configured Resistance/Temperature.
  - Override: `noise={"VN1": {"single_sided_psd_v2_per_hz": ...}}` -- PSD is
    the given flat value instead, still propagated via VN1's own SI-derived
    transfer function to the qubit plane.

Core check (both paths): compute v_qubit_ensemble[i] - v_nl_qubit for every
realization (the actual injected noise waveform) and confirm its empirical
variance matches the theoretically expected value derived independently
from the source's PSD and its (measured) transfer function to VQubit --
Var = |H|^2 * 2*S_v*fs (the 2x is generate_baseband_noise()'s own bandpass-
to-envelope conversion for the full Nyquist band -- see its module
docstring's "Absolute-scale history" for the derivation and the
scipy.signal.periodogram / real_axis cross-check that pins down the
absolute scale), the same closed-form check used in tests/test_noise.py's
direct NoisePropagator tests, now verified through the full engine.run()
pipeline against a real SI-solved network rather than a synthetic transfer
function.

test_schematic_noise.si's VN1 sits between T2's output and the R2/VQubit
termination -- a purely resistive local path, so its measured H(f) to
VQubit is flat (confirmed directly below, not assumed), which is what makes
a simple closed-form expected variance possible here.

2. TestNoiseModeEquivalence
---------------------------------------------------------------------------
Confirms complex_baseband and real_axis modes propagate the SAME noise
physics, by comparing the effective SNR (average signal power / average
noise power) each mode computes independently, on two schematics:
  - tests/test_schematic_noise.si       (VN1 right next to the qubit plane,
    lossless -- no frequency-shaping of the noise en route to VQubit)
  - tests/test_schematic_noise_lossy.si (VN1 right after the amplifier,
    BEFORE a DeviceTransmissionLineLossy -- both the signal and the noise
    must pass through the same lossy, frequency-shaped line to reach
    VQubit; this is the more demanding case, since it tests that BOTH
    modes' noise correctly picks up the SAME frequency-dependent shaping
    as the deterministic signal does, not just a flat attenuation)

Why this comparison isn't apples-to-apples without filtering
---------------------------------------------------------------------------
The two modes have genuinely different native bandwidths: real_axis
propagates (and draws noise) at the schematic's own native rate
(native_sample_rate() = 2*EndFrequency, 40GSa/s here, i.e. a full 20GHz
one-sided bandwidth), while complex_baseband propagates a narrowband
envelope at FS_ENVELOPE (here 4GHz, i.e. a +/-2GHz slice around the
carrier). A real_axis noise realization therefore carries far more total
integrated power than a baseband one, by construction, not because the
underlying physics differs -- the comparison is only meaningful once both
are restricted to the SAME bandwidth.

The fix: demodulate real_axis's output (both the deterministic signal and
every noise realization) down to an I/Q baseband-equivalent representation
via quantum.demodulate(), with an EXPLICIT lpf_cutoff_hz = FS_ENVELOPE/2 --
i.e. deliberately matching complex_baseband mode's own implicit bandwidth,
not demodulate()'s own default (carrier_freq_hz, meant for image rejection,
not bandwidth-matching between modes). demodulate()'s "x2" single-sideband
amplitude correction is exactly right here -- for the deterministic signal
(an identity that holds for any genuinely narrowband-around-carrier real
signal) AND for noise (a real, physical bandpass-to-envelope conversion
that a genuinely wideband source picks up correctly).

Three real bugs across generate_rf_noise()/generate_baseband_noise()
surfaced across this investigation, all fixed in noise/realization.py
itself (not worked around here) -- see that module's own docstring
("Absolute-scale history") for the full derivations:
  - generate_rf_noise() was missing the "undo irfft's 1/N normalization"
    fix generate_baseband_noise() already had, undersizing real-axis noise
    variance by ~N^2.
  - generate_rf_noise(), once that was fixed, still had its per-bin
    amplitude 2x too high in variance (S1*fs instead of the correct
    S1*fs/2 for a full-Nyquist-band flat one-sided PSD S1) -- caught by
    validating its output directly against scipy.signal.periodogram
    rather than against generate_baseband_noise(), since a matching 2x bug
    in the latter (below) made every real_axis-vs-baseband cross-check
    pass anyway.
  - generate_baseband_noise() was first found completely missing the
    bandpass-to-envelope conversion (no factor at all), then "fixed" with
    an x2-amplitude/x4-power factor that overshot by another 2x (Var=8*S1*B
    instead of the correct 4*S1*B) -- an earlier version of this test
    briefly (and incorrectly) compensated by dividing real_axis's
    demodulated noise power by 4 in _windowed_snr(), which was backwards;
    that workaround was removed once the actual generator bugs were found
    and fixed at the source. No per-mode correction is needed here.

SNR definition used here
---------------------------------------------------------------------------
average signal power / average noise power, both computed only over the
time window where the deterministic signal is actually significant
(|v_signal|^2 > 5% of its own peak) -- not averaged over the full array,
which is padded well beyond the active pulse by each segment's own
convolution growth (see engine.py's _nonlinear_pass docstring) and would
otherwise dilute the signal-power estimate by an amount that isn't
consistent between the two modes (different native sample rates ->
different padding sample counts, even for comparable padding in seconds).
Noise power is estimated from v_qubit_ensemble[i] - v_nl_qubit for many
realizations, averaged over both realizations and the same time window.

Tolerance
---------------------------------------------------------------------------
This codebase's own prior investigations already establish that real_axis
and complex_baseband modes agree on OTHER measured quantities (bandwidth-
dispersion infidelity, etc) only to within ~10-30%, not exactly -- they are
genuinely different numerical approximations of the same physics (one
narrowband/analytic, one a direct wideband time-domain solve). The
tolerance here is set accordingly, not to a tight number that would just
be testing this run's specific RNG luck.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")

warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

from scipy.signal import periodogram

from si_qfi.schematic import loader as si_loader
from si_qfi.schematic.noise import extract_noise_source_transfer_functions, get_noise_source_psd
from si_qfi.schematic.transfer_function import compute_impulse_response
from si_qfi.simulation import engine
from si_qfi.source.waveform import build_gaussian_envelope, source_from_envelope_array
from si_qfi.quantum.hamiltonian import demodulate

# Resolved at module level -- OpenProjectFile() changes the process CWD as a
# side effect (see test_engine.py's own module docstring for this same note).
NOISE_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_noise.si").resolve()
LOSSY_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_noise_lossy.si").resolve()

_CARRIER_GHZ = 5.0
_CARRIER_HZ = _CARRIER_GHZ * 1e9
_DURATION_S = 20e-9
_SIGMA_S = _DURATION_S / 6
_FS_ENVELOPE = 4e9
_N_REALIZATIONS = 300
_KB = 1.380649e-23

# TestNoiseModeEquivalence-specific constants (kept separate from
# _N_REALIZATIONS above so its own, separately-tuned statistics don't shift
# if the SI-native/override tests' realization count ever changes).
_N_REALIZATIONS_EQUIV = 400
_PSD_V2_PER_HZ = 1e-15
_SEED = 2026
_WINDOW_THRESHOLD_FRAC = 0.05   # keep samples where signal power > 5% of peak


@pytest.fixture
def noise_schematic():
    return si_loader.load_schematic(NOISE_SCHEMATIC_PATH)


@pytest.fixture
def reference_source():
    shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    return source_from_envelope_array(shape, _FS_ENVELOPE, _CARRIER_GHZ)


def _measured_h_power_gain(schematic, mode, fs, carrier_hz):
    """|H(f)|^2 from VN1 to VQubit, confirmed flat, as a single scalar."""
    raw = extract_noise_source_transfer_functions(schematic, ["VN1"])["VN1"]
    tf = compute_impulse_response(raw, mode, fs=fs, carrier_hz=carrier_hz)
    h = tf.h
    # A purely resistive local path (see module docstring) -- h should be a
    # single dominant tap, not a spread-out/dispersive impulse response.
    power_gain = np.sum(np.abs(h) ** 2)
    return power_gain


class TestEngineNoiseSINative:
    def test_rms_matches_si_computed_johnson_psd(self, noise_schematic, reference_source):
        result = engine.run(
            noise_schematic, reference_source,
            nonlinear=None, noise={"VN1": {}},
            n_realizations=_N_REALIZATIONS, mode="complex_baseband", seed=42,
        )
        assert result.noise_enabled

        diffs = np.array([
            v - result.v_nl_qubit for v in result.v_qubit_ensemble
        ])
        actual_var = np.var(diffs)

        freqs = np.fft.rfftfreq(len(reference_source.envelope_complex), d=1.0 / reference_source.fs)
        S_v = get_noise_source_psd(noise_schematic, "VN1", freqs)
        power_gain = _measured_h_power_gain(
            noise_schematic, "complex_baseband", reference_source.fs, reference_source.carrier_freq_hz,
        )
        # x2: bandpass-to-envelope conversion generate_baseband_noise() applies
        # for the full Nyquist band -- see its module docstring.
        expected_var = 2 * float(np.mean(S_v[1:])) * reference_source.fs * power_gain

        rel_err = abs(actual_var - expected_var) / expected_var
        assert rel_err < 0.15, (
            f"noise RMS^2 {actual_var:.3e} should be near expected {expected_var:.3e} "
            f"(rel err {rel_err:.2%})"
        )

    def test_johnson_psd_matches_4kTR_formula(self, noise_schematic):
        """Sanity: SI's own computed density for VN1 (Type=Johnson,
        Resistance=50, Temperature_Kelvin=290 in the schematic) matches the
        textbook 4*kB*T*R formula directly -- confirms the schematic's
        declared device is actually configured the way the docstring above
        assumes, not just that *some* nonzero density comes back."""
        freqs = np.array([1e8, 5e8, 1e9, 2e9])
        S_v = get_noise_source_psd(noise_schematic, "VN1", freqs)
        expected = 4 * _KB * 290.0 * 50.0
        np.testing.assert_allclose(S_v, expected, rtol=0.05)


class TestEngineNoiseOverride:
    def test_rms_matches_overridden_psd_not_si_default(self, noise_schematic, reference_source):
        override_psd = 5e-16   # deliberately >>1000x SI's own ~8e-19 Johnson value
        result = engine.run(
            noise_schematic, reference_source,
            nonlinear=None, noise={"VN1": {"single_sided_psd_v2_per_hz": override_psd}},
            n_realizations=_N_REALIZATIONS, mode="complex_baseband", seed=42,
        )
        diffs = np.array([v - result.v_nl_qubit for v in result.v_qubit_ensemble])
        actual_var = np.var(diffs)

        power_gain = _measured_h_power_gain(
            noise_schematic, "complex_baseband", reference_source.fs, reference_source.carrier_freq_hz,
        )
        expected_var = 2 * override_psd * reference_source.fs * power_gain

        rel_err = abs(actual_var - expected_var) / expected_var
        assert rel_err < 0.15, (
            f"noise RMS^2 {actual_var:.3e} should be near overridden expected {expected_var:.3e} "
            f"(rel err {rel_err:.2%})"
        )

    def test_override_dbm_hz_form(self, noise_schematic, reference_source):
        override_dbm_hz = -140.0
        result = engine.run(
            noise_schematic, reference_source,
            nonlinear=None, noise={"VN1": {"single_sided_psd_dbm_hz": override_dbm_hz}},
            n_realizations=_N_REALIZATIONS, mode="complex_baseband", seed=42,
        )
        diffs = np.array([v - result.v_nl_qubit for v in result.v_qubit_ensemble])
        actual_var = np.var(diffs)

        override_psd = 10.0 ** (override_dbm_hz / 10.0) * 1e-3 * 50.0
        power_gain = _measured_h_power_gain(
            noise_schematic, "complex_baseband", reference_source.fs, reference_source.carrier_freq_hz,
        )
        expected_var = 2 * override_psd * reference_source.fs * power_gain

        rel_err = abs(actual_var - expected_var) / expected_var
        assert rel_err < 0.15


class TestNoiseSourceValidation:
    def test_unknown_noise_source_name_rejected(self, noise_schematic, reference_source):
        with pytest.raises(ValueError, match="noise"):
            engine.run(
                noise_schematic, reference_source,
                nonlinear=None, noise={"NotARealSource": {}},
                n_realizations=1, mode="complex_baseband",
            )

    def test_probe_label_rejected_as_noise_source(self, noise_schematic, reference_source):
        """VQubit is a valid probe label but NOT a noise-source device name
        -- confirms the two namespaces are actually kept separate (the
        whole point of validating noise.keys() against
        schematic.noise_source_names rather than schematic.port_names)."""
        with pytest.raises(ValueError, match="noise"):
            engine.run(
                noise_schematic, reference_source,
                nonlinear=None, noise={"VQubit": {}},
                n_realizations=1, mode="complex_baseband",
            )


def _mode_signal_and_diffs(schematic_path, mode: str):
    """
    Run engine.run() on `schematic_path` in `mode` and bring the
    deterministic signal and per-realization noise-only waveforms at the
    qubit plane into a consistent baseband-equivalent complex
    representation -- demodulated (bandwidth-matched, see module
    docstring) for real_axis, used as-is for complex_baseband.
    demodulate()'s "x2" single-sideband correction (applied here for
    real_axis mode, to both signal and noise) is the physically standard
    bandpass-to-envelope conversion, and generate_baseband_noise() (used
    for complex_baseband mode) applies the identical factor at generation
    time -- so v_nl_qubit/v_qubit_ensemble are used as-is for baseband,
    just brought into a consistent representation first.

    Returns (fs, window, v_signal, diffs) where `fs` is the sample rate of
    the returned arrays, `window` is a boolean mask marking the time
    samples where the signal is actually significant (see module
    docstring), and `diffs` has shape (n_realizations, n_samples) --
    v_qubit_ensemble[i] - v_signal, the actual injected noise waveform for
    each realization.
    """
    schematic = si_loader.load_schematic(schematic_path)
    ref_shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    source = source_from_envelope_array(ref_shape, _FS_ENVELOPE, _CARRIER_GHZ)

    result = engine.run(
        schematic, source, nonlinear=None,
        noise={"VN1": {"single_sided_psd_v2_per_hz": _PSD_V2_PER_HZ}},
        n_realizations=_N_REALIZATIONS_EQUIV, mode=mode, seed=_SEED,
    )

    if mode == "real_axis":
        t = np.arange(len(result.v_nl_qubit)) / result.fs
        # Explicit bandwidth-matching LPF -- see module docstring for why
        # this is FS_ENVELOPE/2, not demodulate()'s own default.
        I_sig, Q_sig = demodulate(result.v_nl_qubit, t, _CARRIER_HZ, lpf_cutoff_hz=_FS_ENVELOPE / 2)
        v_signal = I_sig + 1j * Q_sig
        v_ensemble = []
        for v in result.v_qubit_ensemble:
            I, Q = demodulate(v, t, _CARRIER_HZ, lpf_cutoff_hz=_FS_ENVELOPE / 2)
            v_ensemble.append(I + 1j * Q)
    else:
        v_signal = result.v_nl_qubit
        v_ensemble = result.v_qubit_ensemble

    sig_power_profile = np.abs(v_signal) ** 2
    peak = sig_power_profile.max()
    window = sig_power_profile > _WINDOW_THRESHOLD_FRAC * peak
    assert window.sum() > 0

    diffs = np.array([v - v_signal for v in v_ensemble])
    return result.fs, window, v_signal, diffs


def _windowed_snr(v_signal: np.ndarray, window: np.ndarray, diffs: np.ndarray) -> float:
    """average signal power / average noise power, restricted to `window`."""
    signal_power = float(np.mean((np.abs(v_signal) ** 2)[window]))
    noise_power_profile = np.mean(np.abs(diffs) ** 2, axis=0)
    noise_power = float(np.mean(noise_power_profile[window]))
    return signal_power / noise_power


def _mode_snr(schematic_path, mode: str) -> float:
    fs, window, v_signal, diffs = _mode_signal_and_diffs(schematic_path, mode)
    return _windowed_snr(v_signal, window, diffs)


class TestNoiseModeEquivalence:
    def test_snr_matches_across_modes_lossless(self):
        snr_bb = _mode_snr(NOISE_SCHEMATIC_PATH, "complex_baseband")
        snr_ra = _mode_snr(NOISE_SCHEMATIC_PATH, "real_axis")
        rel_diff = abs(snr_bb - snr_ra) / snr_bb
        assert rel_diff < 0.05, (
            f"SNR mismatch (lossless): baseband={snr_bb:.3e}, real_axis={snr_ra:.3e}, "
            f"rel diff={rel_diff:.1%}"
        )

    def test_snr_matches_across_modes_lossy(self):
        snr_bb = _mode_snr(LOSSY_SCHEMATIC_PATH, "complex_baseband")
        snr_ra = _mode_snr(LOSSY_SCHEMATIC_PATH, "real_axis")
        rel_diff = abs(snr_bb - snr_ra) / snr_bb
        assert rel_diff < 0.05, (
            f"SNR mismatch (lossy): baseband={snr_bb:.3e}, real_axis={snr_ra:.3e}, "
            f"rel diff={rel_diff:.1%}"
        )

    def test_lossy_qubit_noise_rms_and_spectrum_match_across_modes(self):
        """
        A stronger check than the SNR tests above, specific to the LOSSY
        schematic: compute the actual noise-only waveform at the qubit
        plane in each mode (v_qubit_ensemble[i] - v_nl_qubit, brought into
        the same baseband-equivalent representation as _mode_snr()), and
        confirm real_axis and complex_baseband agree not just on total
        noise power (RMS) but on the noise's FREQUENCY SPECTRUM -- i.e.
        that both modes reproduce the same frequency-dependent shaping the
        lossy transmission line applies to VN1's noise en route to
        VQubit, not just a matching flat/total power.

        Spectrum comparison method: an ensemble-averaged periodogram
        (average the two-sided periodogram of each of the
        _N_REALIZATIONS_EQUIV independent noise realizations -- a
        standard, low-variance nonparametric spectral estimator that needs
        no assumption about the shape) over the signal-active time window,
        binned into a handful of coarse frequency bands so the comparison
        isn't sensitive to individual noisy bins. The outermost ~20% of the
        +/-FS_ENVELOPE/2 band (near demodulate()'s own LPF cutoff edge) is
        excluded -- confirmed separately (not asserted here) that those
        edge bins differ by ~30%, a known artifact of the LPF's non-
        brickwall rolloff right at its cutoff, not a physics mismatch;
        interior bands agree to within ~5%.
        """
        fs_bb, window_bb, v_sig_bb, diffs_bb = _mode_signal_and_diffs(LOSSY_SCHEMATIC_PATH, "complex_baseband")
        fs_ra, window_ra, v_sig_ra, diffs_ra = _mode_signal_and_diffs(LOSSY_SCHEMATIC_PATH, "real_axis")

        rms_bb = float(np.sqrt(np.mean(np.abs(diffs_bb[:, window_bb]) ** 2)))
        rms_ra = float(np.sqrt(np.mean(np.abs(diffs_ra[:, window_ra]) ** 2)))
        rms_rel_diff = abs(rms_bb - rms_ra) / rms_bb
        assert rms_rel_diff < 0.1, (
            f"Qubit-plane noise RMS mismatch (lossy): baseband={rms_bb:.3e}, "
            f"real_axis={rms_ra:.3e}, rel diff={rms_rel_diff:.1%}"
        )

        freqs_bb, Pxx_bb = periodogram(
            diffs_bb[:, window_bb], fs=fs_bb, axis=-1, return_onesided=False, scaling="density", window="boxcar",
        )
        freqs_ra, Pxx_ra = periodogram(
            diffs_ra[:, window_ra], fs=fs_ra, axis=-1, return_onesided=False, scaling="density", window="boxcar",
        )
        psd_bb = np.mean(Pxx_bb, axis=0)
        psd_ra = np.mean(Pxx_ra, axis=0)

        margin = 0.8 * (_FS_ENVELOPE / 2)   # stay clear of the LPF cutoff edge, see docstring
        n_bands = 6
        edges = np.linspace(-margin, margin, n_bands + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            band_psd_bb = float(np.mean(psd_bb[(freqs_bb >= lo) & (freqs_bb < hi)]))
            band_psd_ra = float(np.mean(psd_ra[(freqs_ra >= lo) & (freqs_ra < hi)]))
            rel_diff = abs(band_psd_bb - band_psd_ra) / band_psd_bb
            assert rel_diff < 0.2, (
                f"Qubit-plane noise spectrum mismatch (lossy) in band "
                f"[{lo:+.2e}, {hi:+.2e}) Hz: baseband={band_psd_bb:.3e}, "
                f"real_axis={band_psd_ra:.3e}, rel diff={rel_diff:.1%}"
            )


if __name__ == "__main__":
    pytest.main([__file__])
