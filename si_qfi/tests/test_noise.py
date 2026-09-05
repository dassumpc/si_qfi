"""
tests/test_noise.py
====================
Unit tests for noise PSD computation and stochastic realization generation.
Verifiable without SignalIntegrity or QuTiP installed.
"""

import numpy as np
import pytest
from scipy.signal import periodogram


# ---------------------------------------------------------------------------
# Noise tests
# ---------------------------------------------------------------------------

class TestNoise:

    def test_baseband_noise_variance(self):
        """
        Generated noise variance should be 2x the naive "PSD * fs" integral
        for a flat two-sided input covering the full Nyquist band -- the
        bandpass-to-complex-envelope conversion factor a physical (wideband)
        one-sided PSD picks up when represented as a complex envelope
        (Var = 4*S_v*B for one-sided cutoff B, = 2*S_v*fs at B=fs/2),
        matching what real_axis mode's noise gets via demodulating an
        equally-corrected generate_rf_noise() realization for the identical
        physical source -- see generate_rf_noise()/generate_baseband_noise()'s
        own module docstring ("Absolute-scale history") for the full
        derivation, confirmed against scipy.signal.periodogram and a direct
        real_axis-vs-baseband cross-check.
        """
        from si_qfi.noise.realization import generate_baseband_noise
        N = 8192
        fs = 1e9
        target_psd = 1e-12   # V²/Hz
        psd_array = np.full(N, target_psd)
        rng = np.random.default_rng(42)
        noise = generate_baseband_noise(N, fs, psd_array, rng=rng)
        expected_var = 2 * target_psd * fs
        actual_var = np.var(noise)
        # Allow 20% tolerance (statistical)
        assert abs(actual_var - expected_var) / expected_var < 0.2, (
            f"Noise variance {actual_var:.3e} should be near {expected_var:.3e}"
        )

        # Direct check against scipy.signal.periodogram (an independent
        # implementation) -- this is the check that actually caught this
        # function's absolute-scale bug (see noise/realization.py's
        # "Absolute-scale history"): the Var-based check above passed even
        # when this function was 2x too strong, because generate_rf_noise
        # had an independent, exactly-compensating 2x error of its own, so
        # only validating against an outside reference (not the sibling
        # function) exposed it. For complex input, scipy always returns a
        # two-sided spectrum; the measured density should be 2x the input
        # psd value (the bandpass-to-envelope factor baked into amp on top
        # of the "psd_two_sided" input array).
        freqs, Pxx = periodogram(noise, fs=fs, scaling="density", return_onesided=False, window="boxcar")
        measured_psd = float(np.mean(Pxx))
        expected_psd = 2 * target_psd
        assert abs(measured_psd - expected_psd) / expected_psd < 0.1, (
            f"scipy-measured PSD {measured_psd:.3e} should be near {expected_psd:.3e}"
        )

    def test_rf_noise_is_real(self):
        """RF noise should be real-valued."""
        from si_qfi.noise.realization import generate_rf_noise
        N = 1024
        fs = 20e9
        psd = np.ones(N // 2 + 1) * 1e-12
        rng = np.random.default_rng(0)
        noise = generate_rf_noise(N, fs, psd, rng=rng)
        assert noise.dtype == np.float64
        assert np.all(np.isreal(noise))

    def test_rf_noise_variance(self):
        """Generated real-axis noise variance should match the plain
        Johnson-Nyquist "noise power in bandwidth" formula Var = S_v * fs/2
        for a flat one-sided PSD over the full Nyquist band [0,fs/2] --
        confirmed directly against scipy.signal.periodogram (an independent
        implementation) on this function's own output, see generate_rf_noise's
        own docstring/module docstring for the full derivation. (This was
        previously Var=S_v*fs, 2x too high -- an error that passed every
        real_axis-vs-baseband cross-check because generate_baseband_noise had
        an independent, exactly-compensating 2x bug of its own; only caught
        by validating against scipy directly rather than against the other
        function.)"""
        from si_qfi.noise.realization import generate_rf_noise
        N = 200_000
        fs = 40e9
        target_psd = 1e-15   # V²/Hz
        psd = np.full(N // 2 + 1, target_psd)
        rng = np.random.default_rng(42)
        noise = generate_rf_noise(N, fs, psd, rng=rng)
        expected_var = target_psd * fs / 2
        actual_var = np.var(noise)
        assert abs(actual_var - expected_var) / expected_var < 0.05, (
            f"Noise variance {actual_var:.3e} should be near {expected_var:.3e}"
        )

        # Direct check against scipy.signal.periodogram (an independent
        # implementation) -- the check that actually caught this function's
        # absolute-scale bug in the first place (see noise/realization.py's
        # "Absolute-scale history"): the Var-based check above alone had
        # already passed with the OLD, 2x-too-high formula, because it only
        # compares against a hand-derived formula, not an outside reference.
        # For a real signal, scipy's one-sided density should read back the
        # target PSD directly (no extra factor -- generate_rf_noise applies
        # no bandpass-to-envelope conversion, unlike generate_baseband_noise).
        freqs, Pxx = periodogram(noise, fs=fs, scaling="density", return_onesided=True, window="boxcar")
        measured_psd = float(np.mean(Pxx[1:-1]))   # exclude DC/Nyquist (half-weighted bins)
        assert abs(measured_psd - target_psd) / target_psd < 0.05, (
            f"scipy-measured PSD {measured_psd:.3e} should match target {target_psd:.3e}"
        )

    def test_real_axis_demodulated_matches_baseband_directly(self):
        """
        Direct, schematic-free equivalence check between the two generation
        paths: generate RF (real_axis) noise, demodulate it down to a
        complex baseband-equivalent representation via quantum.demodulate(),
        and confirm its variance matches generate_baseband_noise()'s own
        direct output for the identical physical one-sided PSD S1 and
        one-sided bandwidth B -- i.e. does the same physical noise source
        give the same answer through either generation path?

        This is the check that originally exposed both absolute-scale bugs
        described in noise/realization.py's "Absolute-scale history"
        (promoted here from a one-off scratch script into a permanent
        regression test, so a future change to either function that
        reintroduces a scale mismatch -- compensating or not -- gets caught
        immediately). tests/test_noise_mode_equivalence.py covers the same
        relationship at a higher level (through a real SI schematic +
        engine.run()); this is the minimal, no-SI-needed version of the
        same check, isolating just the two generator functions plus
        demodulate().
        """
        from si_qfi.noise.realization import generate_rf_noise, generate_baseband_noise
        from si_qfi.quantum.hamiltonian import demodulate

        fc = 5e9
        B = 0.5e9
        S1 = 3e-15   # V²/Hz, physical one-sided PSD
        N_REAL = 300

        # real_axis: draw white noise over a native band comfortably above
        # fc+B, then demodulate+filter down to the target bandwidth.
        fs_rf = 20e9
        N_rf = 65536
        t = np.arange(N_rf) / fs_rf
        freqs_rf = np.fft.rfftfreq(N_rf, d=1.0 / fs_rf)
        psd_rf = np.full_like(freqs_rf, S1)

        rng = np.random.default_rng(2026)
        envelopes = []
        for _ in range(N_REAL):
            v_rf = generate_rf_noise(N_rf, fs_rf, psd_rf, rng=rng)
            I, Q = demodulate(v_rf, t, fc, lpf_cutoff_hz=B)
            envelopes.append(I + 1j * Q)
        envelopes = np.array(envelopes)
        # Drop filter-transient edges before measuring (a few filter time
        # constants at each end).
        edge = N_rf // 10
        var_real_axis = np.var(envelopes[:, edge:-edge])

        # baseband: direct generation with the SAME physical S1 and B.
        fs_bb = 8 * B   # oversample comfortably above the target bandwidth
        N_bb = 65536
        freqs_bb = np.fft.rfftfreq(N_bb, d=1.0 / fs_bb)
        psd_bb = np.where(freqs_bb <= B, S1, 0.0)
        rng2 = np.random.default_rng(2026)
        baseband_realizations = np.array([
            generate_baseband_noise(N_bb, fs_bb, psd_bb, rng=rng2) for _ in range(N_REAL)
        ])
        var_baseband = np.var(baseband_realizations)

        rel_diff = abs(var_real_axis - var_baseband) / var_baseband
        assert rel_diff < 0.15, (
            f"real_axis (demodulated) Var={var_real_axis:.3e} should match "
            f"baseband (direct) Var={var_baseband:.3e} for the same physical "
            f"S1={S1:.3e}, B={B:.3e} (rel diff {rel_diff:.1%})"
        )

    def test_baseband_noise_is_complex(self):
        """Baseband noise should be complex-valued."""
        from si_qfi.noise.realization import generate_baseband_noise
        N = 512
        fs = 500e6
        psd = np.ones(N) * 1e-12
        rng = np.random.default_rng(1)
        noise = generate_baseband_noise(N, fs, psd, rng=rng)
        assert np.iscomplexobj(noise)

    def test_psd_override_v2_per_hz(self):
        """Direct V²/Hz override should pass through unchanged, flat over freqs."""
        from si_qfi.noise.psd import psd_from_override
        freqs = np.array([1e9, 5e9, 10e9])
        psd = psd_from_override({"single_sided_psd_v2_per_hz": 1e-18}, freqs)
        np.testing.assert_allclose(psd, 1e-18, rtol=1e-12)

    def test_psd_override_dbm_hz(self):
        """dBm/Hz override should convert to V²/Hz into 50 ohm correctly."""
        from si_qfi.noise.psd import psd_from_override
        freqs = np.array([1e9, 5e9])
        psd_dbm_hz = -174.0   # a textbook "kT at room temp"-ish reference level
        psd = psd_from_override({"single_sided_psd_dbm_hz": psd_dbm_hz}, freqs)
        expected = 10.0 ** (psd_dbm_hz / 10.0) * 1e-3 * 50.0
        np.testing.assert_allclose(psd, expected, rtol=1e-12)

    def test_psd_override_type_thermal(self):
        """type='thermal' should give plain Johnson-Nyquist S_v=4*kB*T*R,
        matching the SAME 4kTR convention SI's own native computation uses
        (see tests/test_engine_noise.py's test_johnson_psd_matches_4kTR_formula) --
        NOT the PRD's un-factored kB*T*R (see noise/psd.py's module docstring)."""
        from si_qfi.noise.psd import psd_from_override
        _KB = 1.380649e-23
        freqs = np.array([1e8, 1e9])
        psd = psd_from_override({"type": "thermal", "temperature_k": 4.0}, freqs)
        expected = 4 * _KB * 4.0 * 50.0   # default resistance 50 ohm
        np.testing.assert_allclose(psd, expected, rtol=1e-12)

    def test_psd_override_type_thermal_custom_resistance(self):
        from si_qfi.noise.psd import psd_from_override
        _KB = 1.380649e-23
        freqs = np.array([1e8])
        psd = psd_from_override(
            {"type": "thermal", "temperature_k": 290.0, "resistance_ohms": 75.0}, freqs,
        )
        expected = 4 * _KB * 290.0 * 75.0
        np.testing.assert_allclose(psd, expected, rtol=1e-12)

    def test_psd_override_type_thermal_missing_temperature_rejected(self):
        from si_qfi.noise.psd import psd_from_override
        with pytest.raises(ValueError, match="temperature_k"):
            psd_from_override({"type": "thermal"}, np.array([1e8]))

    def test_psd_override_type_noise_figure_default_reference(self):
        """type='noise_figure' with no explicit temperature_k should use the
        standard IEEE 290K noise-figure reference temperature, NOT the
        component's physical temperature (there isn't one specified here at
        all -- see psd_from_override()'s docstring for why 290K, not a
        physical bath temperature, is the correct default)."""
        from si_qfi.noise.psd import psd_from_override
        _KB = 1.380649e-23
        freqs = np.array([1e9])
        nf_db = 3.0
        psd = psd_from_override({"type": "noise_figure", "noise_figure_db": nf_db}, freqs)
        t_excess = 290.0 * (10.0 ** (nf_db / 10.0) - 1.0)
        expected = 4 * _KB * t_excess * 50.0
        np.testing.assert_allclose(psd, expected, rtol=1e-12)

    def test_psd_override_type_noise_figure_custom_reference_and_resistance(self):
        from si_qfi.noise.psd import psd_from_override
        _KB = 1.380649e-23
        freqs = np.array([1e9])
        nf_db, t_ref, r = 6.0, 300.0, 50.0
        psd = psd_from_override(
            {"type": "noise_figure", "noise_figure_db": nf_db, "temperature_k": t_ref, "resistance_ohms": r},
            freqs,
        )
        t_excess = t_ref * (10.0 ** (nf_db / 10.0) - 1.0)
        expected = 4 * _KB * t_excess * r
        np.testing.assert_allclose(psd, expected, rtol=1e-12)

    def test_psd_override_type_noise_figure_zero_db_gives_zero_excess_noise(self):
        """0 dB noise figure means the component is noiseless (adds nothing
        beyond what's already modeled elsewhere) -- a good sanity check on
        the formula's zero point."""
        from si_qfi.noise.psd import psd_from_override
        freqs = np.array([1e9])
        psd = psd_from_override({"type": "noise_figure", "noise_figure_db": 0.0}, freqs)
        np.testing.assert_allclose(psd, 0.0, atol=1e-30)

    def test_psd_override_type_noise_figure_missing_key_rejected(self):
        from si_qfi.noise.psd import psd_from_override
        with pytest.raises(ValueError, match="noise_figure_db"):
            psd_from_override({"type": "noise_figure"}, np.array([1e9]))

    def test_psd_override_type_noise_density_alias(self):
        """type='noise_density' should be exactly equivalent to a plain
        single_sided_psd_v2_per_hz override -- a documentary/explicit-type
        spelling of the same thing (PRD §7.1's example includes a
        'temperature_k' alongside it too, but that's documentary only since
        the PSD is given directly, not derived)."""
        from si_qfi.noise.psd import psd_from_override
        freqs = np.array([1e9, 2e9])
        psd_direct = psd_from_override({"single_sided_psd_v2_per_hz": 1.6e-20}, freqs)
        psd_typed = psd_from_override(
            {"type": "noise_density", "single_sided_psd_v2_per_hz": 1.6e-20, "temperature_k": 0.02}, freqs,
        )
        np.testing.assert_allclose(psd_typed, psd_direct)

    def test_psd_override_type_noise_density_missing_psd_rejected(self):
        from si_qfi.noise.psd import psd_from_override
        with pytest.raises(ValueError, match="single_sided_psd_v2_per_hz"):
            psd_from_override({"type": "noise_density"}, np.array([1e9]))

    def test_psd_override_unknown_type_rejected(self):
        from si_qfi.noise.psd import psd_from_override
        with pytest.raises(ValueError, match="Unknown noise override type"):
            psd_from_override({"type": "shot"}, np.array([1e9]))

    # -----------------------------------------------------------------
    # Phase-noise PSD spec (noise/psd.py's phase_noise_psd_from_spec) --
    # used by engine.run(phase_noise={...}), see simulation/engine.py and
    # examples/phase_noise_case_study_demo.py.
    # -----------------------------------------------------------------

    def test_phase_noise_requires_bandwidth_hz(self):
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        with pytest.raises(ValueError, match="bandwidth_hz"):
            phase_noise_psd_from_spec({"single_sided_psd_rad2_per_hz": 1e-12}, np.array([1e6]))

    def test_phase_noise_bandwidth_hz_must_be_positive(self):
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        with pytest.raises(ValueError, match="positive"):
            phase_noise_psd_from_spec(
                {"single_sided_psd_rad2_per_hz": 1e-12, "bandwidth_hz": -1.0}, np.array([1e6]),
            )

    def test_phase_noise_flat_raw_psd_bandwidth_cut(self):
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        freqs = np.array([0.0, 10e6, 40e6, 60e6, 100e6])
        psd = phase_noise_psd_from_spec(
            {"single_sided_psd_rad2_per_hz": 1e-12, "bandwidth_hz": 50e6}, freqs,
        )
        np.testing.assert_allclose(psd, [1e-12, 1e-12, 1e-12, 0.0, 0.0])

    def test_phase_noise_callable_raw_psd(self):
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        freqs = np.array([0.0, 1e6, 5e6])
        shape_fn = lambda f: 1e-13 * np.exp(-(f / 2e6) ** 2)
        psd = phase_noise_psd_from_spec(
            {"single_sided_psd_rad2_per_hz": shape_fn, "bandwidth_hz": 50e6}, freqs,
        )
        np.testing.assert_allclose(psd, shape_fn(freqs))

    def test_phase_noise_dbc_hz_conversion(self):
        """S_phi(f) = 2 * 10**(L(f)/10) -- the standard IEEE 1139 small-angle
        relation."""
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        freqs = np.array([1e6, 10e6])
        dbc_fn = lambda f: -120.0 - 20.0 * np.log10(f / 1e6)   # a simple 1/f-ish curve
        psd = phase_noise_psd_from_spec({"dbc_hz": dbc_fn, "bandwidth_hz": 50e6}, freqs)
        expected = 2.0 * 10.0 ** (dbc_fn(freqs) / 10.0)
        np.testing.assert_allclose(psd, expected)

    def test_phase_noise_dbc_hz_flat_number_rejected(self):
        """A flat dBc/Hz number is physically wrong for phase noise (real
        L(f) is essentially never flat) -- dbc_hz must be a callable."""
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        with pytest.raises(ValueError, match="callable"):
            phase_noise_psd_from_spec({"dbc_hz": -140.0, "bandwidth_hz": 50e6}, np.array([1e6]))

    def test_phase_noise_both_keys_rejected(self):
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        with pytest.raises(ValueError, match="exactly one"):
            phase_noise_psd_from_spec(
                {"single_sided_psd_rad2_per_hz": 1e-12, "dbc_hz": lambda f: -140.0 * np.ones_like(f),
                 "bandwidth_hz": 50e6},
                np.array([1e6]),
            )

    def test_phase_noise_neither_key_rejected(self):
        from si_qfi.noise.psd import phase_noise_psd_from_spec
        with pytest.raises(ValueError, match="single_sided_psd_rad2_per_hz.*dbc_hz"):
            phase_noise_psd_from_spec({"bandwidth_hz": 50e6}, np.array([1e6]))

    def test_generate_phase_noise_is_real_and_matches_generate_rf_noise(self):
        """generate_phase_noise() is an explicitly-named wrapper around
        generate_rf_noise() (same math, different physical units/caller
        contract, see its own docstring) -- confirm it actually delegates,
        not a reimplementation that could silently drift."""
        from si_qfi.noise.realization import generate_phase_noise, generate_rf_noise
        N, fs = 2048, 1e9
        psd = np.full(N // 2 + 1, 1e-12)
        phi = generate_phase_noise(N, fs, psd, rng=np.random.default_rng(3))
        expected = generate_rf_noise(N, fs, psd, rng=np.random.default_rng(3))
        assert phi.dtype == np.float64
        np.testing.assert_allclose(phi, expected)

    def test_psd_override_callable_colored_shape(self):
        """single_sided_psd_v2_per_hz also accepts a callable freqs->S_v(freqs)
        for a colored (non-flat) PSD -- used by
        examples/noise_filter_function_demo.py to inject quasi-static/
        narrowband noise sources."""
        from si_qfi.noise.psd import psd_from_override
        freqs = np.array([0.0, 1e6, 5e6, 1e7])
        shape_fn = lambda f: 1e-15 * np.exp(-(f / 2e6) ** 2)   # Gaussian-in-frequency, peaked at DC
        psd = psd_from_override({"single_sided_psd_v2_per_hz": shape_fn}, freqs)
        np.testing.assert_allclose(psd, shape_fn(freqs))
        assert psd[0] > psd[-1]   # peaked at DC, decays with frequency

    def test_psd_override_callable_wrong_shape_rejected(self):
        from si_qfi.noise.psd import psd_from_override
        freqs = np.array([1e6, 2e6, 3e6])
        with pytest.raises(ValueError, match="shape"):
            psd_from_override({"single_sided_psd_v2_per_hz": lambda f: np.array([1.0, 2.0])}, freqs)

    def test_psd_override_callable_negative_rejected(self):
        from si_qfi.noise.psd import psd_from_override
        freqs = np.array([1e6, 2e6, 3e6])
        with pytest.raises(ValueError, match="negative"):
            psd_from_override({"single_sided_psd_v2_per_hz": lambda f: -np.ones_like(f)}, freqs)

    def test_noise_propagator_realization_rms_matches_psd(self):
        """
        NoisePropagator.generate_realization()'s output variance should match
        the target PSD scaled by the propagation transfer function's power
        gain |H|^2 -- the direct, no-SI-needed check that turning a PSD into
        an actual realized waveform (spectral shaping -> IFFT -> convolve)
        preserves the expected statistics, using a synthetic flat PSD and a
        purely-scaling (delta-like) impulse response so the expected
        variance has a simple closed form: Var = |H|^2 * 2*S_v*fs (matching
        test_baseband_noise_variance's own bandpass-to-envelope-corrected
        convention, generalized by the constant propagation gain).
        """
        from si_qfi.noise.propagation import NoisePropagator
        N = 8192
        fs = 1e9
        target_psd = 1e-18   # V^2/Hz
        h_scale = 0.5        # a pure scaling (delta) impulse response -- no dispersion
        psd_cache = {"VN1": np.full(N // 2 + 1, target_psd)}
        h_to_qubit = {"VN1": np.array([h_scale])}

        propagator = NoisePropagator(
            psd_cache=psd_cache, transfer_functions_to_qubit=h_to_qubit,
            n_samples=N, fs=fs, mode="complex_baseband",
        )
        rng = np.random.default_rng(7)
        realizations = np.array([propagator.generate_realization(rng) for _ in range(50)])

        expected_var = (h_scale ** 2) * 2 * target_psd * fs
        actual_var = np.var(realizations)
        assert abs(actual_var - expected_var) / expected_var < 0.15, (
            f"Propagated noise variance {actual_var:.3e} should be near {expected_var:.3e}"
        )

    def test_noise_propagator_sums_multiple_sources(self):
        """Two independent noise sources should add in variance (uncorrelated
        power sum), matching the module docstring's stated superposition."""
        from si_qfi.noise.propagation import NoisePropagator
        N = 8192
        fs = 1e9
        psd_cache = {
            "VN1": np.full(N // 2 + 1, 1e-18),
            "VN2": np.full(N // 2 + 1, 3e-18),
        }
        h_to_qubit = {"VN1": np.array([1.0]), "VN2": np.array([1.0])}
        propagator = NoisePropagator(
            psd_cache=psd_cache, transfer_functions_to_qubit=h_to_qubit,
            n_samples=N, fs=fs, mode="complex_baseband",
        )
        rng = np.random.default_rng(11)
        realizations = np.array([propagator.generate_realization(rng) for _ in range(50)])
        expected_var = 2 * (1e-18 + 3e-18) * fs
        actual_var = np.var(realizations)
        assert abs(actual_var - expected_var) / expected_var < 0.15


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
