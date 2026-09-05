"""
tests/test_engine_phase_noise.py
==================================
Integration tests for engine.run()'s phase_noise= parameter (LO/oscillator
phase noise) -- see noise/psd.py's phase_noise_psd_from_spec() and
simulation/engine.py's module docstring for the full design, and
examples/phase_noise_case_study_demo.py for the worked physics case study.

Core architectural point this file exists to verify directly, not just
assert: phase noise is injected at the SOURCE, before the nonlinear pass,
so engine.run() re-runs _nonlinear_pass() once per Monte Carlo realization
when phase_noise is enabled (rather than once total, as it does for plain
additive noise). test_phase_noise_prenl_differs_from_posthoc_with_
nonlinearity is the test that actually proves this matters: it compares the
real implementation against the (physically wrong, once real nonlinearity
sits between the LO and the qubit plane) alternative of rotating the
SHARED deterministic v_nl_qubit by a phase draw after the fact, and
confirms they give measurably different ensemble statistics.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")

warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope, source_from_envelope_array
from si_qfi.simulation import engine
from si_qfi.noise.psd import phase_noise_psd_from_spec
from si_qfi.noise.realization import generate_phase_noise

BASIC_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_basic.si").resolve()
NOISE_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_noise.si").resolve()
NL_LABEL = "DriverOutput"

_CARRIER_GHZ = 5.0
_DURATION_S = 20e-9
_SIGMA_S = _DURATION_S / 6
_FS_ENVELOPE = 4e9
_PHASE_NOISE_PSD = 5e-11   # rad^2/Hz, deliberately large -- makes the effect easy to measure
_PHASE_NOISE_BW = 200e6
_SEED = 7


@pytest.fixture
def basic_schematic():
    return si_loader.load_schematic(BASIC_SCHEMATIC_PATH)


@pytest.fixture
def noise_schematic():
    return si_loader.load_schematic(NOISE_SCHEMATIC_PATH)


@pytest.fixture
def reference_source():
    shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=2.5)
    return source_from_envelope_array(shape, _FS_ENVELOPE, _CARRIER_GHZ)


class TestPhaseNoiseBasics:
    def test_default_none_behaves_exactly_as_before(self, basic_schematic, reference_source):
        """phase_noise=None (the default) must be indistinguishable from the
        parameter not existing at all -- bit-identical v_nl_qubit and a
        single-element (non-ensemble) v_qubit_ensemble, same as pre-feature
        behavior."""
        result = engine.run(
            basic_schematic, reference_source, nonlinear=None, noise=None,
            n_realizations=5, mode="complex_baseband",
        )
        assert result.noise_enabled is False
        assert result.phase_noise_enabled is False
        assert len(result.v_qubit_ensemble) == 1
        np.testing.assert_array_equal(result.v_qubit_ensemble[0], result.v_nl_qubit)

    def test_baseband_ensemble_varies(self, basic_schematic, reference_source):
        result = engine.run(
            basic_schematic, reference_source, nonlinear=None, noise=None,
            phase_noise={"single_sided_psd_rad2_per_hz": _PHASE_NOISE_PSD, "bandwidth_hz": _PHASE_NOISE_BW},
            n_realizations=20, mode="complex_baseband", seed=_SEED,
        )
        assert result.noise_enabled is True
        assert result.phase_noise_enabled is True
        assert len(result.v_qubit_ensemble) == 20
        diffs = np.array([v - result.v_nl_qubit for v in result.v_qubit_ensemble])
        assert np.var(diffs) > 0
        assert all(len(v) == len(result.v_nl_qubit) for v in result.v_qubit_ensemble)

    def test_real_axis_ensemble_varies_and_is_real(self, basic_schematic, reference_source):
        result = engine.run(
            basic_schematic, reference_source, nonlinear=None, noise=None,
            phase_noise={"single_sided_psd_rad2_per_hz": _PHASE_NOISE_PSD, "bandwidth_hz": _PHASE_NOISE_BW},
            n_realizations=10, mode="real_axis", seed=_SEED,
        )
        diffs = np.array([v - result.v_nl_qubit for v in result.v_qubit_ensemble])
        assert np.var(diffs) > 0
        assert result.v_qubit_ensemble[0].dtype == np.float64

    def test_bandwidth_exceeding_nyquist_is_clipped_with_warning(self, basic_schematic, reference_source):
        """complex_baseband mode's envelope Nyquist (fs/2 = 2GHz here) can't
        represent a wider phase-noise bandwidth -- must warn and clip, not
        silently alias or raise."""
        with pytest.warns(UserWarning, match="bandwidth_hz.*clipped"):
            result = engine.run(
                basic_schematic, reference_source, nonlinear=None, noise=None,
                phase_noise={"single_sided_psd_rad2_per_hz": _PHASE_NOISE_PSD, "bandwidth_hz": 10e9},
                n_realizations=5, mode="complex_baseband", seed=_SEED,
            )
        assert len(result.v_qubit_ensemble) == 5

    def test_combines_with_additive_noise(self, noise_schematic, reference_source):
        """Phase noise and additive (VN1-style) noise are independent and
        should stack -- combined ensemble variance strictly exceeds either
        alone."""
        kwargs = dict(
            schematic=noise_schematic, source=reference_source, nonlinear=None,
            n_realizations=200, mode="complex_baseband", seed=_SEED,
        )
        result_phase_only = engine.run(
            **kwargs, noise=None,
            phase_noise={"single_sided_psd_rad2_per_hz": _PHASE_NOISE_PSD, "bandwidth_hz": _PHASE_NOISE_BW},
        )
        result_combined = engine.run(
            **kwargs, noise={"VN1": {"single_sided_psd_v2_per_hz": 1e-15}},
            phase_noise={"single_sided_psd_rad2_per_hz": _PHASE_NOISE_PSD, "bandwidth_hz": _PHASE_NOISE_BW},
        )
        var_phase_only = np.var([v - result_phase_only.v_nl_qubit for v in result_phase_only.v_qubit_ensemble])
        var_combined = np.var([v - result_combined.v_nl_qubit for v in result_combined.v_qubit_ensemble])
        assert var_combined > var_phase_only


class TestPhaseNoisePreNLvsPostHoc:
    def test_prenl_differs_from_posthoc_with_nonlinearity(self, basic_schematic, reference_source):
        """The central architectural claim behind this feature: once a real
        nonlinearity sits between the LO and the qubit plane, adding phase
        noise POST-HOC (rotating the single shared v_nl_qubit by a phase
        draw after the nonlinear pass has already run once) is physically
        wrong, and genuinely gives a different ensemble than injecting the
        SAME phase draws at the SOURCE and re-running the nonlinear pass
        per realization (what engine.run(phase_noise=...) actually does).
        Confirmed directly here, not just asserted -- with matching seeds
        (so both paths draw the IDENTICAL phase realizations), the two
        ensemble variances differ by double digits of percent for a
        compressing Saleh amplifier."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # SalehModel overdrive warning, expected/benign here

            nl_spec = {NL_LABEL: {"model": "saleh", "op1db_amplitude": 1.0}}

            # (a) correct: pre-NL phase noise via the real implementation.
            result_pre = engine.run(
                basic_schematic, reference_source, nonlinear=nl_spec, noise=None,
                phase_noise={"single_sided_psd_rad2_per_hz": _PHASE_NOISE_PSD, "bandwidth_hz": _PHASE_NOISE_BW},
                n_realizations=300, mode="complex_baseband", seed=_SEED,
            )
            var_pre = np.var([v - result_pre.v_nl_qubit for v in result_pre.v_qubit_ensemble])

            # (b) wrong (for comparison): rotate the single shared v_nl_qubit
            # by the SAME phase draws, applied AFTER the (already-run-once)
            # nonlinear pass instead of before it.
            result_det = engine.run(
                basic_schematic, reference_source, nonlinear=nl_spec, noise=None,
                n_realizations=1, mode="complex_baseband",
            )
            n_draw = len(result_det.v_nl_qubit)
            freqs_phi = np.fft.rfftfreq(n_draw, d=1.0 / result_det.fs)
            psd_arr = phase_noise_psd_from_spec(
                {"single_sided_psd_rad2_per_hz": _PHASE_NOISE_PSD, "bandwidth_hz": _PHASE_NOISE_BW}, freqs_phi,
            )
            rng = np.random.default_rng(_SEED)
            posthoc_ensemble = [
                result_det.v_nl_qubit * np.exp(1j * generate_phase_noise(n_draw, result_det.fs, psd_arr, rng=rng))
                for _ in range(300)
            ]
            var_post = np.var([v - result_det.v_nl_qubit for v in posthoc_ensemble])

        rel_diff = abs(var_pre - var_post) / var_post
        assert rel_diff > 0.05, (
            f"Expected pre-NL and post-hoc phase noise to give measurably different "
            f"ensemble variance under a compressing nonlinearity: pre={var_pre:.4e}, "
            f"post={var_post:.4e} (rel diff {rel_diff:.1%}) -- if this ever converges to "
            f"~0%, something has silently made phase noise commute with the nonlinearity, "
            f"which would mean this feature's whole reason for re-running the NL pass per "
            f"realization has stopped mattering for this test case."
        )


if __name__ == "__main__":
    pytest.main([__file__])
