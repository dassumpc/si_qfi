"""
tests/test_quantum_dispersion.py
=================================
Regression tests for examples/bandwidth_dispersion_fidelity_demo.py's
finding: pure LINEAR channel dispersion (no nonlinearity anywhere) costs
real, measurable gate fidelity, scaling with pulse bandwidth roughly as
bandwidth^2, and accumulating through additional lossy line segments.

This was discovered as a byproduct of the nonlinearity investigation in
tests/test_quantum_nonlinear.py's two-amp section -- a first version of
that demo used a 100ns pulse and reported a ~1e-5 "floor" that turned out
to be this effect, not anything nonlinear. See that file's module
docstring and examples/two_amp_harmonic_remixing_demo.py's "IMPORTANT
CORRECTION" note for the full story. This file exists so the effect that
caused that confusion is itself locked in and understood on its own
terms, not just avoided by using a longer pulse elsewhere.

nonlinear=None throughout -- every infidelity here is attributable
entirely to the SI schematic's own linear (but frequency-dependent)
transfer function.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")
qutip = pytest.importorskip("qutip")

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope
from si_qfi import quantum

# Resolved before any schematic is opened -- see
# examples/bandwidth_dispersion_fidelity_demo.py's module docstring for
# why (OpenProjectFile() changes the process's cwd as a side effect).
_TESTS_DIR = Path(__file__).parent.resolve()
LOSSLESS_PATH = _TESTS_DIR / "test_schematic_basic.si"
LOSSY_1_PATH = _TESTS_DIR / "test_schematic_lossy_T_line.si"
LOSSY_2_PATH = _TESTS_DIR / "test_schematic_lossy_T_line_2_amplifier.si"

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6
_FS_ENVELOPE = 8e9
_LPF_CUTOFF_HZ = 500e6


@pytest.fixture
def lossless_schematic():
    return si_loader.load_schematic(LOSSLESS_PATH)


@pytest.fixture
def lossy_1_schematic():
    return si_loader.load_schematic(LOSSY_1_PATH)


@pytest.fixture
def lossy_2_schematic():
    return si_loader.load_schematic(LOSSY_2_PATH)


def _qubit_model_2lvl():
    return quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)


def _infidelity_no_nl(schematic, duration_s, qmodel, mode="complex_baseband", lpf_cutoff_hz=None):
    """tuneup_amplitude()-calibrated infidelity, NO nonlinearity -- the
    whole chain is linear, so the analytic-guess fast path handles this in
    2 engine.run() calls, same cost as the hand-rolled version this
    replaces."""
    sigma_s = duration_s / 6
    ref_shape = build_gaussian_envelope(duration_s, sigma_s, _FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, _FS_ENVELOPE, _CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X",
        mode=mode, lpf_cutoff_hz=lpf_cutoff_hz,
    )
    return 1.0 - tuned.fidelity.noise_free.F_avg


def test_lossless_schematic_stays_at_floor_regardless_of_bandwidth(lossless_schematic):
    """Control: a matched, lossless schematic has zero dispersion by
    construction, so infidelity should stay at the float64 noise floor
    (no trend) whether the pulse is narrow or wide -- confirms the effect
    tested below is genuinely about dispersion, not a generic artifact of
    self-calibrating a pulse through any schematic."""
    qmodel = _qubit_model_2lvl()
    infid_narrow = _infidelity_no_nl(lossless_schematic, 400e-9, qmodel)
    infid_wide = _infidelity_no_nl(lossless_schematic, 50e-9, qmodel)
    assert infid_narrow < 1e-9
    assert infid_wide < 1e-9


def test_lossy_schematic_infidelity_decreases_with_narrower_bandwidth(lossy_1_schematic):
    """Halving the pulse bandwidth (doubling duration) should measurably
    reduce dispersion-driven infidelity -- monotonic trend across a few
    points, not just two endpoints."""
    qmodel = _qubit_model_2lvl()
    durations = [100e-9, 200e-9, 400e-9, 800e-9]
    infidelities = [_infidelity_no_nl(lossy_1_schematic, d, qmodel) for d in durations]
    assert infidelities == sorted(infidelities, reverse=True)
    assert all(i > 0 for i in infidelities)


def test_lossy_infidelity_scales_roughly_quadratically_with_bandwidth(lossy_1_schematic):
    """Fit infidelity ~ bandwidth^p on a log-log basis -- should land close
    to p=2 (textbook scaling for a channel that isn't perfectly flat
    across the signal's own bandwidth), not e.g. p~1 (linear) or p~0 (no
    dependence, which would mean this isn't really a bandwidth effect)."""
    qmodel = _qubit_model_2lvl()
    durations = np.array([100e-9, 200e-9, 400e-9, 800e-9, 1600e-9])
    bandwidths = 1.0 / (2 * np.pi * (durations / 6))
    infidelities = np.array([_infidelity_no_nl(lossy_1_schematic, d, qmodel) for d in durations])
    slope, _ = np.polyfit(np.log(bandwidths), np.log(infidelities), 1)
    assert 1.5 < slope < 2.5


def test_two_lossy_segments_worse_than_one_at_same_bandwidth(lossy_1_schematic, lossy_2_schematic):
    """Dispersion should accumulate through additional lossy line
    segments -- the two-amplifier schematic (two lossy segments) should
    show measurably WORSE infidelity than the one-amplifier schematic
    (one lossy segment) at the same pulse duration, with no nonlinearity
    involved at all."""
    qmodel = _qubit_model_2lvl()
    duration = 200e-9
    infid_1 = _infidelity_no_nl(lossy_1_schematic, duration, qmodel)
    infid_2 = _infidelity_no_nl(lossy_2_schematic, duration, qmodel)
    assert infid_2 > 1.5 * infid_1


def test_dispersion_infidelity_agrees_between_modes(lossy_2_schematic):
    """Unlike the nonlinear harmonic-remixing effect (real_axis-only, see
    tests/test_quantum_nonlinear.py), ordinary linear dispersion should be
    captured similarly by both modes -- complex_baseband's own H(f)
    representation already carries in-band linear channel effects
    exactly; real_axis should agree to within an order of magnitude, not
    diverge structurally the way the nonlinear effect does."""
    qmodel = _qubit_model_2lvl()
    duration = 400e-9
    infid_bb = _infidelity_no_nl(lossy_2_schematic, duration, qmodel, "complex_baseband")
    infid_ra = _infidelity_no_nl(lossy_2_schematic, duration, qmodel, "real_axis", _LPF_CUTOFF_HZ)
    ratio = infid_ra / infid_bb
    assert 0.1 < ratio < 10.0


if __name__ == "__main__":
    pytest.main([__file__])
