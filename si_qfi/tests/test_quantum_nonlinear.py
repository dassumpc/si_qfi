"""
tests/test_quantum_nonlinear.py
================================
Gate-fidelity tests with a REAL nonlinearity active in the drive chain --
companion to tests/test_quantum.py (which is deliberately linear/
"no impairment"). Covers all four NL-model x mode combinations this
codebase actually supports: Saleh AM-AM-only and Saleh AM-AM+AM-PM in
complex_baseband mode, and SalehRealAxisModel / VolterraModel (AM-AM only
-- see below) in real_axis mode.

THE HEADLINE PHYSICS RESULT THESE TESTS LOCK IN (investigated via
examples/nonlinearity_fidelity_demo.py, verified independently by hand
against a direct nonlinear.saleh.SalehModel call with no schematic/engine/
QuTiP involved at all before being written into this file):

  Pure AM-AM (real gain compression, no phase distortion) applied to a
  SINGLE-AXIS (I-only) drive on a resonant 2-level qubit does NOT limit
  achievable gate fidelity AT ALL, provided the pulse amplitude is
  recalibrated through the compression (see quantum.tuneup_amplitude()).
  This is because a memoryless real-gain nonlinearity multiplies I(t) by a
  real, non-negative scalar at every instant -- it never rotates energy
  into the Q axis, so it can only ever reshape/rescale the TOTAL
  integrated rotation angle (theta = eta*integral(I(t))dt), never distort
  the axis itself. For a 2-level qubit with no other timescale (no
  detuning, no leakage), only theta matters -- not the instantaneous pulse
  shape -- so recalibrating amplitude to hit the exact target theta
  recovers F~1 to full numerical precision, EVEN WELL INTO COMPRESSION.

  This has a real, hard limit though: because a Saleh/Volterra AM-AM curve
  eventually turns over (raw output DECREASES for large enough input --
  see max_monotonic_amplitude in nonlinear/saleh.py and
  nonlinear/volterra.py), a given (op1db, pulse shape/duration) combination
  has a MAXIMUM ACHIEVABLE total rotation angle. If that maximum is below
  the target (e.g. pi for an X gate), NO amplitude -- however large --
  reaches it: recalibration doesn't gracefully degrade, it becomes flatly
  IMPOSSIBLE. This is a genuinely different failure mode from "some
  residual infidelity" -- see test_*_amam_only_severe_compression_not_achievable
  below.

  AM-PM (phase-vs-amplitude distortion), in contrast, genuinely DOES limit
  fidelity, and cannot be undone by recalibrating the I-axis amplitude:
  Phi[A] rotates energy into the Q axis as a function of instantaneous
  amplitude, which distorts the EFFECTIVE rotation axis over the course of
  the pulse, not just its magnitude -- recalibrating total I-axis area
  doesn't fix an axis that's pointing the wrong way at each instant.
  Infidelity grows smoothly and monotonically with AM-PM depth (unlike
  AM-AM's all-or-nothing cliff), and is worse the deeper the pulse drives
  into compression (since Phi[A] itself grows with amplitude).

  AM-PM is ONLY implemented for SalehModel in complex_baseband mode in
  this codebase -- SalehRealAxisModel and VolterraModel are both AM-AM
  only (see their module docstrings: a real, memoryless, instantaneous
  nonlinearity has no separate "envelope phase" to modulate). So in
  real_axis mode, nonlinearity in this codebase can ONLY ever show the
  "recalibrate and it's fine, or it's a hard wall" AM-AM behavior above --
  it cannot currently reproduce a real amplifier's real-axis phase
  distortion (if the amplifier being modeled has one, complex_baseband
  mode with SalehModel's AM-PM is the only way to capture its effect here).

Calibration (2026-07-13): both this file's single- and two-stage
calibration now go through quantum.tuneup_amplitude(), which internally
does the same coarse-geomspace-scan + bisection-on-the-rising-branch this
file used to hand-roll (Saleh/Volterra AM-AM curves are not monotonic in
scale once driven far enough), generalized to accept the nonlinear= dict
directly so it handles 0/1/2 NL nodes without separate single-/two-stage
code paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")
qutip = pytest.importorskip("qutip")

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope
from si_qfi.nonlinear.saleh import SalehModel, SalehRealAxisModel
from si_qfi.nonlinear.volterra import VolterraModel
from si_qfi import quantum

BASIC_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_basic.si").resolve()

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6
_DURATION_S = 100e-9
_SIGMA_S = _DURATION_S / 6
_FS_ENVELOPE = 2e9
_LPF_CUTOFF_HZ = 500e6
_NL_LABEL = "DriverOutput"


@pytest.fixture
def basic_schematic():
    return si_loader.load_schematic(BASIC_SCHEMATIC_PATH)


def _qubit_model_2lvl():
    return quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)


def _tune(schematic, mode, nl_model_fn, qmodel, lpf_cutoff_hz=None):
    """quantum.tuneup_amplitude() with this file's single-NL-node spec."""
    ref_shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    nl_spec = {_NL_LABEL: nl_model_fn.spec()}
    return quantum.tuneup_amplitude(
        schematic, ref_shape, _FS_ENVELOPE, _CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X",
        nonlinear=nl_spec, mode=mode, lpf_cutoff_hz=lpf_cutoff_hz,
    )


class _SalehSpec:
    """nl_model_fn for baseband SalehModel, with or without AM-PM."""
    def __init__(self, op1db, enable_am_pm=False, am_pm_peak_deg=0.0):
        self.op1db = op1db
        self.enable_am_pm = enable_am_pm
        self.am_pm_peak_deg = am_pm_peak_deg

    def spec(self):
        d = {"model": "saleh", "op1db_amplitude": self.op1db}
        if self.enable_am_pm:
            d["enable_am_pm"] = True
            d["am_pm_peak_deg"] = self.am_pm_peak_deg
        return d


class _SalehRealAxisSpec:
    """nl_model_fn for SalehRealAxisModel -- AM-AM only, no enable_am_pm option."""
    def __init__(self, op1db):
        self.op1db = op1db

    def spec(self):
        return {"model": "saleh", "op1db_amplitude": self.op1db}


class _VolterraSpec:
    """nl_model_fn for VolterraModel (cubic describing function) -- AM-AM only."""
    def __init__(self, op1db):
        self.op1db = op1db

    def spec(self):
        return {"model": "volterra", "option": "describing",
                "op1db_amplitude": self.op1db, "memory_depth": 0}


# ---------------------------------------------------------------------------
# Baseband, Saleh, AM-AM only -- recalibration should recover F~1 exactly,
# for op1db choices spanning "barely into compression" to "quite deep",
# confirming this holds broadly, not just at one lucky point.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op1db", [10.0, 1.5, 1.0])
def test_baseband_saleh_amam_only_recalibrated_unity(basic_schematic, op1db):
    fn = _SalehSpec(op1db, enable_am_pm=False)
    qmodel = _qubit_model_2lvl()
    tuned = _tune(basic_schematic, "complex_baseband", fn, qmodel)
    assert tuned.achieved, f"op1db={op1db} should be comfortably achievable for this pulse"
    assert tuned.fidelity.noise_free.F_avg == pytest.approx(1.0, abs=1e-4)


def test_baseband_saleh_amam_only_severe_compression_not_achievable(basic_schematic):
    """
    Below some critical op1db (for THIS pulse shape/duration/eta), no
    amplitude reaches a full pi rotation -- the AM-AM curve's raw output
    turns over (see nonlinear/saleh.py's max_monotonic_amplitude) before
    accumulating enough area. This is the "hard wall" failure mode, a
    fundamentally different thing from a gracefully-degrading fidelity --
    confirmed here as a real (not a search-artifact) property of the model.
    """
    fn = _SalehSpec(op1db=0.3, enable_am_pm=False)
    qmodel = _qubit_model_2lvl()
    tuned = _tune(basic_schematic, "complex_baseband", fn, qmodel, )
    assert not tuned.achieved
    assert tuned.fidelity.noise_free.F_avg < 0.99


# ---------------------------------------------------------------------------
# Baseband, Saleh, AM-AM + AM-PM -- recalibrating the I-axis area alone
# CANNOT undo AM-PM: infidelity should be measurably worse than the
# AM-AM-only case at the same op1db, and should grow with am_pm_peak_deg.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("am_pm_peak_deg", [5.0, 15.0])
def test_baseband_saleh_ampm_reduces_fidelity_vs_amam_only(basic_schematic, am_pm_peak_deg):
    op1db = 1.5
    qmodel = _qubit_model_2lvl()
    fn_amam = _SalehSpec(op1db, enable_am_pm=False)
    fn_ampm = _SalehSpec(op1db, enable_am_pm=True, am_pm_peak_deg=am_pm_peak_deg)

    tuned_amam = _tune(basic_schematic, "complex_baseband", fn_amam, qmodel)
    tuned_ampm = _tune(basic_schematic, "complex_baseband", fn_ampm, qmodel)
    assert tuned_amam.achieved and tuned_ampm.achieved

    fid_amam = tuned_amam.fidelity.noise_free.F_avg
    fid_ampm = tuned_ampm.fidelity.noise_free.F_avg

    assert fid_amam == pytest.approx(1.0, abs=1e-4)   # AM-AM-only baseline: still ~1
    assert fid_ampm < fid_amam - 1e-4                  # AM-PM measurably worse
    assert 0.0 <= fid_ampm <= 1.0


def test_baseband_saleh_ampm_infidelity_grows_with_peak_deg(basic_schematic):
    """Infidelity should increase monotonically (not just be 'present') as
    am_pm_peak_deg increases -- confirms it's a real, graded effect, not a
    one-off threshold."""
    op1db = 1.5
    qmodel = _qubit_model_2lvl()
    infidelities = []
    for peak_deg in [2.0, 10.0, 20.0]:
        fn = _SalehSpec(op1db, enable_am_pm=True, am_pm_peak_deg=peak_deg)
        tuned = _tune(basic_schematic, "complex_baseband", fn, qmodel)
        assert tuned.achieved
        infidelities.append(1.0 - tuned.fidelity.noise_free.F_avg)
    assert infidelities[0] < infidelities[1] < infidelities[2]


# ---------------------------------------------------------------------------
# Real-axis, SalehRealAxisModel and VolterraModel -- both AM-AM only (no
# AM-PM mechanism exists for either in real_axis mode, see module
# docstring), so both should show the SAME "recalibrate -> F~1, or hard
# wall" pattern as the baseband AM-AM-only case above.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op1db", [10.0, 1.5, 1.0])
def test_realaxis_saleh_amam_only_recalibrated_unity(basic_schematic, op1db):
    fn = _SalehRealAxisSpec(op1db)
    qmodel = _qubit_model_2lvl()
    tuned = _tune(basic_schematic, "real_axis", fn, qmodel, lpf_cutoff_hz=_LPF_CUTOFF_HZ)
    assert tuned.achieved
    assert tuned.fidelity.noise_free.F_avg == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("op1db", [10.0, 1.5, 1.0])
def test_realaxis_volterra_amam_only_recalibrated_unity(basic_schematic, op1db):
    """
    Same claim, VolterraModel's cubic describing function instead of
    Saleh's bounded rational curve -- a DIFFERENT model shape, same
    physics conclusion, confirming the "AM-AM alone doesn't limit fidelity
    once recalibrated" result isn't an artifact of one particular curve
    shape. op1db=1.0 is deliberately kept away from
    max_monotonic_amplitude here (contrast with the cascaded-overdrive
    breakdown in tests/test_engine.py, where Volterra's cubic is pushed
    PAST that point and genuinely breaks down -- this test is specifically
    about the well-behaved regime).
    """
    fn = _VolterraSpec(op1db)
    qmodel = _qubit_model_2lvl()
    tuned = _tune(basic_schematic, "real_axis", fn, qmodel, lpf_cutoff_hz=_LPF_CUTOFF_HZ)
    assert tuned.achieved
    assert tuned.fidelity.noise_free.F_avg == pytest.approx(1.0, abs=1e-4)


def test_realaxis_saleh_amam_only_severe_compression_not_achievable(basic_schematic):
    """Real-axis companion to the baseband "hard wall" test above -- same
    conclusion, different model/mode."""
    fn = _SalehRealAxisSpec(op1db=0.3)
    qmodel = _qubit_model_2lvl()
    tuned = _tune(basic_schematic, "real_axis", fn, qmodel, lpf_cutoff_hz=_LPF_CUTOFF_HZ)
    assert not tuned.achieved


# ---------------------------------------------------------------------------
# Two-amplifier cascade (examples/two_amp_harmonic_remixing_demo.py):
# does AM-AM alone limit fidelity in real_axis mode once there are TWO
# cascaded stages, via 3rd-harmonic energy from stage 1 remixing back
# in-band at stage 2? Verified: yes, but specifically as an ACHIEVABILITY-
# RANGE EXTENSION with a genuine partial-infidelity "gray zone" attached to
# it, not a simple "always somewhat worse". See the demo script's module
# docstring for the full physical argument (the 3*x1^2*x3 cubic cross-term)
# and examples/two_amp_harmonic_remixing_demo.png for the swept curves.
# ---------------------------------------------------------------------------

TWO_AMP_SCHEMATIC_PATH = (
    Path(__file__).parent / "test_schematic_lossy_T_line_2_amplifier.si"
).resolve()
_NODE_1, _NODE_2 = "DriverOutput", "DriverOutput2"

# Longer pulse than the single-amp tests' shared _DURATION_S (100ns) --
# this schematic's transmission lines are LOSSY/dispersive (unlike
# test_schematic_basic.si's lossless, flat lines), so even with NO
# nonlinearity at all, a 100ns (wider-bandwidth) pulse picks up a genuine
# ~1e-5 infidelity from ordinary linear dispersion -- confirmed directly
# via engine.run(nonlinear=None), and confirmed to scale with pulse
# bandwidth as expected (roughly infidelity ~ bandwidth^2). That floor
# sat uncomfortably close to the smallest genuine two-stage-only "gray
# zone" values, muddying the comparison (caught when the user pushed back
# on why the demo's floor looked so much higher than the single-amp
# demo's ~1e-11). 400ns (1/4 the bandwidth) pushes the baseline down to
# ~6e-7, well clear of the effect under test -- see
# examples/two_amp_harmonic_remixing_demo.py's module docstring for the
# full investigation.
_TWO_AMP_DURATION_S = 400e-9
_TWO_AMP_SIGMA_S = _TWO_AMP_DURATION_S / 6


@pytest.fixture
def two_amp_schematic():
    return si_loader.load_schematic(TWO_AMP_SCHEMATIC_PATH)


def _tune_two_stage(schematic, mode, nl1_fn, nl2_fn, qmodel, lpf_cutoff_hz=None):
    """quantum.tuneup_amplitude() with up to two NL nodes (either may be
    None for a single-stage comparison through this same schematic)."""
    ref_shape = build_gaussian_envelope(_TWO_AMP_DURATION_S, _TWO_AMP_SIGMA_S, _FS_ENVELOPE, amp=1.0)
    nl_spec = {}
    if nl1_fn:
        nl_spec[_NODE_1] = nl1_fn.spec()
    if nl2_fn:
        nl_spec[_NODE_2] = nl2_fn.spec()
    return quantum.tuneup_amplitude(
        schematic, ref_shape, _FS_ENVELOPE, _CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X",
        nonlinear=nl_spec if nl_spec else None, mode=mode, lpf_cutoff_hz=lpf_cutoff_hz,
    )


def test_realaxis_two_stage_extends_achievable_range_beyond_single_stage(two_amp_schematic):
    """
    The headline finding: at op1db=0.17 (real_axis, Saleh real-axis AM-AM
    only, 400ns pulse -- see _TWO_AMP_DURATION_S), a SINGLE nonlinear
    stage (only NODE_2, closest to the qubit) cannot complete a full pi
    pulse at all -- but adding a SECOND nonlinear stage (NODE_1,
    identically specified) upstream makes it achievable, with real but
    bounded infidelity, clearly above the no-NL baseline floor for this
    schematic (~5e-7 at this pulse duration -- see
    _TWO_AMP_DURATION_S's docstring for why 400ns, not 100ns). This is the
    harmonic-remixing "gray zone": stage 1's 3rd-harmonic content,
    generated by driving it hard, survives the dispersive inter-stage
    line and mixes back in-band at stage 2 (see module docstring of
    examples/two_amp_harmonic_remixing_demo.py for the cubic cross-term
    argument) -- something neither mode's single-stage math nor
    complex_baseband's harmonic-blind two-stage math can reproduce.
    """
    op1db = 0.17
    qmodel = _qubit_model_2lvl()

    tuned_single = _tune_two_stage(
        two_amp_schematic, "real_axis", None, _SalehRealAxisSpec(op1db), qmodel,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert not tuned_single.achieved, "single-stage at op1db=0.17 should be unreachable"

    tuned_two = _tune_two_stage(
        two_amp_schematic, "real_axis", _SalehRealAxisSpec(op1db), _SalehRealAxisSpec(op1db), qmodel,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert tuned_two.achieved, "two-stage at the SAME op1db should be reachable"

    infid_two = 1.0 - tuned_two.fidelity.noise_free.F_avg
    # Real, bounded, non-floor infidelity -- not F~1 (that would mean the
    # gray zone isn't real) and not down at the ~5e-7 no-NL baseline floor
    # either (that would mean this point isn't actually testing the
    # nonlinear effect), and not catastrophic.
    assert 1e-6 < infid_two < 1e-2


def test_realaxis_two_stage_matches_single_stage_at_mild_compression(two_amp_schematic):
    """Sanity check on the flip side: at MILD compression (op1db=1.0, well
    away from either cliff), single- and two-stage give essentially the
    same (near-floor) infidelity -- the gray zone is specifically a
    deep-compression phenomenon, not a blanket "two stages always worse"."""
    op1db = 1.0
    qmodel = _qubit_model_2lvl()

    tuned_single = _tune_two_stage(
        two_amp_schematic, "real_axis", None, _SalehRealAxisSpec(op1db), qmodel,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    tuned_two = _tune_two_stage(
        two_amp_schematic, "real_axis", _SalehRealAxisSpec(op1db), _SalehRealAxisSpec(op1db), qmodel,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert tuned_single.achieved and tuned_two.achieved
    assert tuned_single.fidelity.noise_free.F_avg == pytest.approx(
        tuned_two.fidelity.noise_free.F_avg, abs=1e-4,
    )


def test_realaxis_two_stage_no_nl_baseline_is_small(two_amp_schematic):
    """
    This schematic's own lossy, dispersive transmission lines distort the
    pulse a little even with NO nonlinearity active at all -- confirmed
    directly (caught when a first version of this investigation's demo
    used a 100ns pulse and got a ~1e-5 "floor" that turned out to be this
    linear effect, not anything to do with AM-AM). At the 400ns pulse
    used throughout this section, that baseline should be small enough
    (< 1e-6) not to be mistaken for the nonlinear gray-zone effect above.
    """
    qmodel = _qubit_model_2lvl()
    tuned = _tune_two_stage(two_amp_schematic, "real_axis", None, None, qmodel, lpf_cutoff_hz=_LPF_CUTOFF_HZ)
    assert tuned.achieved
    assert (1.0 - tuned.fidelity.noise_free.F_avg) < 1e-6


def test_realaxis_two_stage_reaches_lower_op1db_than_baseband_two_stage(two_amp_schematic):
    """
    complex_baseband is structurally blind to harmonic content (it only
    ever tracks the single fc-centered envelope band), so its two-stage
    achievability cliff should sit at a HIGHER (milder) op1db than
    real_axis's -- confirming the extra range real_axis reaches is
    specifically attributable to harmonic-domain physics baseband cannot
    represent, not just "two-stage is generally more forgiving" (which
    both modes show to /some/ degree, but real_axis shows more of it).
    """
    op1db = 0.17   # real_axis two-stage: reachable (see test above); baseband: not
    qmodel = _qubit_model_2lvl()
    fn1_bb, fn2_bb = _SalehSpec(op1db, enable_am_pm=False), _SalehSpec(op1db, enable_am_pm=False)
    tuned_bb = _tune_two_stage(two_amp_schematic, "complex_baseband", fn1_bb, fn2_bb, qmodel)
    assert not tuned_bb.achieved, "baseband's two-stage cliff should be above op1db=0.17"

    tuned_ra = _tune_two_stage(
        two_amp_schematic, "real_axis", _SalehRealAxisSpec(op1db), _SalehRealAxisSpec(op1db), qmodel,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert tuned_ra.achieved


if __name__ == "__main__":
    pytest.main([__file__])
