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
  recalibrated through the compression (see _calibrate_and_run() below).
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
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

pytest.importorskip("SignalIntegrity")
qutip = pytest.importorskip("qutip")

from si_qfi.schematic import loader as si_loader
from si_qfi.schematic import transfer_function as si_tf
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform, build_gaussian_envelope
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


def _source_from_shape(shape: np.ndarray, fs: float, carrier_ghz: float) -> SourceWaveform:
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def _pre_nl_waveform(schematic, source, mode, mid_label=_NL_LABEL):
    """Independently compute the waveform arriving at mid_label BEFORE any
    nonlinearity -- same pattern as tests/test_engine.py's helper of the
    same name."""
    raw = si_tf._extract_single_tf(schematic.si_app, schematic.source_label, mid_label, schematic.source_label)
    h = si_tf.compute_impulse_response(raw, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz).h
    if mode == "real_axis":
        fs_native = si_tf.native_sample_rate(raw)
        _, v_initial = source.rf_waveform_at(fs_native)
        fs_out = fs_native
    else:
        v_initial = source.envelope_complex
        fs_out = source.fs
    return np.convolve(v_initial, h, mode="full"), fs_out


def _segment_h(schematic, source, mode, label_in, label_out):
    """Same pattern as tests/test_engine.py's helper of the same name."""
    raw = si_tf._extract_single_tf(schematic.si_app, label_in, label_out, schematic.source_label)
    return si_tf.compute_impulse_response(raw, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz).h


def _calibrate_and_run(
    schematic, mode, nl_model_fn, target_theta=np.pi,
    lpf_cutoff_hz=None, scale_lo=1e-4, scale_hi=200.0, n_scan=60, n_bisect=30,
):
    """
    Self-calibrate a Gaussian I-only pulse's amplitude to hit exactly
    target_theta radians of rotation THROUGH a real nonlinearity, then run
    the full engine.run() pipeline once at that amplitude.

    Unlike tests/test_quantum.py's linear-system 2-shot calibration (valid
    only because that file has no NL node -- a single reference run +
    exact rescale works for a LINEAR chain), a real nonlinearity means
    input_scale -> realized_theta is not simply proportional, and -- per
    this file's module docstring -- is not even guaranteed MONOTONIC
    (Saleh/Volterra AM-AM curves turn over at large amplitude). So this
    does a coarse log-spaced scan first to bracket the target on its
    RISING branch (bisecting blindly over a non-monotonic range converges
    to the wrong point -- confirmed the hard way while building this),
    then bisects within that bracket.

    All of this search runs on pure numpy (the pre-NL waveform and the
    post-NL-node-to-qubit segment are both computed ONCE via SI, since the
    schematic itself is linear -- only nl_model_fn()'s output is
    re-evaluated per trial amplitude) -- no repeated SI/engine.run() calls
    during the search itself, only a single final one once calibrated.

    nl_model_fn : a zero-arg callable returning a fresh NonlinearNode
        instance (e.g. lambda: SalehModel.from_op1db_oip3(...)), plus a
        .spec() method returning the equivalent nonlinear= spec dict for
        engine.run() -- see the NLSpec helper classes below.

    Returns (result, achieved, theta_hit_or_max_achievable):
        achieved=True: result is a real SimulationResult, calibrated to
            within numerical bisection precision of target_theta.
        achieved=False: target_theta is not reachable by ANY amplitude for
            this nonlinearity/pulse shape -- result is None, and the third
            return value is the MAXIMUM achievable theta found during the
            scan (always < target_theta in this case).
    """
    ref_shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    v_pre, fs_pre = _pre_nl_waveform(schematic, source_ref, mode)
    h_post = _segment_h(schematic, source_ref, mode, _NL_LABEL, schematic.qubit_probe_label)

    def theta_for_scale(scale):
        nl_model = nl_model_fn()
        if mode == "complex_baseband":
            v_nl = nl_model.apply_baseband(v_pre * scale)
        else:
            v_nl = nl_model.apply_real_axis(v_pre * scale)
        v_post = fftconvolve(v_nl, h_post, mode="full")
        t = np.arange(len(v_post)) / fs_pre
        if mode == "complex_baseband":
            env_i = np.real(v_post)
        else:
            env_i, _ = quantum.demodulate(v_post, t, _CARRIER_GHZ * 1e9, lpf_cutoff_hz)
        return float(_ETA * np.trapz(env_i, t))

    # The coarse scan deliberately sweeps well past each model's own
    # max_monotonic_amplitude (needed to find the RISING branch and to
    # detect the "not achievable" case at all) -- that's expected, not a
    # bug, so silence the resulting overdrive warnings here rather than
    # let them spam every test's output (same pattern as
    # tests/test_nonlinear.py's _find_compression_point()).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scales = np.geomspace(scale_lo, scale_hi, n_scan)
        thetas = np.array([theta_for_scale(s) for s in scales])
    achievable_max = float(np.max(thetas))
    idx = int(np.argmax(thetas >= target_theta))
    if thetas[idx] < target_theta:
        return None, False, achievable_max

    lo, hi = scales[max(idx - 1, 0)], scales[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            th = theta_for_scale(mid)
            if th < target_theta:
                lo = mid
            else:
                hi = mid
    scale = 0.5 * (lo + hi)

    cal_shape = ref_shape * scale
    source_cal = _source_from_shape(cal_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    nl_spec = {_NL_LABEL: nl_model_fn.spec()}
    result = engine.run(
        schematic, source_cal, nonlinear=nl_spec, noise=None, n_realizations=1, mode=mode,
    )
    return result, True, theta_for_scale(scale)


class _SalehSpec:
    """nl_model_fn for baseband SalehModel, with or without AM-PM."""
    def __init__(self, op1db, enable_am_pm=False, am_pm_peak_deg=0.0):
        self.op1db = op1db
        self.enable_am_pm = enable_am_pm
        self.am_pm_peak_deg = am_pm_peak_deg

    def __call__(self):
        return SalehModel.from_op1db_oip3(
            op1db_amplitude=self.op1db,
            enable_am_pm=self.enable_am_pm, am_pm_peak_deg=self.am_pm_peak_deg,
        )

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

    def __call__(self):
        return SalehRealAxisModel.from_op1db_oip3(op1db_amplitude=self.op1db)

    def spec(self):
        return {"model": "saleh", "op1db_amplitude": self.op1db}


class _VolterraSpec:
    """nl_model_fn for VolterraModel (cubic describing function) -- AM-AM only."""
    def __init__(self, op1db):
        self.op1db = op1db

    def __call__(self):
        return VolterraModel(option="describing", op1db_amplitude=self.op1db, memory_depth=0)

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
    result, achieved, theta_hit = _calibrate_and_run(basic_schematic, "complex_baseband", fn)
    assert achieved, f"op1db={op1db} should be comfortably achievable for this pulse"
    assert theta_hit == pytest.approx(np.pi, abs=1e-3)

    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X")
    assert fid.F_avg == pytest.approx(1.0, abs=1e-4)


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
    result, achieved, theta_max = _calibrate_and_run(basic_schematic, "complex_baseband", fn)
    assert not achieved
    assert result is None
    assert theta_max < np.pi


# ---------------------------------------------------------------------------
# Baseband, Saleh, AM-AM + AM-PM -- recalibrating the I-axis area alone
# CANNOT undo AM-PM: infidelity should be measurably worse than the
# AM-AM-only case at the same op1db, and should grow with am_pm_peak_deg.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("am_pm_peak_deg", [5.0, 15.0])
def test_baseband_saleh_ampm_reduces_fidelity_vs_amam_only(basic_schematic, am_pm_peak_deg):
    op1db = 1.5
    fn_amam = _SalehSpec(op1db, enable_am_pm=False)
    fn_ampm = _SalehSpec(op1db, enable_am_pm=True, am_pm_peak_deg=am_pm_peak_deg)

    result_amam, ok1, _ = _calibrate_and_run(basic_schematic, "complex_baseband", fn_amam)
    result_ampm, ok2, _ = _calibrate_and_run(basic_schematic, "complex_baseband", fn_ampm)
    assert ok1 and ok2

    qmodel = _qubit_model_2lvl()
    fid_amam = quantum.gate_fidelity(result_amam, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X")
    fid_ampm = quantum.gate_fidelity(result_ampm, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X")

    assert fid_amam.F_avg == pytest.approx(1.0, abs=1e-4)   # AM-AM-only baseline: still ~1
    assert fid_ampm.F_avg < fid_amam.F_avg - 1e-4            # AM-PM measurably worse
    assert 0.0 <= fid_ampm.F_avg <= 1.0


def test_baseband_saleh_ampm_infidelity_grows_with_peak_deg(basic_schematic):
    """Infidelity should increase monotonically (not just be 'present') as
    am_pm_peak_deg increases -- confirms it's a real, graded effect, not a
    one-off threshold."""
    op1db = 1.5
    qmodel = _qubit_model_2lvl()
    infidelities = []
    for peak_deg in [2.0, 10.0, 20.0]:
        fn = _SalehSpec(op1db, enable_am_pm=True, am_pm_peak_deg=peak_deg)
        result, ok, _ = _calibrate_and_run(basic_schematic, "complex_baseband", fn)
        assert ok
        fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X")
        infidelities.append(1.0 - fid.F_avg)
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
    result, achieved, theta_hit = _calibrate_and_run(
        basic_schematic, "real_axis", fn, lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert achieved
    assert theta_hit == pytest.approx(np.pi, abs=1e-3)

    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X",
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert fid.F_avg == pytest.approx(1.0, abs=1e-4)


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
    result, achieved, theta_hit = _calibrate_and_run(
        basic_schematic, "real_axis", fn, lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert achieved
    assert theta_hit == pytest.approx(np.pi, abs=1e-3)

    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X",
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert fid.F_avg == pytest.approx(1.0, abs=1e-4)


def test_realaxis_saleh_amam_only_severe_compression_not_achievable(basic_schematic):
    """Real-axis companion to the baseband "hard wall" test above -- same
    conclusion, different model/mode."""
    fn = _SalehRealAxisSpec(op1db=0.3)
    result, achieved, theta_max = _calibrate_and_run(
        basic_schematic, "real_axis", fn, lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert not achieved
    assert theta_max < np.pi


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


def _calibrate_two_stage(
    schematic, mode, nl1_fn, nl2_fn, target_theta=np.pi,
    lpf_cutoff_hz=None, scale_lo=1e-4, scale_hi=2000.0, n_scan=80, n_bisect=35,
):
    """
    Two-node generalization of _calibrate_and_run() above -- nl1_fn/nl2_fn
    (either may be None) apply at _NODE_1/_NODE_2 respectively, letting a
    "single-stage" run (one of them None) be compared against a
    "two-stage" run through the EXACT SAME schematic/channel.
    """
    ref_shape = build_gaussian_envelope(_TWO_AMP_DURATION_S, _TWO_AMP_SIGMA_S, _FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    v_pre1, fs_pre = _pre_nl_waveform(schematic, source_ref, mode, _NODE_1)
    h_mid = _segment_h(schematic, source_ref, mode, _NODE_1, _NODE_2)
    h_post = _segment_h(schematic, source_ref, mode, _NODE_2, schematic.qubit_probe_label)

    def theta_for_scale(scale):
        nl1 = nl1_fn() if nl1_fn else None
        nl2 = nl2_fn() if nl2_fn else None
        if mode == "complex_baseband":
            v_nl1 = nl1.apply_baseband(v_pre1 * scale) if nl1 else v_pre1 * scale
            v_pre2 = fftconvolve(v_nl1, h_mid, mode="full")
            v_nl2 = nl2.apply_baseband(v_pre2) if nl2 else v_pre2
        else:
            v_nl1 = nl1.apply_real_axis(v_pre1 * scale) if nl1 else v_pre1 * scale
            v_pre2 = fftconvolve(v_nl1, h_mid, mode="full")
            v_nl2 = nl2.apply_real_axis(v_pre2) if nl2 else v_pre2
        v_post = fftconvolve(v_nl2, h_post, mode="full")
        t = np.arange(len(v_post)) / fs_pre
        if mode == "complex_baseband":
            env_i = np.real(v_post)
        else:
            env_i, _ = quantum.demodulate(v_post, t, _CARRIER_GHZ * 1e9, lpf_cutoff_hz)
        return float(_ETA * np.trapz(env_i, t))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scales = np.geomspace(scale_lo, scale_hi, n_scan)
        thetas = np.array([theta_for_scale(s) for s in scales])
        idx = int(np.argmax(thetas >= target_theta))
        if thetas[idx] < target_theta:
            return None, False, float(thetas.max())

        lo, hi = scales[max(idx - 1, 0)], scales[idx]
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            th = theta_for_scale(mid)
            if th < target_theta:
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
        theta_hit = theta_for_scale(scale)

    cal_shape = ref_shape * scale
    source_cal = _source_from_shape(cal_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    nl_spec = {}
    if nl1_fn:
        nl_spec[_NODE_1] = nl1_fn.spec()
    if nl2_fn:
        nl_spec[_NODE_2] = nl2_fn.spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = engine.run(
            schematic, source_cal, nonlinear=nl_spec if nl_spec else None,
            noise=None, n_realizations=1, mode=mode,
        )
    return result, True, theta_hit


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

    result_single, achieved_single, _ = _calibrate_two_stage(
        two_amp_schematic, "real_axis", None, _SalehRealAxisSpec(op1db),
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert not achieved_single, "single-stage at op1db=0.17 should be unreachable"

    result_two, achieved_two, theta_hit = _calibrate_two_stage(
        two_amp_schematic, "real_axis", _SalehRealAxisSpec(op1db), _SalehRealAxisSpec(op1db),
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert achieved_two, "two-stage at the SAME op1db should be reachable"
    assert theta_hit == pytest.approx(np.pi, abs=1e-3)

    fid_two = quantum.gate_fidelity(
        result_two, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X",
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    # Real, bounded, non-floor infidelity -- not F~1 (that would mean the
    # gray zone isn't real) and not down at the ~5e-7 no-NL baseline floor
    # either (that would mean this point isn't actually testing the
    # nonlinear effect), and not catastrophic.
    assert 1e-6 < (1.0 - fid_two.F_avg) < 1e-2


def test_realaxis_two_stage_matches_single_stage_at_mild_compression(two_amp_schematic):
    """Sanity check on the flip side: at MILD compression (op1db=1.0, well
    away from either cliff), single- and two-stage give essentially the
    same (near-floor) infidelity -- the gray zone is specifically a
    deep-compression phenomenon, not a blanket "two stages always worse"."""
    op1db = 1.0
    qmodel = _qubit_model_2lvl()

    result_single, ok1, _ = _calibrate_two_stage(
        two_amp_schematic, "real_axis", None, _SalehRealAxisSpec(op1db), lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    result_two, ok2, _ = _calibrate_two_stage(
        two_amp_schematic, "real_axis", _SalehRealAxisSpec(op1db), _SalehRealAxisSpec(op1db),
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert ok1 and ok2

    fid_single = quantum.gate_fidelity(
        result_single, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X", lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    fid_two = quantum.gate_fidelity(
        result_two, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X", lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert fid_single.F_avg == pytest.approx(fid_two.F_avg, abs=1e-4)


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
    result, achieved, _ = _calibrate_two_stage(two_amp_schematic, "real_axis", None, None, lpf_cutoff_hz=_LPF_CUTOFF_HZ)
    assert achieved
    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X", lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert (1.0 - fid.F_avg) < 1e-6


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
    fn1_bb, fn2_bb = _SalehSpec(op1db, enable_am_pm=False), _SalehSpec(op1db, enable_am_pm=False)
    result_bb, achieved_bb, _ = _calibrate_two_stage(
        two_amp_schematic, "complex_baseband", fn1_bb, fn2_bb,
    )
    assert not achieved_bb, "baseband's two-stage cliff should be above op1db=0.17"

    result_ra, achieved_ra, _ = _calibrate_two_stage(
        two_amp_schematic, "real_axis", _SalehRealAxisSpec(op1db), _SalehRealAxisSpec(op1db),
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert achieved_ra


if __name__ == "__main__":
    pytest.main([__file__])
