"""
Tests for si_qfi.simulation.engine.run(), covering the no-nonlinearity
baseline case against both test schematics (test_schematic_basic.si,
lossless; test_schematic_lossy_T_line.si, frequency-dependent loss), in
both simulation modes.

Convolution/growth convention (see engine.py's _nonlinear_pass docstring)
---------------------------------------------------------------------------
Each segment's convolution is `fftconvolve(v, h, mode="full")` with NO
truncation back down to len(v): a true zero-padded linear convolution,
kept in full rather than clipped to the original waveform's duration (which
would silently discard real, physical late-arriving content). This means
v_nl_qubit is longer than the initial drive waveform by len(h)-1 samples
per segment -- with zero NL nodes (this file), there's exactly one segment
(source_label -> qubit_label), so v_nl_qubit has length
len(v_initial) + len(h) - 1, matching a plain fftconvolve(v_initial, h,
mode="full") computed independently as a ground-truth reference.

Schematic paths are resolved to absolute paths at MODULE level, before any
schematic is opened -- SignalIntegrityAppHeadless.OpenProjectFile() changes
the process's current working directory as a side effect (to the
schematic's own folder), so a relative path resolved lazily after another
schematic has already been opened would silently resolve against the wrong
directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

pytest.importorskip("SignalIntegrity")

from si_qfi.schematic import loader as si_loader
from si_qfi.schematic import transfer_function as si_tf
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform

BASIC_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_basic.si").resolve()
LOSSY_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_lossy_T_line.si").resolve()

_CARRIER_GHZ = 5.0
_GAIN_TOL = 5e-3   # relative tolerance for the peak-magnitude gain check


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_schematic():
    return si_loader.load_schematic(BASIC_SCHEMATIC_PATH)


@pytest.fixture
def lossy_schematic():
    return si_loader.load_schematic(LOSSY_SCHEMATIC_PATH)


@pytest.fixture
def source_waveform():
    """
    A drive waveform at the schematic's own native rate (40 GSa/s) --
    matches test_schematic_hookup.py's source_waveform fixture.
    """
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    fs = 4e9
    n = 50
    t = np.arange(n) / fs
    values = np.exp(-((t - t[-1] / 2) ** 2) / (2 * (t[-1] / 6) ** 2)).astype(complex)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(values))
    return SourceWaveform(carrier_freq_ghz=_CARRIER_GHZ, envelope=envelope)


# ---------------------------------------------------------------------------
# Shared assertion helper for the 4 baseline (no-NL) scenarios
# ---------------------------------------------------------------------------

def _check_no_nonlinearity_run(schematic, source, mode):
    """
    Run engine.run() with no nonlinear/noise annotations and check:
      1. v_nl_qubit is the full, untruncated linear convolution of the
         (possibly resampled) drive waveform with the single source->qubit
         segment's impulse response -- proves no truncation/circular-
         convolution artifact (item 1).
      2. Its peak magnitude is the drive waveform's own peak magnitude
         scaled by |H(carrier_hz)| -- the physically expected "pulse with
         its amplitude (and phase) changed" result, for both the flat
         (lossless) and frequency-dependent-loss schematics.
      3. result.extra["intermediate_waveforms"] contains exactly the
         source and qubit labels with the expected lengths (item 2/3).
    Returns the SimulationResult for any additional caller-side checks.
    """
    result = engine.run(schematic, source, mode=mode, n_realizations=1)

    raw = si_tf._extract_single_tf(
        schematic.si_app, schematic.source_label, schematic.qubit_probe_label,
        schematic.source_label,
    )
    tf = si_tf.compute_impulse_response(
        raw, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz,
    )

    if mode == "real_axis":
        fs_native = si_tf.native_sample_rate(raw)
        _, v_initial = source.rf_waveform_at(fs_native)
    else:
        v_initial = source.envelope_complex

    # (1) Full, untruncated linear convolution -- no truncation, no circular
    # wraparound artifact.
    expected_len = len(v_initial) + len(tf.h) - 1
    assert len(result.v_nl_qubit) == expected_len
    expected_fs = source.fs if mode == "complex_baseband" else si_tf.native_sample_rate(raw)
    assert np.isclose(result.fs, expected_fs)
    reference = fftconvolve(v_initial, tf.h, mode="full")
    assert np.allclose(result.v_nl_qubit, reference, rtol=1e-8, atol=1e-12)

    # (2) Physically expected gain: peak magnitude scaled by |H(carrier)|.
    # Peak-magnitude (not same-index) comparison is deliberately
    # delay/position-invariant -- see test_schematic_hookup.py's low-fs
    # baseband tests for why: the channel's own delay moves the output
    # peak's index relative to the input's, so comparing at a fixed index
    # would be comparing the wrong samples.
    h_at_carrier = raw.H[np.argmin(np.abs(raw.freqs - source.carrier_freq_hz))]
    expected_peak = np.abs(h_at_carrier) * np.max(np.abs(v_initial))
    assert np.isclose(np.max(np.abs(result.v_nl_qubit)), expected_peak, rtol=_GAIN_TOL)

    # (3) Intermediate-waveform plumbing (item 2/3): with zero NL nodes,
    # there's exactly one segment, so exactly the source and qubit labels
    # are recorded, with the input's own length and the grown output's
    # length respectively.
    intermediates = result.extra["intermediate_waveforms"]
    assert set(intermediates.keys()) == {schematic.source_label, schematic.qubit_probe_label}
    assert len(intermediates[schematic.source_label]) == len(v_initial)
    assert np.array_equal(intermediates[schematic.source_label], v_initial)
    assert len(intermediates[schematic.qubit_probe_label]) == expected_len
    assert np.array_equal(intermediates[schematic.qubit_probe_label], result.v_nl_qubit)

    return result


# ---------------------------------------------------------------------------
# The 4 baseline (no-nonlinearity) tests
# ---------------------------------------------------------------------------

def test_run_no_nonlinearity_basic_real_axis(basic_schematic, source_waveform):
    """Lossless schematic, real-axis mode: flat 2.5x gain, pure delay."""
    result = _check_no_nonlinearity_run(basic_schematic, source_waveform, "real_axis")
    assert result.v_nl_qubit.dtype == np.float64


def test_run_no_nonlinearity_basic_baseband(basic_schematic, source_waveform):
    """Lossless schematic, complex-baseband mode: flat 2.5x gain."""
    result = _check_no_nonlinearity_run(basic_schematic, source_waveform, "complex_baseband")
    assert np.iscomplexobj(result.v_nl_qubit)


def test_run_no_nonlinearity_lossy_real_axis(lossy_schematic, source_waveform):
    """
    Frequency-dependent-loss schematic, real-axis mode. The channel isn't
    flat over its full 0-20GHz sweep, but the drive's own bandwidth
    (~150MHz around the 5GHz carrier) is narrow enough that the narrowband
    (H(carrier)-scaled) approximation still holds to within _GAIN_TOL.
    """
    result = _check_no_nonlinearity_run(lossy_schematic, source_waveform, "real_axis")
    assert result.v_nl_qubit.dtype == np.float64


def test_run_no_nonlinearity_lossy_baseband(lossy_schematic, source_waveform):
    """Frequency-dependent-loss schematic, complex-baseband mode."""
    result = _check_no_nonlinearity_run(lossy_schematic, source_waveform, "complex_baseband")
    assert np.iscomplexobj(result.v_nl_qubit)


# ---------------------------------------------------------------------------
# NL-node partitioning: an identity ("minimal") nonlinearity at DriverOutput
# splits the single source->qubit segment into two (source->DriverOutput,
# DriverOutput->qubit). extract_all_transfer_functions() builds segment
# transfer functions as ratios (see transfer_function.py's module docstring
# / _extract_single_tf), so H(source->DriverOutput) * H(DriverOutput->qubit)
# == H(source->qubit) *exactly* -- an identity NL node in between should
# therefore reproduce the same physical result as the direct, zero-NL-node
# case.
#
# Note: with no truncation (see _nonlinear_pass), the two-segment chain's
# v_nl_qubit is LONGER than the one-segment case's (it grows once per
# segment), so this can't be a direct array-equality check -- instead we
# compare the physically meaningful quantity both cases already satisfy
# independently in the baseline tests above: peak magnitude scaled by
# |H(carrier)|.
#
# The identity NL model differs by mode (registry.py allows 'volterra' or
# 'saleh' for real_axis, and 'saleh' for baseband):
#   - baseband: SalehModel(alpha_a=1.0, beta_a=0.0) -- gain(A) = 1.0
#     constant, no AM/PM -- apply_baseband(u) == u exactly.
#   - real_axis: VolterraModel('diagonal', coefficients=[[1.0]], orders=[1],
#     memory_depth=0) -- apply_real_axis(v) == v exactly.
# ---------------------------------------------------------------------------

def _identity_nl_spec(mode: str) -> dict:
    if mode == "real_axis":
        return {
            "DriverOutput": {
                "model": "volterra", "option": "diagonal",
                "coefficients": [[1.0]], "orders": [1], "memory_depth": 0,
            },
        }
    return {"DriverOutput": {"model": "saleh", "alpha_a": 1.0, "beta_a": 0.0}}


def _check_identity_nl_partitioning(schematic, source, mode):
    """
    Run engine.run() with an identity nonlinearity annotated at DriverOutput
    and check that it reproduces the direct (zero-NL-node) case's physical
    gain, and that the chain was actually partitioned into two segments
    (item 2/3 plumbing: 3 intermediate-waveform entries, correct lengths).
    """
    nl_spec = _identity_nl_spec(mode)
    result_nl = engine.run(schematic, source, nonlinear=nl_spec, mode=mode, n_realizations=1)
    result_direct = engine.run(schematic, source, nonlinear=None, mode=mode, n_realizations=1)

    source_label, mid_label, qubit_label = (
        schematic.source_label, "DriverOutput", schematic.qubit_probe_label,
    )

    raw_seg1 = si_tf._extract_single_tf(schematic.si_app, source_label, mid_label, source_label)
    raw_seg2 = si_tf._extract_single_tf(schematic.si_app, mid_label, qubit_label, source_label)
    h1 = si_tf.compute_impulse_response(raw_seg1, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz).h
    h2 = si_tf.compute_impulse_response(raw_seg2, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz).h

    if mode == "real_axis":
        fs_native = si_tf.native_sample_rate(raw_seg1)
        _, v_initial = source.rf_waveform_at(fs_native)
    else:
        v_initial = source.envelope_complex

    # Two segments -> two rounds of growth (identity NL doesn't change length).
    expected_len = len(v_initial) + len(h1) - 1 + len(h2) - 1
    assert len(result_nl.v_nl_qubit) == expected_len

    # Physically-meaningful comparison to the direct (one-segment) case:
    # both should hit the same peak magnitude, since the identity NL node
    # contributes nothing beyond what the two segments already multiply
    # back to exactly.
    peak_nl = np.max(np.abs(result_nl.v_nl_qubit))
    peak_direct = np.max(np.abs(result_direct.v_nl_qubit))
    assert np.isclose(peak_nl, peak_direct, rtol=_GAIN_TOL)

    # Intermediate-waveform plumbing: now genuinely a 3-node chain.
    intermediates = result_nl.extra["intermediate_waveforms"]
    assert set(intermediates.keys()) == {source_label, mid_label, qubit_label}
    assert len(intermediates[source_label]) == len(v_initial)
    assert len(intermediates[mid_label]) == len(v_initial) + len(h1) - 1
    assert len(intermediates[qubit_label]) == expected_len
    assert np.array_equal(intermediates[qubit_label], result_nl.v_nl_qubit)

    return result_nl


def test_run_identity_nonlinearity_basic_real_axis(basic_schematic, source_waveform):
    _check_identity_nl_partitioning(basic_schematic, source_waveform, "real_axis")


def test_run_identity_nonlinearity_basic_baseband(basic_schematic, source_waveform):
    _check_identity_nl_partitioning(basic_schematic, source_waveform, "complex_baseband")


def test_run_identity_nonlinearity_lossy_real_axis(lossy_schematic, source_waveform):
    _check_identity_nl_partitioning(lossy_schematic, source_waveform, "real_axis")


def test_run_identity_nonlinearity_lossy_baseband(lossy_schematic, source_waveform):
    _check_identity_nl_partitioning(lossy_schematic, source_waveform, "complex_baseband")


# ---------------------------------------------------------------------------
# NL model x mode validation, and real (non-identity) nonlinearity stress
# tests against DriverOutput on the basic (lossless) schematic. One
# schematic is enough here -- the NL model only ever sees the instantaneous
# amplitude reaching it, so which schematic supplies that amplitude doesn't
# change whether the model math/mode-validation is correct (that channel
# extraction/segmentation math is already covered by the identity-NL and
# baseline tests above, against both schematics).
#
# registry.py's mode restrictions (also see engine.py's _validate_mode,
# which pre-empts the volterra+baseband case specifically before any
# schematic-dependent work happens):
#   - 'volterra'   -> real_axis only
#   - 'saleh'      -> either mode (dispatches to SalehModel for
#                      complex_baseband, SalehRealAxisModel for real_axis --
#                      same spec-dict keys either way, see nonlinear/saleh.py)
# ---------------------------------------------------------------------------

def _pre_nl_waveform(schematic, source, mode, mid_label="DriverOutput"):
    """
    Independently compute the waveform arriving at `mid_label` BEFORE any
    nonlinearity is applied -- i.e. source_label's drive waveform convolved
    once with the source_label->mid_label segment's impulse response. Used
    as the input to an independently-recomputed NL formula, so the stress
    tests below check the engine's actual output against a ground truth
    that doesn't call any si_qfi.nonlinear code at all.
    """
    raw_seg1 = si_tf._extract_single_tf(
        schematic.si_app, schematic.source_label, mid_label, schematic.source_label,
    )
    h1 = si_tf.compute_impulse_response(
        raw_seg1, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz,
    ).h
    if mode == "real_axis":
        fs_native = si_tf.native_sample_rate(raw_seg1)
        _, v_initial = source.rf_waveform_at(fs_native)
    else:
        v_initial = source.envelope_complex
    return np.convolve(v_initial, h1, mode="full")


def _describing_coeffs(op1db=None, oip3=None):
    """
    Independently recompute the a1/a3 coefficients
    nonlinear/volterra.py's _build_describing_coeff() derives from ONE
    OUTPUT-referred op1db_amplitude/oip3_amplitude (all nonlinearity specs
    in this codebase are output-referred by convention -- see its
    docstring), for comparison against the engine's actual output. Returns
    (orders, coeffs). Exactly one of op1db/oip3 must be given (matches
    VolterraModel's own 'describing' option, which no longer supports
    fitting both at once). a1 is always 1.0 -- VolterraModel takes no
    small_signal_gain/gain argument at all (purely output-referred
    nonlinearity, see nonlinear/volterra.py module docstring).
    """
    assert (op1db is None) != (oip3 is None), "exactly one of op1db/oip3"
    one_db_ratio = 10 ** (-1.0 / 20.0)
    a1 = 1.0
    if oip3 is not None:
        a3 = -(4.0 / 3.0) * a1 / oip3**2
    else:
        op1db_in = op1db / (a1 * one_db_ratio)
        a3 = a1 * (one_db_ratio - 1.0) / op1db_in**2
    return [1, 3], [a1, a3]


def test_run_volterra_rejected_in_baseband_mode(basic_schematic, source_waveform):
    """volterra is real_axis-only -- engine._validate_mode rejects it early,
    before any schematic-dependent extraction happens."""
    nl_spec = {
        "DriverOutput": {
            "model": "volterra", "option": "diagonal",
            "coefficients": [[1.0]], "orders": [1],
        },
    }
    with pytest.raises(ValueError, match="volterra' requires mode='real_axis'"):
        engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="complex_baseband")


def _saleh_realaxis_coeffs(op1db=None, oip3=None):
    """
    Independently recompute SalehRealAxisModel's beta_a from ONE
    OUTPUT-referred op1db/oip3 -- mirrors nonlinear/saleh.py's
    from_op1db_oip3(), including the real-axis 4/3 OIP3 factor (see its
    module docstring -- this differs from the baseband SalehModel's factor
    of 1.0). alpha_a is always 1.0 -- from_op1db_oip3() takes no gain
    argument at all (purely output-referred nonlinearity, PRD §3.6), so
    oip3_in == oip3 exactly and op1db_in == op1db/one_db_ratio. Exactly one
    of op1db/oip3 must be given (from_op1db_oip3() no longer supports both).
    """
    assert (op1db is None) != (oip3 is None), "exactly one of op1db/oip3"
    one_db_ratio = 10 ** (-1.0 / 20.0)
    if oip3 is not None:
        beta_a = (4.0 / 3.0) / oip3**2
    else:
        op1db_in = op1db / one_db_ratio
        beta_a = (10 ** (1.0 / 20.0) - 1.0) / op1db_in**2
    return beta_a


def test_run_saleh_realaxis_compresses(basic_schematic, source_waveform):
    """
    Real-axis mode, 'saleh' model (dispatches to SalehRealAxisModel), driven
    into meaningful compression -- same oip3_amplitude and headroom
    reasoning as test_run_volterra_describing_compresses_real_axis, checked
    against the same 1/(1+beta_a*x^2) formula computed independently here
    (alpha_a is always 1.0 -- see _saleh_realaxis_coeffs).
    """
    oip3 = 8.0
    nl_spec = {
        "DriverOutput": {"model": "saleh", "oip3_amplitude": oip3},
    }
    result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    beta_a = _saleh_realaxis_coeffs(oip3=oip3)
    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "real_axis")
    v_expected = v_pre / (1.0 + beta_a * v_pre**2)

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)
    assert np.max(np.abs(v_actual)) < np.max(np.abs(v_pre))

    from si_qfi.nonlinear.saleh import SalehRealAxisModel
    model = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=oip3)
    assert np.max(np.abs(v_pre)) < model.max_monotonic_amplitude


def test_run_saleh_realaxis_warns_when_overdriven(basic_schematic, source_waveform):
    """Same schematic/mode, oip3_amplitude chosen (mirroring the analogous
    Volterra test) so that max_monotonic_amplitude sits below the ~2.5 peak
    amplitude actually reaching DriverOutput -- SalehRealAxisModel's
    overdrive check should fire."""
    oip3 = 1.3
    nl_spec = {
        "DriverOutput": {"model": "saleh", "oip3_amplitude": oip3},
    }
    with pytest.warns(UserWarning, match="exceeds the amplitude"):
        result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    beta_a = _saleh_realaxis_coeffs(oip3=oip3)
    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "real_axis")
    v_expected = v_pre / (1.0 + beta_a * v_pre**2)
    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)


def test_run_saleh_realaxis_compresses_op1db(basic_schematic, source_waveform):
    """
    Real-axis mode, 'saleh' model, fit from op1db_amplitude alone --
    companion to test_run_saleh_realaxis_compresses above (same model/mode,
    other fit branch). The op1db branch of from_op1db_oip3() doesn't use
    the real-axis 4/3 OIP3 factor (see _saleh_realaxis_coeffs' docstring),
    so beta_a here comes out identical to the baseband op1db=1.0 fit used
    in test_run_saleh_baseband_compresses_op1db -- not a coincidence, just
    means this particular check exercises the shared (non-OIP3) code path.
    """
    op1db = 1.0
    nl_spec = {
        "DriverOutput": {"model": "saleh", "op1db_amplitude": op1db},
    }
    result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    beta_a = _saleh_realaxis_coeffs(op1db=op1db)
    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "real_axis")
    v_expected = v_pre / (1.0 + beta_a * v_pre**2)

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)
    assert np.max(np.abs(v_actual)) < np.max(np.abs(v_pre))

    from si_qfi.nonlinear.saleh import SalehRealAxisModel
    model = SalehRealAxisModel.from_op1db_oip3(op1db_amplitude=op1db)
    assert np.max(np.abs(v_pre)) < model.max_monotonic_amplitude


def test_run_volterra_describing_compresses_real_axis(basic_schematic, source_waveform):
    """
    Real-axis mode, volterra 'describing' option, driven into meaningful
    compression (peak amplitude at DriverOutput is ~2.5; oip3_amplitude=8.0,
    OUTPUT-referred, keeps that comfortably below this model's own
    max_monotonic_amplitude (4.0 for this value) so the result is realistic
    compression, not the "pushed past where the polynomial breaks down"
    regime covered separately by test_run_volterra_describing_warns_when_overdriven
    below.

    Checked against the *same* cubic describing-function formula
    (y = a1*x + a3*x^3, with a1/a3 derived exactly as
    nonlinear/volterra.py's _build_describing_coeff() does, including the
    output-referred -> input-referred conversion -- see its docstring)
    computed independently here, not by calling any si_qfi.nonlinear code.
    """
    oip3 = 8.0
    nl_spec = {
        "DriverOutput": {
            "model": "volterra", "option": "describing",
            "oip3_amplitude": oip3, "memory_depth": 0,
        },
    }
    result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    orders, (a1, a3) = _describing_coeffs(oip3=oip3)
    assert orders == [1, 3]

    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "real_axis")
    v_expected = a1 * v_pre + a3 * v_pre**3

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)
    # Confirm this is actually stressing the model, not accidentally
    # linear: peak output is measurably below the small-signal (a1=1,
    # fixed) prediction (~13% compression at these parameters).
    assert np.max(np.abs(v_actual)) < 0.9 * np.max(np.abs(v_pre))
    # And confirm we really did stay within the model's own valid range --
    # this test is specifically NOT about the overdriven regime.
    from si_qfi.nonlinear.volterra import VolterraModel
    model = VolterraModel(option="describing", oip3_amplitude=oip3, memory_depth=0)
    assert np.max(np.abs(v_pre)) < model.max_monotonic_amplitude


def test_run_volterra_describing_warns_when_overdriven(basic_schematic, source_waveform):
    """
    Same schematic/mode, but oip3_amplitude=3.0 puts this model's
    max_monotonic_amplitude at 1.5 -- well below the ~2.5 peak amplitude
    actually reaching DriverOutput -- so VolterraModel's overdrive check
    (nonlinear/volterra.py's _warn_if_overdriven) should fire. The engine
    still runs (this is a warning, not an error); the math itself is still
    exactly what the (now non-physical, sign-flipping) polynomial predicts,
    matching the same cubic formula.
    """
    oip3 = 3.0
    nl_spec = {
        "DriverOutput": {
            "model": "volterra", "option": "describing",
            "oip3_amplitude": oip3, "memory_depth": 0,
        },
    }
    with pytest.warns(UserWarning, match="exceeds the amplitude"):
        result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    orders, (a1, a3) = _describing_coeffs(oip3=oip3)
    assert orders == [1, 3]

    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "real_axis")
    v_expected = a1 * v_pre + a3 * v_pre**3
    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)


def test_run_volterra_describing_compresses_op1db(basic_schematic, source_waveform):
    """
    Real-axis mode, volterra 'describing' option, fit from op1db_amplitude
    alone -- companion to test_run_volterra_describing_compresses_real_axis
    above (same model/mode, other fit branch). op1db_amplitude=2.0 keeps
    max_monotonic_amplitude at ~3.93, comfortably above the ~2.5 peak
    amplitude actually reaching DriverOutput, so this is realistic
    compression rather than the overdriven regime.
    """
    op1db = 2.0
    nl_spec = {
        "DriverOutput": {
            "model": "volterra", "option": "describing",
            "op1db_amplitude": op1db, "memory_depth": 0,
        },
    }
    result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    orders, (a1, a3) = _describing_coeffs(op1db=op1db)
    assert orders == [1, 3]

    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "real_axis")
    v_expected = a1 * v_pre + a3 * v_pre**3

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)
    # Confirm this is actually stressing the model, not accidentally
    # linear.
    assert np.max(np.abs(v_actual)) < 0.9 * np.max(np.abs(v_pre))
    # And confirm we really did stay within the model's own valid range.
    from si_qfi.nonlinear.volterra import VolterraModel
    model = VolterraModel(option="describing", op1db_amplitude=op1db, memory_depth=0)
    assert np.max(np.abs(v_pre)) < model.max_monotonic_amplitude


def test_run_saleh_baseband_compresses_op1db(basic_schematic, source_waveform):
    """
    Complex-baseband mode, plain Saleh AM-AM only (AM-PM disabled), fit
    from op1db_amplitude alone -- checked against SalehModel.gain() applied
    directly to the independently-computed pre-NL waveform. Companion to
    test_run_saleh_baseband_compresses_oip3 below (same model/mode, other
    fit branch) and to test_run_saleh_ampm_baseband below (same
    op1db_amplitude, but with AM-PM enabled).
    """
    from si_qfi.nonlinear.saleh import SalehModel

    op1db = 1.0
    nl_spec = {"DriverOutput": {"model": "saleh", "op1db_amplitude": op1db}}
    result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="complex_baseband")

    saleh = SalehModel.from_op1db_oip3(op1db_amplitude=op1db)
    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "complex_baseband")
    amp = np.abs(v_pre)
    v_expected = saleh.gain(amp) * v_pre

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)
    peak_amp = np.max(amp)
    assert saleh.gain(np.array([peak_amp]))[0] < 0.8   # meaningful AM-AM compression
    assert peak_amp < saleh.max_monotonic_amplitude


def test_run_saleh_baseband_compresses_oip3(basic_schematic, source_waveform):
    """
    Complex-baseband mode, plain Saleh AM-AM only, fit from oip3_amplitude
    alone -- baseband OIP3-beta factor is 1.0 (no 4/3 correction; that only
    applies to the real-axis variants -- see nonlinear/saleh.py's module
    docstring). Same check style as test_run_saleh_baseband_compresses_op1db.
    """
    from si_qfi.nonlinear.saleh import SalehModel

    oip3 = 8.0
    nl_spec = {"DriverOutput": {"model": "saleh", "oip3_amplitude": oip3}}
    result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="complex_baseband")

    saleh = SalehModel.from_op1db_oip3(oip3_amplitude=oip3)
    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "complex_baseband")
    amp = np.abs(v_pre)
    v_expected = saleh.gain(amp) * v_pre

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)
    assert np.max(np.abs(v_actual)) < np.max(np.abs(v_pre))
    assert np.max(amp) < saleh.max_monotonic_amplitude


def test_run_saleh_ampm_baseband(basic_schematic, source_waveform):
    """
    Complex-baseband mode, Saleh model with AM-PM enabled, driven well past
    P1dB (peak amplitude at DriverOutput is ~2.5, op1db_amplitude=1.0) --
    exercises both AM-AM compression and a real (non-zero) AM-PM phase
    shift, checked against SalehModel's own closed-form gain()/phase_shift()
    applied directly to the independently-computed pre-NL waveform.
    """
    from si_qfi.nonlinear.saleh import SalehModel

    op1db = 1.0
    saleh = SalehModel.from_op1db_oip3(
        op1db_amplitude=op1db,
        enable_am_pm=True, am_pm_peak_deg=15.0,
    )
    nl_spec = {
        "DriverOutput": {
            "model": "saleh", "op1db_amplitude": op1db,
            "enable_am_pm": True, "am_pm_peak_deg": 15.0,
        },
    }
    result = engine.run(basic_schematic, source_waveform, nonlinear=nl_spec, mode="complex_baseband")

    v_pre = _pre_nl_waveform(basic_schematic, source_waveform, "complex_baseband")
    amp = np.abs(v_pre)
    safe_amp = np.where(amp > 0, amp, 1.0)
    v_expected = saleh.gain(amp) * np.exp(1j * saleh.phase_shift(amp)) * (v_pre / safe_amp) * amp

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput"]
    assert np.allclose(v_actual, v_expected, rtol=1e-9)
    # Confirm both AM-AM compression and a real AM-PM shift are present at
    # the actual peak amplitude reaching this node (not just at p1db, where
    # am_pm_peak_deg was calibrated).
    peak_amp = np.max(amp)
    assert saleh.gain(np.array([peak_amp]))[0] < 0.8   # meaningful AM-AM compression
    assert np.rad2deg(saleh.phase_shift(np.array([peak_amp])))[0] > 5.0   # meaningful AM-PM


# ---------------------------------------------------------------------------
# Two-amplifier schematic (tests/test_schematic_lossy_T_line_2_amplifier.si):
#
#     VSource -> R1 -> T3 -> D1 (amp, gain=30) -> DriverOutput -> T1 (lossy)
#         -> D2 (amp, gain=30) -> DriverOutput2 -> T2 (lossy) -> R2 -> VQubit
#
# ViDriver/DriverInput2 are also SI probes (each amp's input node) but
# aren't used as NL nodes here -- the two NL nodes are the amps' *output*
# nodes, DriverOutput and DriverOutput2, matching the single-amp
# schematics' "DriverOutput" convention. Both are annotated with the SAME
# spec dict -- since op1db/oip3_amplitude are OUTPUT-referred (PRD §3.6),
# this models two identical amplifiers (same physical P1dB/IP3 rating),
# not two arbitrarily different ones.
#
# Segment DC gains (verified directly against
# si_tf.extract_all_transfer_functions(schematic, nl_labels=["DriverOutput",
# "DriverOutput2"])):
#     VSource -> DriverOutput       : 7.5   (R1 divider * D1's S21)
#     DriverOutput -> DriverOutput2 : 15.0  (T1 line * D2's S21 -- DriverInput2
#                                      sits electrically in between but isn't
#                                      a declared NL node, so SI's own network
#                                      solve absorbs it into this one segment,
#                                      same "ratio-based segment" math already
#                                      covered by test_schematic_hookup.py)
#     DriverOutput2 -> VQubit       : 1.0   (T2 line, matched at DC)
#
# Because linear gain accumulates through the chain (7.5x, then another
# 15x), the SAME output-referred compression point reused at both nodes is
# reached at a much larger absolute signal level at the second stage -- so
# node 2 always ends up more compressed (often far more) than node 1, even
# though it's nominally "the same amplifier" -- this is ordinary
# cascade-amplifier behavior (a chain's distortion is usually dominated by
# its last stage), verified explicitly below via each node's peak input
# vs. its own max_monotonic_amplitude, not just asserted.
# ---------------------------------------------------------------------------

TWO_AMP_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_lossy_T_line_2_amplifier.si").resolve()


@pytest.fixture
def two_amp_schematic():
    return si_loader.load_schematic(TWO_AMP_SCHEMATIC_PATH)


def _segment_h(schematic, source, mode, label_in, label_out):
    """
    Independently extract+compute the impulse response for an arbitrary
    INTERIOR schematic segment label_in->label_out -- generalizes
    _pre_nl_waveform's segment-1 extraction (which always starts at
    source_label) to a later segment of a multi-NL-node chain, e.g.
    DriverOutput->DriverOutput2 here.
    """
    raw_seg = si_tf._extract_single_tf(schematic.si_app, label_in, label_out, schematic.source_label)
    return si_tf.compute_impulse_response(
        raw_seg, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz,
    ).h


def _harmonic_band_power(v, fs, f_target, half_bw=0.3e9):
    """
    Sum |FFT(v)|^2 in a band around f_target -- used to check for the
    presence (odd harmonics) or absence (even harmonics) of energy at
    n*f_carrier in a real-axis output. The drive is a short
    Gaussian-enveloped pulse rather than a pure/infinite CW tone, so each
    harmonic shows up as a BAND of width ~ the envelope's own bandwidth
    (~150 MHz here) centered at n*f_carrier, not a single bin --
    half_bw=0.3GHz is comfortably wider than that envelope bandwidth and
    comfortably narrower than the 5GHz spacing between harmonics, so
    neighboring harmonic bands don't bleed into each other.
    """
    freqs = np.fft.rfftfreq(len(v), 1.0 / fs)
    V = np.abs(np.fft.rfft(v))
    mask = (freqs > f_target - half_bw) & (freqs < f_target + half_bw)
    return float(np.sum(V[mask] ** 2))


def test_run_saleh_realaxis_two_amp_cascade_oip3(two_amp_schematic, source_waveform):
    """
    Real-axis mode, TWO Saleh NL nodes (DriverOutput, DriverOutput2), the
    SAME oip3_amplitude=8.0 spec reused at both. Checked against the same
    1/(1+beta_a*x^2) formula applied twice in sequence -- once per node,
    with the actual interior segment's own impulse response
    (DriverOutput->DriverOutput2, extracted independently via
    _segment_h()) convolved in between -- not by calling any
    si_qfi.nonlinear code.

    Cascade sanity check: node 1's peak input (~4.22) is comfortably below
    its own max_monotonic_amplitude (6.93 for oip3=8.0), so node 1 alone is
    only modestly compressed; but node 2 -- fed by this segment's own 15x
    linear gain on top of whatever left node 1 -- sees a peak input (~27.4)
    more than 3x past that SAME max_monotonic_amplitude. Same spec, very
    different operating point: exactly the "last stage dominates the
    chain's distortion" cascade behavior described above. Both nodes'
    overdrive warnings are therefore expected to fire.

    Harmonic-content check: SalehRealAxisModel's gain(A) =
    alpha_a/(1+beta_a*A^2) is an ODD function of the real input (g(-A) =
    -g(A)). A real periodic waveform built only from ODD harmonics of f0
    has half-wave symmetry (v(t+T/2) = -v(t) over one period T=1/f0); an
    odd memoryless nonlinearity preserves that symmetry exactly (y(-v) =
    -y(v)), and convolving a half-wave-symmetric signal with any LTI
    impulse response preserves it too. So cascading any number of odd,
    memoryless stages with LTI filtering in between can only ever produce
    MORE odd harmonics (3rd, 5th, ...) -- never a 2nd harmonic or any other
    even order, no matter how many odd harmonics (e.g. node 1's own 3rd
    harmonic) feed into the next stage. Checked directly below: FFT power
    in bands around 2*fc/4*fc sits at the numerical noise floor (<1e-3x the
    weaker of the 1*fc/3*fc bands, empirically ~5-8 orders of magnitude
    below), not elevated by the second nonlinear stage.
    """
    oip3 = 8.0
    spec = {"model": "saleh", "oip3_amplitude": oip3}
    nl_spec = {"DriverOutput": spec, "DriverOutput2": spec}

    with pytest.warns(UserWarning, match="exceeds the amplitude"):
        result = engine.run(two_amp_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    beta_a = _saleh_realaxis_coeffs(oip3=oip3)
    v_pre1 = _pre_nl_waveform(two_amp_schematic, source_waveform, "real_axis")
    v_nl1 = v_pre1 / (1.0 + beta_a * v_pre1 ** 2)

    h2 = _segment_h(two_amp_schematic, source_waveform, "real_axis", "DriverOutput", "DriverOutput2")
    v_pre2 = fftconvolve(v_nl1, h2, mode="full")
    v_nl2 = v_pre2 / (1.0 + beta_a * v_pre2 ** 2)

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput2"]
    assert np.allclose(v_actual, v_nl2, rtol=1e-9)

    from si_qfi.nonlinear.saleh import SalehRealAxisModel
    model = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=oip3)
    assert np.max(np.abs(v_pre1)) < model.max_monotonic_amplitude          # node 1: within range
    assert np.max(np.abs(v_pre2)) > 3 * model.max_monotonic_amplitude     # node 2: well past it

    # No blow-up despite node 2 being overdriven -- SalehRealAxisModel's
    # bounded form just saturates back toward zero for very large inputs,
    # unlike an unbounded polynomial (contrast with the Volterra cascade
    # test below).
    v_final = result.v_nl_qubit
    assert np.all(np.isfinite(v_final))
    assert np.max(np.abs(v_final)) < 10.0

    fc = source_waveform.carrier_freq_hz
    power_odd = [_harmonic_band_power(v_final, result.fs, k * fc) for k in (1, 3)]
    power_even = [_harmonic_band_power(v_final, result.fs, k * fc) for k in (2, 4)]
    for p_even in power_even:
        assert p_even < 1e-3 * min(power_odd)


def test_run_volterra_describing_two_amp_cascade_op1db(two_amp_schematic, source_waveform):
    """
    Real-axis mode, TWO volterra 'describing' NL nodes (DriverOutput,
    DriverOutput2), the SAME op1db_amplitude=2.0 spec reused at both.
    Checked against the same cubic formula (y = a1*x + a3*x^3) applied
    twice in sequence, same pattern as
    test_run_saleh_realaxis_two_amp_cascade_oip3 above.

    Unlike that Saleh case, node 1 alone is ALREADY mildly past its own
    max_monotonic_amplitude here (peak input ~4.22 vs. 3.93) -- op1db=2.0
    was deliberately reused unchanged from the single-amp schematic's
    test_run_volterra_describing_compresses_op1db, where it was a
    comfortably-valid choice; reusing the exact same spec against a
    schematic with an extra 15x-gain second stage pushes node 1 itself
    slightly out of range too. Node 2 (fed by that same 15x gain) ends up
    FAR further out (peak input ~24.3, > 6x max_monotonic_amplitude) --
    the cubic's runaway a3*x^3 term dominates completely there, producing
    a physically-nonsensical, sign-flipping output swing (final peak ~138,
    vs. a peak drive of ~1) -- exactly the "no saturation mechanism"
    breakdown VolterraModel's own overdrive warning describes, and exactly
    why a bounded model (Saleh) is preferable for a node that may see this
    much accumulated gain. Both nodes' overdrive warnings are expected.

    The formula match below still holds exactly even in this
    non-physical regime (it's just evaluating the same well-defined cubic
    polynomial), and -- more importantly -- so does the odd-harmonics-only
    claim from the Saleh cascade test above: the argument there relies
    only on the model being an odd, memoryless function of a
    half-wave-symmetric input, not on staying anywhere near a "sane"
    operating point. Checked here too, in this pathological regime, as a
    stronger form of the same claim.
    """
    op1db = 2.0
    spec = {
        "model": "volterra", "option": "describing",
        "op1db_amplitude": op1db, "memory_depth": 0,
    }
    nl_spec = {"DriverOutput": spec, "DriverOutput2": spec}

    with pytest.warns(UserWarning, match="exceeds the amplitude"):
        result = engine.run(two_amp_schematic, source_waveform, nonlinear=nl_spec, mode="real_axis")

    orders, (a1, a3) = _describing_coeffs(op1db=op1db)
    assert orders == [1, 3]

    v_pre1 = _pre_nl_waveform(two_amp_schematic, source_waveform, "real_axis")
    v_nl1 = a1 * v_pre1 + a3 * v_pre1 ** 3

    h2 = _segment_h(two_amp_schematic, source_waveform, "real_axis", "DriverOutput", "DriverOutput2")
    v_pre2 = fftconvolve(v_nl1, h2, mode="full")
    v_nl2 = a1 * v_pre2 + a3 * v_pre2 ** 3

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput2"]
    assert np.allclose(v_actual, v_nl2, rtol=1e-9)

    from si_qfi.nonlinear.volterra import VolterraModel
    model = VolterraModel(option="describing", op1db_amplitude=op1db, memory_depth=0)
    assert np.max(np.abs(v_pre2)) > 6 * model.max_monotonic_amplitude

    v_final = result.v_nl_qubit
    assert np.all(np.isfinite(v_final))
    # Confirms the "runaway polynomial" breakdown really did happen --
    # final peak is a full ~30x the drive's own peak amplitude (~4.5,
    # after the two segments' combined 7.5x linear gain), not a
    # physically-plausible compressed value.
    assert np.max(np.abs(v_final)) > 50.0

    fc = source_waveform.carrier_freq_hz
    power_odd = [_harmonic_band_power(v_final, result.fs, k * fc) for k in (1, 3)]
    power_even = [_harmonic_band_power(v_final, result.fs, k * fc) for k in (2, 4)]
    for p_even in power_even:
        assert p_even < 1e-3 * min(power_odd)


def test_run_saleh_baseband_two_amp_cascade_op1db(two_amp_schematic, source_waveform):
    """
    Complex-baseband mode, TWO Saleh NL nodes (DriverOutput,
    DriverOutput2), the SAME op1db_amplitude=1.0 spec reused at both --
    checked against SalehModel.gain() applied twice in sequence, same
    cascade pattern as the real-axis tests above (no harmonic check here:
    complex baseband represents the signal as an already-demodulated
    envelope, so there's no literal n*f_carrier tone to look for -- see
    engine.py's _check_harmonic_suppression for the baseband-specific
    proxy check instead, which is a frequency-domain-only heuristic, not
    exercised here).

    Both nodes end up past their own (identical) max_monotonic_amplitude
    (node 1 peak ~4.22 vs. 3.21; node 2 peak ~13.6, ~4x further out) --
    same accumulated-gain cascade effect as the real-axis tests -- but
    SalehModel's bounded gain(A) never diverges, so the final qubit-plane
    output stays a physically-plausible size (~0.9, well below what a
    purely linear 7.5x cascade of the ~4.2-peak drive would give).
    """
    op1db = 1.0
    spec = {"model": "saleh", "op1db_amplitude": op1db}
    nl_spec = {"DriverOutput": spec, "DriverOutput2": spec}

    with pytest.warns(UserWarning, match="exceeds the amplitude"):
        result = engine.run(two_amp_schematic, source_waveform, nonlinear=nl_spec, mode="complex_baseband")

    from si_qfi.nonlinear.saleh import SalehModel
    saleh = SalehModel.from_op1db_oip3(op1db_amplitude=op1db)

    v_pre1 = _pre_nl_waveform(two_amp_schematic, source_waveform, "complex_baseband")
    v_nl1 = saleh.gain(np.abs(v_pre1)) * v_pre1

    h2 = _segment_h(two_amp_schematic, source_waveform, "complex_baseband", "DriverOutput", "DriverOutput2")
    v_pre2 = fftconvolve(v_nl1, h2, mode="full")
    v_nl2 = saleh.gain(np.abs(v_pre2)) * v_pre2

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput2"]
    assert np.allclose(v_actual, v_nl2, rtol=1e-9)

    assert np.max(np.abs(v_pre2)) > 3 * np.max(np.abs(v_pre1))
    v_final = result.v_nl_qubit
    assert np.all(np.isfinite(v_final))
    assert np.max(np.abs(v_final)) < 5.0


def test_run_saleh_baseband_two_amp_cascade_oip3(two_amp_schematic, source_waveform):
    """
    Complex-baseband mode, TWO Saleh NL nodes, the SAME oip3_amplitude=8.0
    spec reused at both -- companion to
    test_run_saleh_baseband_two_amp_cascade_op1db above (same cascade,
    other fit branch). Node 1's peak input (~4.22) is within its own
    max_monotonic_amplitude (8.0 for oip3=8.0), same as the analogous
    real-axis oip3 test; node 2's (~27.9) is not.
    """
    oip3 = 8.0
    spec = {"model": "saleh", "oip3_amplitude": oip3}
    nl_spec = {"DriverOutput": spec, "DriverOutput2": spec}

    with pytest.warns(UserWarning, match="exceeds the amplitude"):
        result = engine.run(two_amp_schematic, source_waveform, nonlinear=nl_spec, mode="complex_baseband")

    from si_qfi.nonlinear.saleh import SalehModel
    saleh = SalehModel.from_op1db_oip3(oip3_amplitude=oip3)

    v_pre1 = _pre_nl_waveform(two_amp_schematic, source_waveform, "complex_baseband")
    v_nl1 = saleh.gain(np.abs(v_pre1)) * v_pre1

    h2 = _segment_h(two_amp_schematic, source_waveform, "complex_baseband", "DriverOutput", "DriverOutput2")
    v_pre2 = fftconvolve(v_nl1, h2, mode="full")
    v_nl2 = saleh.gain(np.abs(v_pre2)) * v_pre2

    v_actual = result.extra["intermediate_waveforms"]["DriverOutput2"]
    assert np.allclose(v_actual, v_nl2, rtol=1e-9)

    assert np.max(np.abs(v_pre1)) < saleh.max_monotonic_amplitude
    assert np.max(np.abs(v_pre2)) > 3 * saleh.max_monotonic_amplitude
    v_final = result.v_nl_qubit
    assert np.all(np.isfinite(v_final))
    assert np.max(np.abs(v_final)) < 5.0


# ---------------------------------------------------------------------------
# Item 1 regression: full, zero-padded convolution -- no SI schematic
# needed, exercises _nonlinear_pass directly with a synthetic
# TransferFunction (core math works without SI installed).
# ---------------------------------------------------------------------------

def test_nonlinear_pass_keeps_full_zero_padded_convolution():
    """
    _nonlinear_pass's convolution must be a true linear (zero-padded)
    convolution, kept in full -- not truncated back to len(v), and not a
    circular/wraparound convolution. Uses an h longer than v (as SI's own
    native impulse responses typically are relative to a short drive
    pulse) specifically because a circular-convolution bug would only be
    visible in that regime.
    """
    from si_qfi.schematic.transfer_function import TransferFunction
    from si_qfi.simulation.engine import _nonlinear_pass

    rng = np.random.default_rng(0)
    h = rng.normal(size=37) + 1j * rng.normal(size=37)   # longer than v
    v_in = rng.normal(size=12) + 1j * rng.normal(size=12)

    tf = TransferFunction(
        label_in="VSource", label_out="VQubit", si_frequency_response=None, h=h, dt=1.0,
    )
    segment_tfs = {("VSource", "VQubit"): tf}

    v_out, intermediates = _nonlinear_pass(
        v_in, nl_labels=[], nl_nodes={}, segment_tfs=segment_tfs,
        mode="complex_baseband", sim_warnings=[], source_label="VSource", qubit_label="VQubit",
    )

    expected = np.convolve(v_in, h, mode="full")   # unambiguous ground truth
    assert len(v_out) == len(v_in) + len(h) - 1
    assert np.allclose(v_out, expected, rtol=1e-10, atol=1e-12)
    assert set(intermediates.keys()) == {"VSource", "VQubit"}
    assert np.array_equal(intermediates["VSource"], v_in)
    assert np.array_equal(intermediates["VQubit"], v_out)


if __name__ == "__main__":
    pytest.main([__file__])
