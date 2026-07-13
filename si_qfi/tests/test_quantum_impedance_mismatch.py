"""
tests/test_quantum_impedance_mismatch.py
==========================================
Regression tests for examples/impedance_mismatch_demo.py's finding:
impedance mismatch between the amplifier and the qubit costs real gate
fidelity ONLY when there's both (a) a genuine reflection coefficient
(Zmismatch != 50 ohm) AND (b) enough propagation delay for the reflection
to separate in time from the forward wave (round-trip delay comparable to
or longer than the pulse duration). Neither condition alone is enough.

Also exercises si_qfi.schematic.loader.load_schematic()'s new `variables=`
parameter -- the first place in this codebase's test suite a schematic is
parametrized from Python rather than fully fixed in the .si file. See
tests/test_schematic_impedance_mismatch.si and loader.py's docstring for
the underlying SI mechanism (SignalIntegrityAppHeadless.OpenProjectFile's
own `args=` parameter).

nonlinear=None throughout -- this is a purely linear reflection effect.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")
qutip = pytest.importorskip("qutip")

# The reference (amp=1.0) pulse's crude Nyquist-based narrowband-ratio
# estimate (bandwidth_hz = FS_ENVELOPE/2, not the pulse's true occupied
# bandwidth) always trips this heuristic at FS_ENVELOPE=8GHz -- benign,
# same as in tests/test_quantum_nonlinear.py and the other quantum test
# files, not a real accuracy problem (this whole investigation is a
# purely linear effect, already established mode-independent).
warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

from si_qfi.schematic import loader as si_loader
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform, build_gaussian_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_impedance_mismatch.si").resolve()

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6
_DURATION_S = 100e-9
_SIGMA_S = _DURATION_S / 6
_FS_ENVELOPE = 8e9


def _qubit_model_2lvl():
    return quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)


def _source_from_shape(shape: np.ndarray, fs: float, carrier_ghz: float) -> SourceWaveform:
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def _infidelity_at(zmismatch, tprop_s, qmodel, mode="complex_baseband"):
    schematic = si_loader.load_schematic(
        SCHEMATIC_PATH, variables={"Zmismatch": zmismatch, "Tprop": tprop_s},
    )
    ref_shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    result_ref = engine.run(schematic, source_ref, nonlinear=None, noise=None, n_realizations=1, mode=mode)
    v = np.asarray(result_ref.v_nl_qubit)
    t = np.arange(len(v)) / result_ref.fs
    theta_ref = float(_ETA * np.trapz(np.real(v), t))
    scale = np.pi / theta_ref

    cal_shape = ref_shape * scale
    source_cal = _source_from_shape(cal_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    result_cal = engine.run(schematic, source_cal, nonlinear=None, noise=None, n_realizations=1, mode=mode)
    fid = quantum.gate_fidelity(result_cal, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X")
    return 1.0 - fid.F_avg


# ---------------------------------------------------------------------------
# Schematic parametrization mechanism itself (independent of the physics)
# ---------------------------------------------------------------------------

def test_default_variables_reproduce_matched_baseline():
    """With no variables override, Zmismatch defaults to 50.0 and Tprop to
    1e-9 (matching test_schematic_basic.si's original fixed values) -- the
    schematic should behave exactly like a matched, reflection-free line:
    flat 2.5x VSource->VQubit gain, no ripple."""
    from si_qfi.schematic import transfer_function as si_tf

    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    tf = si_tf._extract_single_tf(
        schematic.si_app, schematic.source_label, schematic.qubit_probe_label, schematic.source_label,
    )
    mag = np.abs(tf.H)
    assert np.allclose(mag, 2.5, rtol=1e-6)


def test_variables_override_changes_transfer_function():
    """Passing a genuine mismatch should visibly change the extracted
    transfer function (frequency-dependent ripple from reflections) --
    confirms load_schematic(variables=...) actually reaches the SI
    network solve, not just cosmetically accepted and ignored."""
    from si_qfi.schematic import transfer_function as si_tf

    schematic = si_loader.load_schematic(
        SCHEMATIC_PATH, variables={"Zmismatch": 150.0, "Tprop": 3e-9},
    )
    tf = si_tf._extract_single_tf(
        schematic.si_app, schematic.source_label, schematic.qubit_probe_label, schematic.source_label,
    )
    mag = np.abs(tf.H)
    ripple = (mag.max() - mag.min()) / mag.mean()
    assert ripple > 0.1   # matched case's ripple is ~1e-15 (float noise)


def test_reflection_coefficient_symmetric_around_50_ohm():
    """Physics sanity check: the reflection coefficient (and therefore the
    infidelity) depends only on |Z-50|/|Z+50|, not on which side of 50 ohm
    Z sits -- Z=25 and Z=100 (both Gamma=1/3) should give the identical
    infidelity, confirmed directly rather than assumed."""
    qmodel = _qubit_model_2lvl()
    tprop = 200e-9
    infid_low = _infidelity_at(25.0, tprop, qmodel)
    infid_high = _infidelity_at(100.0, tprop, qmodel)
    assert infid_low == pytest.approx(infid_high, rel=1e-3)


# ---------------------------------------------------------------------------
# The physics: BOTH mismatch AND delay are required
# ---------------------------------------------------------------------------

def test_matched_impedance_negligible_regardless_of_delay():
    """Zmismatch=50 (Gamma=0 exactly) should stay at the numerical floor
    even at a long delay -- no reflection coefficient, no reflection,
    regardless of timing."""
    qmodel = _qubit_model_2lvl()
    infid_short = _infidelity_at(50.0, 5e-9, qmodel)
    infid_long = _infidelity_at(50.0, 200e-9, qmodel)
    assert infid_short < 1e-6
    assert infid_long < 1e-6


def test_severe_mismatch_negligible_at_short_delay():
    """The direct test of 'mismatch alone is not enough': even a severe
    mismatch (300 ohm, Gamma~0.71) should cost essentially nothing when
    the round-trip delay is much shorter than the pulse duration (here
    2*5ns=10ns vs. a 100ns pulse)."""
    qmodel = _qubit_model_2lvl()
    infid = _infidelity_at(300.0, 5e-9, qmodel)
    assert infid < 1e-5


def test_severe_mismatch_significant_at_long_delay():
    """Same severe mismatch, but with round-trip delay (2*200ns=400ns)
    well past the 100ns pulse duration -- should cost real, measurable
    fidelity, several orders of magnitude above the short-delay case."""
    qmodel = _qubit_model_2lvl()
    infid = _infidelity_at(300.0, 200e-9, qmodel)
    assert infid > 1e-2


def test_infidelity_grows_monotonically_with_delay_at_fixed_mismatch():
    """At a fixed, genuine mismatch, infidelity should increase
    monotonically as round-trip delay grows from well-within the pulse to
    well-beyond it -- a smooth transition, not a random walk."""
    qmodel = _qubit_model_2lvl()
    tprops_ns = [1, 25, 75, 150, 300]
    infidelities = [_infidelity_at(150.0, t * 1e-9, qmodel) for t in tprops_ns]
    assert infidelities == sorted(infidelities)


def test_infidelity_grows_monotonically_with_mismatch_at_long_delay():
    """At a fixed long delay, infidelity should increase monotonically
    with the reflection coefficient -- a real, graded dependence on
    mismatch severity, not a threshold effect."""
    qmodel = _qubit_model_2lvl()
    z_values = [50.0, 70.0, 100.0, 150.0, 250.0]
    infidelities = [_infidelity_at(z, 200e-9, qmodel) for z in z_values]
    assert infidelities == sorted(infidelities)


if __name__ == "__main__":
    pytest.main([__file__])
