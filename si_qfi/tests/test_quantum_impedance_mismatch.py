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

Monotonicity checks below tolerate a small numerical floor (see
_assert_monotonic_within_floor) rather than requiring bit-exact ordering --
confirmed directly that a strict `==sorted(...)` check on this schematic can
fail purely from float64 noise when the true infidelity is deep in the
numerical floor (~1e-8) for several consecutive sweep points, e.g. observed
values like [2.07e-8, -2.83e-8, ...] that cross zero from floating-point
error alone, nothing physical.

test_infidelity_extremely_sensitive_to_subcarrier_delay_phase locks in a
second, much larger finding (see examples/impedance_mismatch_demo.py's
Panel D and INVESTIGATIONS.md Section 5): every Tprop value used elsewhere
in this file is a whole number of nanoseconds, and because the carrier is
exactly 5GHz, that always lands the reflected echo's carrier phase on the
safest possible point (0 mod 2*pi). A Tprop off that whole-ns grid by a mere
picoseconds can be catastrophic (infidelity up to ~0.5 observed) -- genuine
waveform distortion (confirmed NOT fixable by a better amplitude/phase
calibration -- see complex_calibrated_infidelity_at() in the demo script),
not a numerical artifact of any particular Tprop grid choice.
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
from si_qfi.source.waveform import build_gaussian_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_impedance_mismatch.si").resolve()

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6
_DURATION_S = 100e-9
_SIGMA_S = _DURATION_S / 6
_FS_ENVELOPE = 8e9
_NUMERICAL_FLOOR = 1e-7   # see module docstring


def _qubit_model_2lvl():
    return quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)


def _infidelity_at(zmismatch, tprop_s, qmodel, mode="complex_baseband"):
    """tuneup_amplitude()-calibrated infidelity for a given (Zmismatch,
    Tprop) pair -- no nonlinearity anywhere, so the analytic-guess fast
    path handles every point in 2 engine.run() calls."""
    schematic = si_loader.load_schematic(
        SCHEMATIC_PATH, variables={"Zmismatch": zmismatch, "Tprop": tprop_s},
    )
    ref_shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, _FS_ENVELOPE, _CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X", mode=mode,
    )
    return 1.0 - tuned.fidelity.noise_free.F_avg


def _assert_monotonic_within_floor(values, floor=_NUMERICAL_FLOOR):
    """Each value may fall below the previous one by at most `floor` --
    tolerates float64 noise when several consecutive true values are all
    deep in the numerical floor, without masking a genuine, larger
    non-monotonic regression."""
    for prev, cur in zip(values[:-1], values[1:]):
        assert cur >= prev - floor, f"{cur} is not >= {prev} - {floor}"


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
    well past the 100ns pulse duration -- the reflected echo is a
    separated, post-pulse perturbation rather than something overlapping
    and distorting the drive itself, so the cost stays modest in absolute
    terms (~1e-5) -- but it's still several orders of magnitude above the
    short-delay case's numerical floor (~1e-8), confirming delay does
    matter even though it isn't catastrophic here."""
    qmodel = _qubit_model_2lvl()
    infid = _infidelity_at(300.0, 200e-9, qmodel)
    assert 1e-6 < infid < 1e-2


def test_infidelity_grows_monotonically_with_delay_at_fixed_mismatch():
    """At a fixed, genuine mismatch, infidelity should increase
    monotonically as round-trip delay grows from well-within the pulse to
    well-beyond it -- a smooth transition, not a random walk (within the
    numerical floor -- see _assert_monotonic_within_floor)."""
    qmodel = _qubit_model_2lvl()
    tprops_ns = [1, 25, 75, 150, 300]
    infidelities = [_infidelity_at(150.0, t * 1e-9, qmodel) for t in tprops_ns]
    _assert_monotonic_within_floor(infidelities)


def test_infidelity_grows_monotonically_with_mismatch_at_long_delay():
    """At a fixed long delay, infidelity should increase monotonically
    with the reflection coefficient -- a real, graded dependence on
    mismatch severity, not a threshold effect (within the numerical floor
    -- see _assert_monotonic_within_floor)."""
    qmodel = _qubit_model_2lvl()
    z_values = [50.0, 70.0, 100.0, 150.0, 250.0]
    infidelities = [_infidelity_at(z, 200e-9, qmodel) for z in z_values]
    _assert_monotonic_within_floor(infidelities)


def test_infidelity_extremely_sensitive_to_subcarrier_delay_phase():
    """See module docstring and examples/impedance_mismatch_demo.py's Panel
    D: every OTHER test in this file uses a whole-nanosecond Tprop, which
    (since the carrier is exactly 5GHz) always puts the reflected echo's
    carrier phase at the safest possible point (0 mod 2*pi). A Tprop just
    picoseconds off that grid can be catastrophic. Tprop=100ns (whole-ns,
    "safe") vs. Tprop=100.02ns (20ps off, within 1/10 of one 0.2ns carrier
    period) at the same severe mismatch -- confirmed directly (see the demo
    script's Panel D data) that this specific pair spans numerical-floor to
    >50% infidelity."""
    qmodel = _qubit_model_2lvl()
    infid_safe = _infidelity_at(300.0, 100.000e-9, qmodel)
    infid_bad = _infidelity_at(300.0, 100.020e-9, qmodel)
    assert infid_safe < 1e-4
    assert infid_bad > 0.1


if __name__ == "__main__":
    pytest.main([__file__])
