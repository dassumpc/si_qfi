"""
tests/test_quantum_t1_t2.py
============================
Regression tests for examples/t1_t2_decoherence_demo.py's findings:
decoherence-driven gate infidelity is governed by the TRUE simulated gate
time (len(result.v_qubit_ensemble[0]) / result.fs) divided by T1 -- NOT the
nominal pulse duration passed to the envelope generator, which can differ
by a schematic-dependent, roughly duration-independent amount (this
schematic's own convolution padding, ~99ns on tests/test_schematic_basic.si
-- see the demo's module docstring and gate_fidelity()'s own T1_us/T2_us
docstring for the full diagnosis). Also confirms T2 (pure dephasing on top
of T1) can dominate total infidelity even when T1 itself is long.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")
qutip = pytest.importorskip("qutip")

warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_basic.si").resolve()

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6
_FS_ENVELOPE = 2e9


def _qubit_model_2lvl():
    return quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)


def _infidelity_and_true_gate_time(schematic, duration_s, qmodel, T1_us=None, T2_us=None):
    """Calibrates (via tuneup_amplitude(), no T1_us/T2_us -- noise-free
    objective) then evaluates the final fidelity WITH T1_us/T2_us in a
    separate gate_fidelity() call, at the tuned scale."""
    sigma_s = duration_s / 6
    ref_shape = build_gaussian_envelope(duration_s, sigma_s, _FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, _FS_ENVELOPE, _CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X", mode="complex_baseband",
    )
    result = tuned.result
    T_gate_true = (len(result.v_qubit_ensemble[0]) - 1) / result.fs
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA, T1_us=T1_us, T2_us=T2_us,
    )
    return 1.0 - fid.noise_free.F_avg, T_gate_true


@pytest.fixture
def basic_schematic():
    return si_loader.load_schematic(SCHEMATIC_PATH)


# ---------------------------------------------------------------------------
# The padding trap: true simulated gate time != nominal pulse duration, and
# it matters for T1/T2 (unlike the closed-system case, already covered
# elsewhere).
# ---------------------------------------------------------------------------

def test_true_gate_time_exceeds_nominal_pulse_duration(basic_schematic):
    """Convolving through this schematic pads the array -- confirmed
    directly rather than assumed, since this is the whole basis for using
    true gate time (not nominal duration) below."""
    _, T_gate_true = _infidelity_and_true_gate_time(
        basic_schematic, 50e-9, _qubit_model_2lvl(), T1_us=None, T2_us=None,
    )
    assert T_gate_true > 50e-9 + 50e-9   # padding is a large fraction of a short pulse


# ---------------------------------------------------------------------------
# T1-limited (T2=2*T1): infidelity / (true_gate_time/T1) should be nearly
# constant across different T1 values and durations, in the perturbative
# (small ratio) regime -- confirming decoherence-driven infidelity is
# governed by true_gate_time/T1, not nominal duration/T1.
# ---------------------------------------------------------------------------

def test_infidelity_collapses_against_true_gate_time_over_T1(basic_schematic):
    qmodel = _qubit_model_2lvl()
    coefficients = []
    for T1_us, d_ns in [(10.0, 50.0), (40.0, 200.0), (100.0, 1000.0)]:
        infid, T_gate_true = _infidelity_and_true_gate_time(
            basic_schematic, d_ns * 1e-9, qmodel, T1_us=T1_us, T2_us=2 * T1_us,
        )
        ratio = T_gate_true / (T1_us * 1e-6)
        coefficients.append(infid / ratio)
    # All three (T1, duration) combos chosen to land near ratio~0.003-0.01
    # (well within the perturbative regime) -- coefficients should agree
    # tightly (measured ~0.332-0.333).
    assert max(coefficients) / min(coefficients) < 1.05


def test_infidelity_does_not_collapse_against_nominal_duration_over_T1(basic_schematic):
    """The negative control confirming the trap is real: using the NOMINAL
    duration instead of true gate time does NOT collapse to a constant --
    it varies by roughly the fixed-padding-to-nominal-duration ratio,
    largest for the shortest nominal pulse."""
    qmodel = _qubit_model_2lvl()
    coefficients = []
    for T1_us, d_ns in [(10.0, 10.0), (10.0, 4000.0)]:
        infid, _ = _infidelity_and_true_gate_time(
            basic_schematic, d_ns * 1e-9, qmodel, T1_us=T1_us, T2_us=2 * T1_us,
        )
        ratio_nominal = (d_ns * 1e-9) / (T1_us * 1e-6)
        coefficients.append(infid / ratio_nominal)
    # d=10ns vs d=4000ns at the same T1: nominal-duration-normalized
    # coefficients should differ by a large factor (measured ~12x) because
    # the fixed ~99ns padding dominates the 10ns case.
    assert max(coefficients) / min(coefficients) > 5.0


# ---------------------------------------------------------------------------
# T2 (pure dephasing): confirms extra dephasing beyond T1 measurably
# increases infidelity, monotonically, even at fixed long T1.
# ---------------------------------------------------------------------------

def test_infidelity_increases_monotonically_as_T2_shortens(basic_schematic):
    qmodel = _qubit_model_2lvl()
    T1_us = 30.0
    T2_values = [60.0, 15.0, 3.75, 0.9375]   # 60us = 2*T1 (T1-limited) down to heavy dephasing
    infidelities = [
        _infidelity_and_true_gate_time(basic_schematic, 200e-9, qmodel, T1_us=T1_us, T2_us=t2)[0]
        for t2 in T2_values
    ]
    assert infidelities == sorted(infidelities)   # T2 decreasing -> infidelity increasing


def test_T1_limited_case_has_lowest_infidelity_of_any_T2_at_fixed_T1(basic_schematic):
    """T2=2*T1 (zero extra dephasing) is the BEST case at fixed T1 -- any
    T2 < 2*T1 adds pure dephasing on top, never reduces infidelity."""
    qmodel = _qubit_model_2lvl()
    T1_us = 30.0
    infid_T1_limited, _ = _infidelity_and_true_gate_time(basic_schematic, 200e-9, qmodel, T1_us=T1_us, T2_us=2 * T1_us)
    infid_dephased, _ = _infidelity_and_true_gate_time(basic_schematic, 200e-9, qmodel, T1_us=T1_us, T2_us=5.0)
    assert infid_dephased > infid_T1_limited


if __name__ == "__main__":
    pytest.main([__file__])
