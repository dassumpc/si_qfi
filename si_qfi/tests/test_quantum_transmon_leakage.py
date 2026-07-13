"""
tests/test_quantum_transmon_leakage.py
=======================================
Regression tests for examples/transmon_leakage_demo.py's findings:
a real (anharmonic) transmon shows genuine leakage-driven state infidelity
that grows sharply as the pulse gets fast relative to 1/|anharmonicity|,
while the idealized 2-level qubit used everywhere else in this codebase
cannot leak by construction. DRAG suppresses the actual leakage POPULATION
dramatically, but only modestly improves overall state infidelity at very
short durations (this codebase's build_drag_envelope() is the leading-order
I/Q correction only, no companion detuning term -- see the demo's module
docstring for the full explanation).

Also exercises two previously-untested-in-anger features of
quantum.gate_fidelity(): target_state (state fidelity) and
FidelityResult.final_states() -- used here instead of ideal_gate="X"
average GATE fidelity specifically because that metric is contaminated by
an unobservable relative phase on the unpopulated |2> level for any
qubit model with n_levels > 2 and non-trivial H0 (see the demo's module
docstring, "trap #2", for the full diagnosis).
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
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform, build_gaussian_envelope, build_drag_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_basic.si").resolve()

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6
_FS_ENVELOPE = 20e9
_ANHARMONICITY_MHZ = -200.0


def _transmon_rotating_frame_qubit_model(anharmonicity_hz, n_levels):
    a = qutip.destroy(n_levels)
    alpha = 2 * np.pi * anharmonicity_hz
    H0 = (alpha / 2.0) * a.dag() * a.dag() * a * a
    return quantum.QubitModel(H0=H0, n_levels=n_levels)


def _qubit_model_2lvl():
    return quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)


def _source_from_shape(shape, fs, carrier_ghz):
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def _run_leakage_case(schematic, duration_s, qmodel, use_drag):
    sigma_s = duration_s / 6
    if use_drag:
        ref_shape = build_drag_envelope(duration_s, sigma_s, _ANHARMONICITY_MHZ * 1e6, _FS_ENVELOPE, pi_amp=1.0)
    else:
        ref_shape = build_gaussian_envelope(duration_s, sigma_s, _FS_ENVELOPE, amp=1.0)

    source_ref = _source_from_shape(ref_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    result_ref = engine.run(schematic, source_ref, nonlinear=None, noise=None, n_realizations=1, mode="complex_baseband")
    v = np.asarray(result_ref.v_nl_qubit)
    t = np.arange(len(v)) / result_ref.fs
    theta_ref = float(_ETA * np.trapz(np.real(v), t))
    cal_shape = ref_shape * (np.pi / theta_ref)

    source = _source_from_shape(cal_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    result = engine.run(schematic, source, nonlinear=None, noise=None, n_realizations=1, mode="complex_baseband")

    n = qmodel.n_levels
    target = qutip.basis(n, 1)
    initial = qutip.basis(n, 0)
    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=_ETA,
        target_state=target, initial_state=initial,
    )
    state_infidelity = 1.0 - fid.state_F_avg
    rho_final = fid.final_states(initial_state=initial)[0]
    populations = np.real(rho_final.diag())
    leakage_population = float(np.sum(populations[2:])) if n > 2 else 0.0
    return state_infidelity, leakage_population


@pytest.fixture
def basic_schematic():
    return si_loader.load_schematic(SCHEMATIC_PATH)


# ---------------------------------------------------------------------------
# 2-level idealized model: no leakage channel exists, infidelity should stay
# at the numerical floor regardless of pulse duration.
# ---------------------------------------------------------------------------

def test_two_level_model_never_leaks_regardless_of_duration(basic_schematic):
    qmodel = _qubit_model_2lvl()
    infid_short, _ = _run_leakage_case(basic_schematic, 5e-9, qmodel, use_drag=False)
    infid_long, _ = _run_leakage_case(basic_schematic, 320e-9, qmodel, use_drag=False)
    assert infid_short < 1e-8
    assert infid_long < 1e-8


# ---------------------------------------------------------------------------
# 3-level transmon: genuine leakage-driven infidelity, growing sharply at
# short duration and vanishing at long (adiabatic) duration.
# ---------------------------------------------------------------------------

def test_three_level_transmon_shows_significant_leakage_at_short_duration(basic_schematic):
    qmodel = _transmon_rotating_frame_qubit_model(_ANHARMONICITY_MHZ * 1e6, n_levels=3)
    infid, leak_pop = _run_leakage_case(basic_schematic, 5e-9, qmodel, use_drag=False)
    assert infid > 0.1
    assert leak_pop > 0.1   # a real, large fraction of population reached |2>


def test_three_level_transmon_leakage_vanishes_at_long_duration(basic_schematic):
    qmodel = _transmon_rotating_frame_qubit_model(_ANHARMONICITY_MHZ * 1e6, n_levels=3)
    infid, leak_pop = _run_leakage_case(basic_schematic, 320e-9, qmodel, use_drag=False)
    assert infid < 1e-3
    assert leak_pop < 1e-8


def test_three_level_transmon_infidelity_decreases_monotonically_with_duration(basic_schematic):
    qmodel = _transmon_rotating_frame_qubit_model(_ANHARMONICITY_MHZ * 1e6, n_levels=3)
    durations_ns = [5, 20, 80, 320]
    infidelities = [
        _run_leakage_case(basic_schematic, d * 1e-9, qmodel, use_drag=False)[0]
        for d in durations_ns
    ]
    assert infidelities == sorted(infidelities, reverse=True)


# ---------------------------------------------------------------------------
# DRAG: dramatically suppresses raw leakage population; only modestly
# improves overall state infidelity at very short duration (uncorrected
# AC-Stark rotation-angle error -- see module docstring).
# ---------------------------------------------------------------------------

def test_drag_suppresses_leakage_population_at_short_duration(basic_schematic):
    qmodel = _transmon_rotating_frame_qubit_model(_ANHARMONICITY_MHZ * 1e6, n_levels=3)
    _, leak_gauss = _run_leakage_case(basic_schematic, 10e-9, qmodel, use_drag=False)
    _, leak_drag = _run_leakage_case(basic_schematic, 10e-9, qmodel, use_drag=True)
    assert leak_drag < leak_gauss / 10.0   # at least 10x suppression (measured ~64x)


def test_drag_leakage_suppression_far_exceeds_state_infidelity_improvement(basic_schematic):
    """The nuance this investigation actually found: DRAG's leakage-population
    win is much bigger than its state-infidelity win at short duration,
    because the simple I/Q-only DRAG correction used here doesn't also
    correct the AC-Stark-shift rotation-angle error."""
    qmodel = _transmon_rotating_frame_qubit_model(_ANHARMONICITY_MHZ * 1e6, n_levels=3)
    infid_gauss, leak_gauss = _run_leakage_case(basic_schematic, 10e-9, qmodel, use_drag=False)
    infid_drag, leak_drag = _run_leakage_case(basic_schematic, 10e-9, qmodel, use_drag=True)

    leak_suppression = leak_gauss / leak_drag
    infid_suppression = infid_gauss / infid_drag
    assert leak_suppression > 10 * infid_suppression


if __name__ == "__main__":
    pytest.main([__file__])
