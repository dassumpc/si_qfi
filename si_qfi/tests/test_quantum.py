"""
tests/test_quantum.py
======================
Correctness tests for si_qfi.quantum -- the QuTiP bridge -- against a plain
2-level qubit (no Transmon/scqubits complexity, no nonlinearity, no noise).
Requires QuTiP; skipped entirely if it isn't installed.

Verified once (2026-07-11, QuTiP 5.0.4) and relied on throughout this file:
  - qt.propagator(H, T, c_ops=, tlist=) returns a plain unitary when
    c_ops=() (closed system) and a superoperator when c_ops is non-empty
    (open system, Lindblad) -- see quantum/__init__.py's module docstring.
  - qt.average_gate_fidelity(oper, target) accepts either against a plain
    unitary target, and is insensitive to oper's global phase.

These pi-pulse tests are also what originally caught a real bug in
gate_fidelity(): without capping the ODE integrator's max_step to the
coefficient array's own sample spacing, QuTiP's default adaptive step size
silently stepped clean OVER an entire Gaussian drive pulse (H0=0 gives it
no intrinsic timescale to size steps against), returning U_actual ~
identity -- i.e. F~1/3 to an X target -- with no error or warning. A clean
analytic array didn't reproduce it; only the real, numerically-noisy
FFT/convolution-derived waveform did. See the "CRITICAL" comment at
gate_fidelity()'s solver_options in quantum/__init__.py. If these tests
ever start failing with F close to 1/3 (not close to 1), that fix likely
regressed.

Calibration strategy (all "pi-pulse" tests below): rather than hand-deriving
the drive amplitude needed to hit a pulse area of pi (which would require
knowing this schematic's exact linear gain and baking that constant into
the test), each test self-calibrates in two engine.run() passes:
  1. Run once with a REFERENCE (arbitrary, amplitude=1) envelope shape.
  2. Numerically integrate the ACTUAL I(t) that reached the qubit plane
     (eta * integral(I(t) dt) = realized pulse area) to get theta_ref.
  3. Rescale the envelope by pi/theta_ref (the system is linear -- no NL
     nodes here -- so this rescaling is exact, not approximate) and run
     once more.
This works identically for complex_baseband and real_axis modes, and
doesn't hardcode test_schematic_basic.si's own gain (2.5, see
test_schematic_hookup.py) anywhere -- so it stays correct even if the
fixture schematic's gain ever changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")
qutip = pytest.importorskip("qutip")

from si_qfi.schematic import loader as si_loader
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform, build_gaussian_envelope
from si_qfi import quantum

BASIC_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_basic.si").resolve()

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6   # rad/(s.V) -- ~10 MHz/V coupling, arbitrary but fixed
_DURATION_S = 100e-9      # 100 ns gate
_SIGMA_S = _DURATION_S / 6
_FS_ENVELOPE = 2e9        # 2 GSa/s baseband envelope grid (200 samples/pulse)
_LPF_CUTOFF_HZ = 500e6    # real-axis demod low-pass: >> envelope BW (~10MHz),
                          # << 2*carrier image (10 GHz)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_schematic():
    return si_loader.load_schematic(BASIC_SCHEMATIC_PATH)


def _qubit_model_2lvl():
    """
    Plain resonant 2-level qubit: H0 = 0 in the rotating frame (exactly
    on-resonance with the carrier -- no detuning term). drive_op is left
    at QubitModel's own default (destroy(2)+destroy(2).dag(), i.e. sigmax),
    which happens to be exactly what build_hamiltonian() itself hardcodes
    internally regardless of drive_op (see quantum/__init__.py) -- so this
    default is the only choice that's actually honored end-to-end today.
    """
    H0 = 0 * qutip.qeye(2)
    return quantum.QubitModel(H0=H0, n_levels=2)


def _source_from_shape(shape: np.ndarray, fs: float, carrier_ghz: float) -> SourceWaveform:
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def _realized_theta(result, eta: float, lpf_cutoff_hz=None) -> float:
    """
    Numerically integrate eta*I(t) over the ACTUAL waveform that reached
    the qubit plane (result.v_nl_qubit) to get the realized pulse area
    (rotation angle, rad) about the I axis -- the same quantity
    quantum.build_hamiltonian()'s op_i integrates against.
    """
    v = np.asarray(result.v_nl_qubit)
    t = np.arange(len(v)) / result.fs
    if result.mode == "complex_baseband":
        env_i = np.real(v)
    else:
        env_i, _ = quantum.demodulate(v, t, result.carrier_freq_hz, lpf_cutoff_hz)
    return float(eta * np.trapz(env_i, t))


def _run_calibrated_pi_pulse(basic_schematic, mode: str, target_theta: float = np.pi):
    """
    Two-pass self-calibration (see module docstring) to a drive of exactly
    `target_theta` radians of rotation about the I axis, then run once more
    with the calibrated amplitude and return (SimulationResult, theta_ref)
    for the caller to feed into quantum.gate_fidelity().
    """
    lpf = _LPF_CUTOFF_HZ if mode == "real_axis" else None

    ref_shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    result_ref = engine.run(
        basic_schematic, source_ref, nonlinear=None, noise=None,
        n_realizations=1, mode=mode,
    )
    theta_ref = _realized_theta(result_ref, _ETA, lpf)
    assert abs(theta_ref) > 1e-12, "reference pulse produced ~zero rotation -- eta or gain is zero?"

    scale = target_theta / theta_ref
    calibrated_shape = ref_shape * scale
    source_cal = _source_from_shape(calibrated_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    result_cal = engine.run(
        basic_schematic, source_cal, nonlinear=None, noise=None,
        n_realizations=1, mode=mode,
    )
    return result_cal


# ---------------------------------------------------------------------------
# Direct QuTiP-path sanity check (no SI schematic at all) -- confirms
# quantum.build_hamiltonian()/qt.propagator()/qt.average_gate_fidelity()
# reproduce the textbook resonant-Rabi pi-pulse result exactly, independent
# of anything schematic-related.
# ---------------------------------------------------------------------------

def test_resonant_pi_pulse_direct_unity_fidelity():
    """
    H0=0 (resonant), constant I envelope calibrated so that
    eta*I_amp*T = pi -- the textbook square-pulse pi-pulse condition for
    H_drive = eta*I(t)*(a+a-dagger)/2. Should give U_actual ~ -i*X (up to
    global phase) and F~1 to the ideal X gate, limited only by numerical
    (cubic-spline/ODE) accuracy -- not by any physical imperfection, since
    there is none here.
    """
    qmodel = _qubit_model_2lvl()
    T = 100e-9
    fs = 1e9
    t = np.arange(0, T, 1.0 / fs)
    eta = _ETA
    I_amp = np.pi / (eta * T)
    env_i = np.full_like(t, I_amp)
    env_q = np.zeros_like(t)

    H = quantum.build_hamiltonian(qmodel, env_i, env_q, t, eta)
    U_actual = qutip.propagator(H, T, c_ops=[], tlist=t)
    U_ideal = quantum.ideal_gate_unitary("X", 2)
    F = qutip.average_gate_fidelity(U_actual, U_ideal)
    assert F == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Full pipeline: engine.run() -> quantum.gate_fidelity(), both modes.
# ---------------------------------------------------------------------------

def test_gate_fidelity_baseband_pi_pulse_near_unity(basic_schematic):
    """
    Complex-baseband mode, full engine.run() -> quantum.gate_fidelity()
    pipeline, self-calibrated to a pi-pulse (see module docstring). No NL,
    no noise -- this is the "no impairment" case, so fidelity should be
    arbitrarily close to 1, limited only by numerical accuracy.
    """
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
    )
    assert fid.F_avg == pytest.approx(1.0, abs=1e-6)
    assert fid.n_realizations == 1


def test_gate_fidelity_realaxis_pi_pulse_near_unity(basic_schematic):
    """
    Real-axis mode, same self-calibrated pi-pulse, now through
    apply_real_axis()'s full RF propagation + demodulate()'s LPF. Looser
    tolerance than the baseband case -- the LPF (needed to remove the
    2*f_carrier image, see demodulate()) is a real (if small) additional
    numerical step baseband mode doesn't need, and both the calibration
    pass and the final pass go through it independently.
    """
    result = _run_calibrated_pi_pulse(basic_schematic, "real_axis")
    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert fid.F_avg == pytest.approx(1.0, abs=1e-3)


def test_gate_fidelity_baseband_and_realaxis_agree(basic_schematic):
    """
    Cross-mode check (PRD's compare_modes() intent, applied at the fidelity
    level): this schematic is flat-gain and lossless (pure delay, no
    dispersion -- see test_schematic_hookup.py), so baseband and real-axis
    should agree closely even away from the exact calibration point.
    Checked at a quarter-turn (theta=pi/2) rather than the pi-pulse itself,
    so this isn't just re-confirming "both hit ~1" for two different
    reasons.
    """
    result_bb = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband", target_theta=np.pi / 2)
    result_ra = _run_calibrated_pi_pulse(basic_schematic, "real_axis", target_theta=np.pi / 2)
    qmodel = _qubit_model_2lvl()

    fid_bb = quantum.gate_fidelity(
        result_bb, qmodel, ideal_gate="X/2", coupling_strength_per_volt=_ETA,
    )
    fid_ra = quantum.gate_fidelity(
        result_ra, qmodel, ideal_gate="X/2", coupling_strength_per_volt=_ETA,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert fid_bb.F_avg == pytest.approx(1.0, abs=1e-6)
    assert fid_ra.F_avg == pytest.approx(1.0, abs=1e-3)
    assert fid_ra.F_avg == pytest.approx(fid_bb.F_avg, abs=1e-3)


def test_gate_fidelity_baseband_half_pi_pulse_near_unity(basic_schematic):
    """
    Companion to test_gate_fidelity_baseband_pi_pulse_near_unity above, for
    the 'X/2' gate (a quarter-turn, theta=pi/2) rather than 'X' (theta=pi)
    -- dedicated/explicitly-named so X/2 has its own direct pass/fail
    signal, not just folded into the cross-mode-agreement check above.
    """
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband", target_theta=np.pi / 2)
    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X/2", coupling_strength_per_volt=_ETA,
    )
    assert fid.F_avg == pytest.approx(1.0, abs=1e-6)
    assert fid.ideal_gate == "X/2"


def test_gate_fidelity_realaxis_half_pi_pulse_near_unity(basic_schematic):
    """Real-axis companion to the baseband X/2 test above."""
    result = _run_calibrated_pi_pulse(basic_schematic, "real_axis", target_theta=np.pi / 2)
    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X/2", coupling_strength_per_volt=_ETA,
        lpf_cutoff_hz=_LPF_CUTOFF_HZ,
    )
    assert fid.F_avg == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# T1/T2 (Lindblad) sanity check -- same calibrated pi-pulse, now with
# intrinsic decoherence. Confirms the open-system (superoperator) branch of
# gate_fidelity() actually reduces fidelity by a physically sensible amount,
# not just "runs without crashing".
# ---------------------------------------------------------------------------

def test_gate_fidelity_with_T1_reduces_fidelity_sensibly(basic_schematic):
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()

    fid_closed = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
    )
    T1_us = 50.0
    fid_open = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
        T1_us=T1_us,
    )
    assert fid_open.F_avg < fid_closed.F_avg
    # T_gate/T1 = 100ns/50us = 0.002 -- a small, bounded infidelity is
    # expected (order 1e-3), not a collapse to ~0.5 (maximally mixed) or a
    # value outside [0, 1].
    assert 0.0 <= fid_open.F_avg <= 1.0
    assert fid_open.F_avg == pytest.approx(1.0, abs=5e-3)
    assert fid_open.F_avg == pytest.approx(0.99868, abs=2e-4)


# ---------------------------------------------------------------------------
# Raw propagator / state-fidelity API (2026-07-11): FidelityResult.propagators
# is populated at zero extra solve cost (the channel gate_fidelity() already
# has to compute to get F_avg), and gate_fidelity() can now compare against
# a target_state (a ket or density matrix -- state fidelity, qt.fidelity())
# instead of, or alongside, ideal_gate (a gate name/matrix -- channel
# fidelity, qt.average_gate_fidelity()).
# ---------------------------------------------------------------------------

def test_gate_fidelity_populates_propagators(basic_schematic):
    """propagators is always populated, one qt.Qobj per realization, and is
    a plain unitary ('oper') for a closed-system (no T1/T2) run."""
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
    )
    assert len(fid.propagators) == fid.n_realizations == 1
    U = fid.propagators[0]
    assert U.type == "oper"
    # Not exactly qt.isunitary (that check's tolerance is tighter than the
    # ~1e-7 numerical floor already characterized for this pulse/sample
    # rate) -- confirm unitarity directly instead: U*U^dagger == I.
    identity_check = (U * U.dag() - qutip.qeye(2)).full()
    assert np.max(np.abs(identity_check)) < 1e-5


def test_gate_fidelity_propagator_is_superoperator_with_T1(basic_schematic):
    """Same, but with T1_us given -- propagators should hold a superoperator
    (open-system channel) instead of a plain unitary."""
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
        T1_us=50.0,
    )
    assert fid.propagators[0].type == "super"


def test_final_states_from_propagators_matches_calibrated_pi_pulse(basic_schematic):
    """
    FidelityResult.final_states() applies the stored propagator to the
    default initial state (|0>) and should reproduce the same physics as
    the calibrated pi-pulse: starting in |0>, a pi-pulse about the I axis
    should leave the qubit in |1> (population ~1, off-diagonal ~0) --
    checked here via the density matrix itself, not just the gate-fidelity
    number, since that's the whole point of exposing it.
    """
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
    )
    rho_final = fid.final_states()[0]
    assert rho_final.type == "oper"
    assert rho_final.tr() == pytest.approx(1.0, abs=1e-4)          # valid density matrix
    assert rho_final.full()[1, 1].real == pytest.approx(1.0, abs=1e-4)   # population in |1>
    assert rho_final.full()[0, 0].real == pytest.approx(0.0, abs=1e-4)   # none left in |0>


def test_gate_fidelity_target_state_only(basic_schematic):
    """
    target_state (no ideal_gate) computes STATE fidelity instead of GATE
    fidelity: a calibrated pi-pulse starting from the default initial state
    |0> should land very close to |1>, so state_fidelities/state_F_avg
    should be near 1 against target_state=|1>. fidelities/F_avg/ideal_gate
    should be left unset (None) since no gate comparison was requested.
    """
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    target = qutip.basis(2, 1)   # |1>

    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=_ETA, target_state=target,
    )
    assert fid.fidelities is None
    assert fid.F_avg is None
    assert fid.ideal_gate is None
    assert fid.state_F_avg == pytest.approx(1.0, abs=1e-4)
    assert len(fid.propagators) == 1   # still populated regardless


def test_gate_fidelity_ideal_gate_and_target_state_together(basic_schematic):
    """Both may be requested together from the same solve -- no propagator
    is computed twice."""
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    target = qutip.basis(2, 1)

    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=_ETA,
        ideal_gate="X", target_state=target,
    )
    assert fid.F_avg == pytest.approx(1.0, abs=1e-6)
    assert fid.state_F_avg == pytest.approx(1.0, abs=1e-4)


def test_gate_fidelity_accepts_custom_qobj_ideal_gate(basic_schematic):
    """ideal_gate may be a custom unitary Qobj directly, not just a named
    gate string -- here the plain X matrix, which should agree exactly
    with ideal_gate='X'."""
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    X_matrix = qutip.Qobj(np.array([[0, 1], [1, 0]], dtype=complex))

    fid_named = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=_ETA,
    )
    fid_custom = quantum.gate_fidelity(
        result, qmodel, ideal_gate=X_matrix, coupling_strength_per_volt=_ETA,
    )
    assert fid_custom.F_avg == pytest.approx(fid_named.F_avg, abs=1e-9)


def test_gate_fidelity_requires_ideal_gate_or_target_state(basic_schematic):
    result = _run_calibrated_pi_pulse(basic_schematic, "complex_baseband")
    qmodel = _qubit_model_2lvl()
    with pytest.raises(ValueError, match="at least one of ideal_gate"):
        quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=_ETA)


if __name__ == "__main__":
    pytest.main([__file__])
