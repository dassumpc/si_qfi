"""
si_qfi.quantum.fidelity
========================
FidelityResult, gate_fidelity() (the main entry point: runs the QuTiP
propagator/mesolve per realization and computes gate and/or state
fidelity), plus the small helpers it needs (apply_channel, ideal_gate_unitary).

QuTiP API (verified against QuTiP 5.0.4, 2026-07-11 -- installed from PyPI,
not the bleeding-edge git clone, which currently requires Python >=3.11 and
so can't be used on this project's Python 3.9 environment):

  import qutip as qt

  Operators:
    qt.destroy(n)        — lowering operator, Fock space dim n
    qt.num(n)             — number operator
    qt.qeye(n)            — identity
    qt.sigmaz()           — Pauli Z (two-level only -- see gate_fidelity()'s
                             T2_us docstring for why this doesn't generalize
                             to n_levels > 2 as-is)

  Time-dependent Hamiltonian (list format):
    H = [H0, [op_i, coeff_i_array], [op_q, coeff_q_array]]
    Passed directly to qt.propagator(..., tlist=t_array) -- propagator's
    **kwargs (including tlist) are forwarded to the QobjEvo built from this
    list internally; no need to construct QobjEvo explicitly. QuTiP applies
    cubic spline interpolation to the array coefficients by default.

  Propagator (the sole solver entry point used here -- see gate_fidelity()):
    qt.propagator(H, T, c_ops=(), tlist=t_array)
    Confirmed: returns a plain unitary Qobj when c_ops=() (dispatches to
    sesolve internally), or a superoperator (type='super', supermatrix form)
    when c_ops is non-empty (dispatches to mesolve internally).

  Fidelity:
    qt.average_gate_fidelity(oper, target=None)
    Confirmed: oper may be EITHER a unitary or a superoperator (as returned
    by propagator() above, either case); target must be a plain unitary.
    Global phase does not affect the result (confirmed via a resonant
    pi-pulse test: U_actual came back as -i*sigmax-like, still gave F≈1
    against the plain [[0,1],[1,0]] X matrix). NOTE: this phase-invariance
    only holds when the SAME global phase applies across the WHOLE Hilbert
    space -- see gate_fidelity()'s ideal_gate docstring for the n_levels>2
    caveat where it does NOT hold.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Union, TYPE_CHECKING

from .models import QubitBase
from .hamiltonian import build_hamiltonian, demodulate

if TYPE_CHECKING:
    from ..simulation.engine import SimulationResult


# ---------------------------------------------------------------------------
# Gate fidelity result
# ---------------------------------------------------------------------------

@dataclass
class SingleFidelity:
    """
    Fidelity computed from a SINGLE deterministic waveform (result.v_nl_qubit
    -- no stochastic noise). Always populated by gate_fidelity(), regardless
    of whether the SimulationResult it was given had noise enabled --
    querying "what's the fidelity of the noiseless/deterministic drive" is
    meaningful either way and costs one solve either way.

    propagator holds the raw channel (a QuTiP Qobj -- a plain unitary for a
    closed-system run, or a superoperator if T1_us/T2_us were given) that
    gate_fidelity() already had to compute to get F_avg -- kept here at zero
    extra cost so callers can inspect/reuse it (e.g. via final_state())
    without re-solving anything.
    """
    propagator: Any = None                    # single qt.Qobj (unitary or superoperator)
    fidelity: Optional[float] = None           # average GATE fidelity to ideal_gate; None if ideal_gate wasn't given
    F_avg: Optional[float] = None              # alias of `fidelity`, for symmetry with NoiseEnsembleFidelity's naming
    ideal_gate: Optional[str] = None           # gate name (or repr of a custom Qobj target), None if only target_state was requested
    state_F_avg: Optional[float] = None        # state fidelity to target_state; None if target_state wasn't given

    def final_state(self, initial_state=None):
        """Apply the stored propagator/channel to `initial_state` (a QuTiP
        ket or density matrix; defaults to the ground state |0><0|),
        returning the resulting density matrix -- no new QuTiP solve."""
        if self.propagator is None:
            raise ValueError(
                "No propagator stored on this SingleFidelity -- "
                "gate_fidelity() should always populate this; if it's "
                "None something upstream went wrong."
            )
        import qutip as qt
        if initial_state is None:
            n = self.propagator.dims[0][0]
            initial_state = qt.basis(n, 0)
        return apply_channel(self.propagator, initial_state)


@dataclass
class NoiseEnsembleFidelity:
    """
    Fidelity statistics over the stochastic noise ensemble (result.
    v_qubit_ensemble). Only constructed by gate_fidelity() when
    result.noise_enabled is True -- see FidelityResult.noise.
    """
    n_realizations: int
    propagators: list = field(default_factory=list)   # per-realization qt.Qobj -- always populated
    fidelities: Optional[np.ndarray] = None            # shape (n_realizations,) -- only set if ideal_gate was given
    F_avg: Optional[float] = None
    F_std: Optional[float] = None
    F_sem: Optional[float] = None
    ideal_gate: Optional[str] = None
    state_fidelities: Optional[np.ndarray] = None      # shape (n_realizations,) -- only set if target_state was given
    state_F_avg: Optional[float] = None
    state_F_std: Optional[float] = None
    state_F_sem: Optional[float] = None

    def final_states(self, initial_state=None) -> list:
        """Apply each stored propagator/channel to `initial_state`,
        returning one density matrix per realization -- no new QuTiP
        solve."""
        if not self.propagators:
            raise ValueError(
                "No propagators stored on this NoiseEnsembleFidelity -- "
                "gate_fidelity() should always populate this when "
                "result.noise_enabled is True; if it's empty something "
                "upstream went wrong."
            )
        import qutip as qt
        if initial_state is None:
            n = self.propagators[0].dims[0][0]
            initial_state = qt.basis(n, 0)
        return [apply_channel(U, initial_state) for U in self.propagators]


@dataclass
class FidelityResult:
    """
    Result of gate_fidelity(): a single deterministic (noise-free) fidelity,
    always populated, plus stochastic noise-ensemble statistics, populated
    ONLY when the SimulationResult passed in had real noise configured
    (result.noise_enabled) -- `noise` is None otherwise. Before this split,
    every field lived flat on one object and a "noise-free" evaluation was
    represented as an n_realizations-sized ensemble of identical values
    (wastefully re-solved n_realizations times, and indistinguishable in
    the API from a genuine noise ensemble) -- separating them makes both
    the intent and the cost explicit.
    """
    noise_free: SingleFidelity
    noise: Optional[NoiseEnsembleFidelity] = None   # None iff the SimulationResult had noise_enabled=False
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        nf = self.noise_free
        gate_part = (
            f"gate='{nf.ideal_gate}', F_avg={nf.F_avg:.5f}"
            if nf.F_avg is not None else "gate=None"
        )
        state_part = f", state_F_avg={nf.state_F_avg:.5f}" if nf.state_F_avg is not None else ""
        noise_part = (
            f", noise(N={self.noise.n_realizations}, F_avg={self.noise.F_avg:.5f})"
            if self.noise is not None and self.noise.F_avg is not None
            else f", noise(N={self.noise.n_realizations})" if self.noise is not None
            else ""
        )
        return f"FidelityResult(noise_free({gate_part}{state_part}){noise_part})"


def apply_channel(U, state):
    """
    Apply a QuTiP propagator/channel `U` (as stored in
    FidelityResult.propagators -- a plain unitary Qobj for a closed-system
    run, or a superoperator for an open-system/T1,T2 run) to `state` (a ket
    or density matrix), returning the resulting density matrix.

    QuTiP's own calling convention differs between the two cases (confirmed
    directly against QuTiP 5.0.4): a superoperator can be applied directly
    to a density matrix via U(rho), but a plain unitary can only be called
    directly on a ket (U(psi)) -- calling a unitary on a density matrix
    raises TypeError("oper cannot act on oper"); the correct unitary case
    is U*rho*U.dag(). This dispatches on U.type so callers don't need to
    know which case applies.
    """
    import qutip as qt
    if U.type == "super":
        rho = qt.ket2dm(state) if state.type == "ket" else state
        return U(rho)
    # U.type == "oper" (plain unitary)
    psi_or_rho = U(state) if state.type == "ket" else U * state * U.dag()
    return qt.ket2dm(psi_or_rho) if psi_or_rho.type == "ket" else psi_or_rho


# ---------------------------------------------------------------------------
# Ideal gate unitaries
# ---------------------------------------------------------------------------

def ideal_gate_unitary(gate_name: str, n_levels: int):
    """
    Return the ideal gate unitary as a QuTiP Qobj.

    Supported gates: 'X', 'Y', 'Z', 'H', 'X/2', 'Y/2', 'I'.
    All are embedded in the n_levels Hilbert space (acting on the {|0⟩,|1⟩} subspace).
    """
    try:
        import qutip as qt
    except ImportError:
        raise ImportError("QuTiP is required.")

    # 2-level unitaries
    gates_2lvl = {
        "I":   np.eye(2, dtype=complex),
        "X":   np.array([[0, 1], [1, 0]], dtype=complex),
        "Y":   np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z":   np.array([[1, 0], [0, -1]], dtype=complex),
        "H":   np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        "X/2": np.array([[1, -1j], [-1j, 1]], dtype=complex) / np.sqrt(2),
        "Y/2": np.array([[1, -1], [1, 1]], dtype=complex) / np.sqrt(2),
    }
    gate_name_upper = gate_name.upper()
    if gate_name_upper not in gates_2lvl:
        raise ValueError(
            f"Unknown gate '{gate_name}'. Supported: {list(gates_2lvl.keys())}."
        )

    # Embed in n_levels Hilbert space (identity on higher levels)
    U_full = np.eye(n_levels, dtype=complex)
    U_2lvl = gates_2lvl[gate_name_upper]
    U_full[:2, :2] = U_2lvl

    return qt.Qobj(U_full)


# ---------------------------------------------------------------------------
# Main gate fidelity computation
# ---------------------------------------------------------------------------

def gate_fidelity(
    result: "SimulationResult",
    qubit: QubitBase,
    coupling_strength_per_volt: float,
    ideal_gate: Optional[Union[str, Any]] = None,
    target_state: Optional[Any] = None,
    initial_state: Optional[Any] = None,
    T1_us: Optional[float] = None,
    T2_us: Optional[float] = None,
    lpf_cutoff_hz: Optional[float] = None,
) -> FidelityResult:
    """
    Compute average gate fidelity (and/or state fidelity to a target state)
    over an ensemble of noisy qubit waveforms.

    Verified against QuTiP 5.0.4 (2026-07-11): qt.propagator(H, T, c_ops=,
    tlist=) and qt.average_gate_fidelity(oper, target) both confirmed to
    behave exactly as this function assumes -- propagator returns a plain
    unitary Qobj when c_ops=[] (dispatches to sesolve internally) and a
    superoperator when c_ops is non-empty (dispatches to mesolve);
    average_gate_fidelity accepts either against a plain unitary target.
    See tests/test_quantum.py for the calibrated-pulse checks this was
    confirmed against.

    Parameters
    ----------
    result : simulation.engine.SimulationResult
        Output of siq.run() -- supplies v_qubit_ensemble, fs, mode, and
        carrier_freq_hz all at once (the single source of truth for what
        was actually simulated), rather than the caller reassembling a
        time axis / mode / carrier by hand.
    qubit : QubitBase
        Qubit Hamiltonian definition (QubitModel, Transmon, or any future
        type implementing QubitBase.as_qubit_model()).
    coupling_strength_per_volt : float
        η: drive coupling constant in rad/(s·V).
    ideal_gate : str or qt.Qobj, optional
        Gate name string (e.g. 'X', 'Y', 'X/2' -- see ideal_gate_unitary())
        OR a custom target unitary supplied directly as a QuTiP Qobj (or a
        plain ndarray, auto-wrapped) -- e.g. a gate not in the named table,
        or one you've derived elsewhere. If given, the returned
        fidelities/F_avg/F_std/F_sem are the average GATE fidelity to this
        target (qt.average_gate_fidelity -- compares the whole channel,
        independent of any particular input state).

        CAUTION for qubit.n_levels > 2 with a non-trivial H0 (e.g. a real
        Transmon): ideal_gate_unitary() embeds the named gate on {0,1} and
        plain IDENTITY on higher levels. average_gate_fidelity then
        penalizes any relative phase U_actual picks up on those higher
        levels from ordinary free evolution under H0 for the gate duration
        -- physically unobservable for a qubit that starts and ends in the
        {0,1} subspace, but NOT phase-invariant away from it the way a pure
        global phase would be (confirmed directly: a propagator that was
        essentially exact -i*X on {0,1} with ~0 leaked population still
        scored average_gate_fidelity ~0.3-0.7, purely from an unpopulated-
        |2>-level phase mismatch against the identity-embedded target). For
        n_levels > 2, prefer `target_state` (state fidelity from a chosen
        initial_state, e.g. |0> -> target |1>) to measure real leakage --
        it's insensitive to this artifact. See
        examples/transmon_leakage_demo.py's module docstring ("trap #2")
        for the full diagnosis.
    target_state : qt.Qobj, optional
        A target ket or density matrix to compare the ACTUAL evolved state
        against (qt.fidelity -- a state, not a channel, comparison). If
        given, `initial_state` is evolved through the simulated waveform
        and the resulting density matrix's fidelity to `target_state` is
        computed per realization and returned as state_fidelities/
        state_F_avg/state_F_std/state_F_sem. At least one of ideal_gate /
        target_state must be given (raises otherwise) -- both may be given
        together to get both metrics from the same solve.

        NOTE: gate fidelity and state fidelity are genuinely different
        metrics, not two views of the same number -- gate fidelity averages
        the channel's behavior over ALL input states (the full Bloch
        sphere), state fidelity only checks ONE specific input->output
        pair. Confirmed directly: they coincide near a perfect gate (both
        ≈1), but under a coherent rotation-angle error (0.7π instead of π)
        gate F_avg=0.863 vs. state_F_avg=0.891, and under heavy PURE
        dephasing (T2 only, no extra T1) gate F_avg=0.649 vs.
        state_F_avg=0.929 -- a large gap, because pure dephasing barely
        touches a computational-basis population transfer but heavily
        damages the Bloch-sphere-averaged gate metric. See
        test_gate_and_state_fidelity_diverge_under_pure_dephasing in
        tests/test_quantum.py.
    initial_state : qt.Qobj, optional
        Ket or density matrix to evolve for the target_state comparison.
        Defaults to the ground state |0> (qt.basis(qubit.n_levels, 0)).
        Ignored if target_state is not given.
    T1_us, T2_us : float, optional
        Intrinsic qubit T1 and T2 (microseconds). If supplied, propagator
        is computed with Lindblad collapse operators (dispatches to
        mesolve internally); otherwise closed-system (sesolve). T2's
        pure-dephasing collapse operator uses qt.sigmaz(), which is only
        meaningful for a 2-level qubit -- passing T2_us with
        qubit.n_levels != 2 will raise a dimension-mismatch error from
        QuTiP itself when the operator is applied.

        CAUTION: T_gate (the duration decoherence acts over) is derived
        from `len(result.v_qubit_ensemble[0]) / result.fs` -- the FULL
        simulated array length, not "the pulse duration" as the caller
        conceived it. Convolving a drive envelope through a schematic can
        pad the array with extra samples (e.g. from the schematic's own
        impulse response / group delay) well beyond the nominal pulse
        length -- confirmed directly on tests/test_schematic_basic.si,
        which adds a ~99ns tail essentially independent of input pulse
        length. That padding is harmless for closed-system fidelity (no
        drive there, so the propagator segment over it is ~identity), but
        NOT for T1_us/T2_us -- decoherence keeps acting through it, so
        reported infidelity is inflated by an amount that has nothing to
        do with the nominal gate and everything to do with this
        schematic's convolution padding. If comparing decoherence-driven
        infidelity across different pulse durations or schematics, always
        measure and normalize against the TRUE simulated T_gate (as above),
        not the value passed to your envelope generator. See
        examples/t1_t2_decoherence_demo.py's module docstring for the full
        diagnosis and a worked example.

        CAUTION (numerical): if T1_us/T2_us imply a decay time comparable
        to or SHORTER than the drive's own sample spacing dt=1/fs, the
        solver's step size is now capped against that too (see
        solver_options below) -- without this, a propagator solved through
        a decoherence process faster than the integrator's steps can
        produce a measurably non-physical (non-CPTP) superoperator
        (confirmed directly: a real density matrix eigenvalue of -7.2e-9 at
        T1=0.5ns against dt=0.5ns before this fix). This matters because
        qt.average_gate_fidelity() on a superoperator has NO protection
        against this the way qt.fidelity() does (QuTiP's own fidelity()
        explicitly truncates negative eigenvalues "to avoid nan
        propagation" -- average_gate_fidelity's superoperator path computes
        a raw, unclipped trace instead) -- this is the mechanism behind any
        negative F_avg seen with very fast T1_us/T2_us relative to fs.
    lpf_cutoff_hz : float, optional
        Low-pass filter cutoff for demodulation in real_axis mode (removes
        the 2·f_carrier image after mixing down to baseband -- see
        demodulate()). Ignored in complex_baseband mode.

    Returns
    -------
    FidelityResult
        .noise_free is always populated (one solve on result.v_nl_qubit).
        .noise is populated (one solve per realization on
        result.v_qubit_ensemble) iff result.noise_enabled is True, else
        None -- no solves are wasted re-evaluating an identical waveform
        n_realizations times when no real noise was configured.
    """
    try:
        import qutip as qt
    except ImportError:
        raise ImportError("QuTiP is required. pip install qutip")

    if ideal_gate is None and target_state is None:
        raise ValueError(
            "gate_fidelity() needs at least one of ideal_gate (a gate name "
            "or custom target unitary) or target_state (a target ket/"
            "density matrix) to compare against -- neither was given."
        )

    fs = result.fs
    mode = result.mode
    carrier_hz = result.carrier_freq_hz

    qmodel = qubit.as_qubit_model()
    n = qmodel.n_levels

    U_ideal = None
    if ideal_gate is not None:
        if isinstance(ideal_gate, str):
            U_ideal = ideal_gate_unitary(ideal_gate, n)
        else:
            # Custom target unitary supplied directly -- a qt.Qobj already,
            # or a plain ndarray (auto-wrapped).
            U_ideal = ideal_gate if isinstance(ideal_gate, qt.Qobj) else qt.Qobj(ideal_gate)

    if target_state is not None and initial_state is None:
        initial_state = qt.basis(n, 0)

    # Collapse operators for open-system simulation
    c_ops = []
    if T1_us is not None:
        gamma1 = 1.0 / (T1_us * 1e-6)   # rad/s
        a = qt.destroy(n)
        c_ops.append(np.sqrt(gamma1) * a)
    if T2_us is not None:
        # Pure dephasing rate: 1/T_phi = 1/T2 - 1/(2T1)
        gamma_phi = 1.0 / (T2_us * 1e-6)
        if T1_us is not None:
            gamma_phi -= 0.5 / (T1_us * 1e-6)
        gamma_phi = max(gamma_phi, 0.0)
        if gamma_phi > 0:
            c_ops.append(np.sqrt(gamma_phi) * qt.sigmaz())

    dt = 1.0 / fs

    # CRITICAL: cap the ODE integrator's step to the coefficient array's own
    # sample spacing. Confirmed by direct testing (2026-07-11) that QuTiP's
    # default adaptive step-size control can silently step clean OVER an
    # entire drive pulse when H0 gives it no intrinsic timescale (e.g. an
    # exactly-resonant qubit, H0=0) -- against a real (numerically-noisy,
    # FFT/convolution-derived) envelope array this produced U_actual ~
    # identity (near-total-nonsense: F~1/3 to a target X gate) with NO error
    # or warning from QuTiP. A clean synthetic analytic array did NOT
    # reproduce this, so it can't be assumed away -- every simulated
    # waveform this module receives has some numerical noise. Without this,
    # gate_fidelity() would silently return wrong numbers for any
    # non-constant envelope (i.e. essentially all real pulses).
    #
    # ALSO cap against T1_us/T2_us: a decoherence rate faster than the step
    # size can produce a measurably non-physical (non-CPTP) superoperator --
    # confirmed directly (a real density matrix eigenvalue of -7.2e-9 at
    # T1=0.5ns against dt=0.5ns before this was added). See the T1_us/T2_us
    # docstring above for why this specifically corrupts average_gate_
    # fidelity (unlike qt.fidelity(), which QuTiP itself protects against
    # negative eigenvalues).
    max_step_candidates = [dt]
    if T1_us is not None:
        max_step_candidates.append(0.1 * T1_us * 1e-6)
    if T2_us is not None:
        max_step_candidates.append(0.1 * T2_us * 1e-6)
    max_step = min(max_step_candidates)

    def _solve_one(v: np.ndarray):
        """
        Solve for the propagator of a single waveform array `v`, and
        compute gate/state fidelity against it. Time axis is derived
        per-call from len(v) (arrays at different stages -- v_nl_qubit vs.
        a noise-ensemble member -- are not guaranteed to share a length),
        not assumed shared across calls.
        """
        v = np.asarray(v)
        n_samples = len(v)
        t_array = np.arange(n_samples) / fs
        T_gate = t_array[-1]

        # nsteps must be raised alongside max_step -- capping the step size
        # means the integrator may need many more internal steps (e.g.
        # ~4000 for a 100ns pulse at real_axis mode's 40 GSa/s native rate,
        # or far more if T1_us/T2_us further shrinks max_step), which
        # exceeds QuTiP's default nsteps ceiling and raises
        # IntegratorException("Excess work done on this call...") otherwise.
        solver_options = {
            "max_step": max_step,
            "nsteps": max(10_000, int(20 * T_gate / max_step)),
        }

        # Extract I/Q. np.real()/np.imag() return non-contiguous views into
        # the underlying complex128 buffer -- copy to plain contiguous
        # float64 arrays before handing them to QuTiP.
        if mode == "complex_baseband":
            env_i = np.ascontiguousarray(np.real(v))
            env_q = np.ascontiguousarray(np.imag(v))
        elif mode == "real_axis":
            env_i, env_q = demodulate(v, t_array, carrier_hz, lpf_cutoff_hz)
            env_i = np.ascontiguousarray(env_i)
            env_q = np.ascontiguousarray(env_q)
        else:
            raise ValueError(f"Unknown mode '{mode}' on result.")

        H = build_hamiltonian(qmodel, env_i, env_q, t_array, coupling_strength_per_volt)
        U_actual = qt.propagator(H, T_gate, c_ops=c_ops, tlist=t_array, options=solver_options)

        gate_fid = float(qt.average_gate_fidelity(U_actual, U_ideal)) if U_ideal is not None else None
        state_fid = None
        if target_state is not None:
            rho_final = apply_channel(U_actual, initial_state)
            state_fid = float(qt.fidelity(rho_final, target_state))
        return U_actual, gate_fid, state_fid

    ideal_gate_name = ideal_gate if isinstance(ideal_gate, str) else (repr(ideal_gate) if ideal_gate is not None else None)

    # --- noise-free: always one solve, on the deterministic v_nl_qubit ---
    U_nf, gate_fid_nf, state_fid_nf = _solve_one(result.v_nl_qubit)
    noise_free = SingleFidelity(
        propagator=U_nf,
        fidelity=gate_fid_nf,
        F_avg=gate_fid_nf,
        ideal_gate=ideal_gate_name,
        state_F_avg=state_fid_nf,
    )

    # --- noise ensemble: only if the SimulationResult had real noise ---
    noise_ensemble = None
    if getattr(result, "noise_enabled", False):
        propagators, fidelities, state_fidelities = [], [], []
        for v in result.v_qubit_ensemble:
            U_actual, gate_fid, state_fid = _solve_one(v)
            propagators.append(U_actual)
            if gate_fid is not None:
                fidelities.append(gate_fid)
            if state_fid is not None:
                state_fidelities.append(state_fid)

        n_real = len(result.v_qubit_ensemble)
        ensemble_kwargs = dict(n_realizations=n_real, propagators=propagators)
        if U_ideal is not None:
            fidelities = np.array(fidelities)
            ensemble_kwargs.update(
                fidelities=fidelities,
                F_avg=float(np.mean(fidelities)),
                F_std=float(np.std(fidelities)),
                F_sem=float(np.std(fidelities) / np.sqrt(n_real)),
                ideal_gate=ideal_gate_name,
            )
        if target_state is not None:
            state_fidelities = np.array(state_fidelities)
            ensemble_kwargs.update(
                state_fidelities=state_fidelities,
                state_F_avg=float(np.mean(state_fidelities)),
                state_F_std=float(np.std(state_fidelities)),
                state_F_sem=float(np.std(state_fidelities) / np.sqrt(n_real)),
            )
        noise_ensemble = NoiseEnsembleFidelity(**ensemble_kwargs)

    return FidelityResult(noise_free=noise_free, noise=noise_ensemble)


# ---------------------------------------------------------------------------
# Amplitude tune-up: shared calibration helper
# ---------------------------------------------------------------------------

@dataclass
class TuneupResult:
    """Result of tuneup_amplitude(): the amplitude scale found, whether the
    target was actually reached, and the SimulationResult/FidelityResult at
    that scale (so the caller doesn't need to re-run anything to use them)."""
    scale: float
    achieved: bool
    result: Any                # simulation.engine.SimulationResult
    fidelity: FidelityResult


# Rotation angle (about the drive's I axis) that scaling a reference
# amp=1 envelope by the "right" factor should hit, for the gate names this
# codebase's demos actually calibrate to -- used only as an informed
# STARTING GUESS for the search below (exact for any linear chain; a very
# good bracket even under nonlinearity), never as the calibration criterion
# itself (fidelity is -- see tuneup_amplitude()'s docstring).
_ROTATION_TARGET_THETA = {"I": 0.0, "X": np.pi, "Y": np.pi, "X/2": np.pi / 2, "Y/2": np.pi / 2}


def tuneup_amplitude(
    schematic,
    reference_shape: np.ndarray,
    fs_envelope: float,
    carrier_ghz: float,
    qubit: QubitBase,
    coupling_strength_per_volt: float,
    ideal_gate: Optional[Union[str, Any]] = None,
    target_state: Optional[Any] = None,
    initial_state: Optional[Any] = None,
    nonlinear: Optional[dict] = None,
    mode: str = "complex_baseband",
    lpf_cutoff_hz: Optional[float] = None,
    scale_bounds: tuple = (1e-4, 500.0),
    n_scan: int = 41,
) -> TuneupResult:
    """
    Find the real amplitude scale factor for `reference_shape` (an amp=1
    reference envelope, e.g. from build_gaussian_envelope()/
    build_drag_envelope() -- may be complex; a single real scale factor
    preserves the I/Q ratio, which is what a DRAG envelope needs to keep
    canceling leakage after calibration) that MAXIMIZES the requested
    fidelity (gate or state -- exactly one of ideal_gate/target_state,
    mirroring gate_fidelity()) through the given schematic/nonlinear
    config, replacing what was previously ~12 near-duplicated hand-rolled
    calibration helpers across this codebase's examples/tests (a classical-
    pulse-area trapz-rescale for linear chains; a coarse geomspace scan +
    bisection on the rising branch for nonlinear ones, since AM-AM
    compression makes realized pulse area non-monotonic in scale once
    driven hard enough -- see Investigation 2 in INVESTIGATIONS.md).

    The search optimizes the NOISE-FREE fidelity only (gate_fidelity()'s
    `.noise_free`, i.e. result.v_nl_qubit, one solve per trial scale) --
    not a noisy/decohered one, even if the caller eventually wants T1_us/
    T2_us/noise applied. This matches how every demo already calibrated
    before this helper existed (calibrate on the clean/characterization
    pulse, then separately evaluate the real, imperfect fidelity) and
    keeps each trial to one solve; T1_us/T2_us/noise are not accepted here
    -- call gate_fidelity() yourself on the returned `.result` at the
    tuned scale for a final fidelity that includes them.

    Strategy (three stages, each one only runs if the previous one didn't
    already land on an excellent fidelity):
      1. If ideal_gate is a simple named rotation ('X','Y','X/2','Y/2','I')
         -- or target_state/initial_state matches the |0>->|1> pattern
         every target_state usage in this codebase actually uses -- one
         reference run gives an analytic scale guess via the classical
         realized pulse area (exact for any linear chain; a strong
         starting bracket even under nonlinearity). If that guess already
         gives near-unit fidelity, return immediately (this covers most of
         this codebase's existing "simple" calibration call sites in 2
         total engine.run() calls, matching their previous cost exactly).
      2. Otherwise, a coarse `np.geomspace(*scale_bounds, n_scan)` scan of
         FIDELITY directly (not classical pulse area) finds the best
         region -- robust to compression making pulse area non-monotonic,
         since fidelity itself is what's being searched, not a proxy for
         it.
      3. A bounded local refinement (scipy.optimize.minimize_scalar,
         method='bounded') around the best point from (1)/(2) polishes the
         scale.

    Parameters
    ----------
    schematic : SISchematic
    reference_shape : np.ndarray
        Amplitude-1 reference envelope (complex or real; scaled by a single
        real scalar during the search).
    fs_envelope, carrier_ghz : float
        Passed straight through to build the SourceWaveform at each trial.
    qubit : QubitBase
    coupling_strength_per_volt : float
    ideal_gate, target_state, initial_state : see gate_fidelity() -- exactly
        one of ideal_gate/target_state is required.
    nonlinear : dict, optional
        Passed straight through to engine.run() at every trial scale --
        0, 1, or N NL nodes all work identically, no special-casing needed
        (unlike the old two-amp-specific hand-rolled two-stage calibration
        this replaces).
    mode : str
    lpf_cutoff_hz : float, optional
    scale_bounds : (float, float)
        Search range for the amplitude scale factor.
    n_scan : int
        Coarse-scan grid size for stage 2 (only reached if stage 1's
        analytic guess isn't available or isn't already excellent).

    Returns
    -------
    TuneupResult
        `achieved` is True iff the best fidelity found is >= 1 - 1e-2 --
        loose enough to not false-negative on real_axis mode's own
        numerical floor (confirmed directly: a genuinely well-calibrated
        two-stage real_axis case peaked at F_avg=0.9957, matching this
        mode's documented ~1e-3-level floor elsewhere in this codebase,
        not a physical limit -- the search was confirmed converged by
        scanning finely around that peak). This codebase's Investigation 2
        found pure AM-AM compression can instead make a target genuinely
        UNREACHABLE within scale_bounds (a hard achievability cliff, not a
        gradual falloff, landing far below this threshold -- e.g. F_avg in
        the 0.3-0.7 range) -- `achieved=False` signals that case rather
        than silently returning the best-effort (poor) result as if it
        were a success.
    """
    if (ideal_gate is None) == (target_state is None):
        raise ValueError(
            "tuneup_amplitude() needs EXACTLY ONE of ideal_gate or "
            "target_state to optimize against."
        )

    import warnings as _warnings
    from ..simulation import engine as _engine
    from ..source.waveform import source_from_envelope_array

    def _realized_theta(result) -> Optional[float]:
        """Classical realized pulse area (rad, about the I axis) for the
        waveform that actually reached the qubit plane -- the same
        ground-truth quantity the old per-demo calibration helpers used to
        decide achievability, independent of and computed alongside the
        fidelity-based objective below."""
        v = np.asarray(result.v_nl_qubit)
        t = np.arange(len(v)) / result.fs
        if mode == "complex_baseband":
            env_i = np.real(v)
        elif mode == "real_axis":
            env_i, _ = demodulate(v, t, result.carrier_freq_hz, lpf_cutoff_hz)
        else:
            return None
        return float(coupling_strength_per_volt * np.trapz(env_i, t))

    def _run_at(scale: float):
        source = source_from_envelope_array(reference_shape * scale, fs_envelope, carrier_ghz)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            result = _engine.run(
                schematic, source, nonlinear=nonlinear, noise=None,
                n_realizations=1, mode=mode,
            )
        theta = _realized_theta(result)
        try:
            fid = gate_fidelity(
                result, qubit, coupling_strength_per_volt=coupling_strength_per_volt,
                ideal_gate=ideal_gate, target_state=target_state, initial_state=initial_state,
                lpf_cutoff_hz=lpf_cutoff_hz,
            )
        except Exception:
            # A wildly over/under-driven trial scale (routinely hit by the
            # coarse scan below, which spans scale_bounds blindly) can push
            # the QuTiP ODE solver into a genuine convergence failure (e.g.
            # IntegratorException("Excess work...") from a pulse rotating
            # through many multiples of 2*pi within the fixed nsteps
            # budget) -- not a bug, just a bad trial point. Treat it as an
            # arbitrarily poor score rather than letting the whole search
            # crash on one unlucky scale.
            return None, result, None, theta
        score = fid.noise_free.F_avg if ideal_gate is not None else fid.noise_free.state_F_avg
        return score, result, fid, theta

    state = {"scale": None, "score": -np.inf, "result": None, "fidelity": None, "max_theta": -np.inf}

    def _consider(scale: float) -> float:
        scale = float(np.clip(scale, scale_bounds[0], scale_bounds[1]))
        score, result, fid, theta = _run_at(scale)
        # max_theta tracks the achievability ceiling INDEPENDENTLY of which
        # scale gave the best fidelity -- a Saleh/Volterra AM-AM curve's
        # theta(scale) turns over well before its OWN peak necessarily
        # coincides with peak fidelity (fidelity degrades smoothly as the
        # achieved rotation falls short of target_theta, so a near-cliff
        # scale can score deceptively well on fidelity alone -- confirmed
        # directly: a case with max theta ~0.95*target still scored
        # F_avg=0.99 on the naive fidelity-only achieved check, well above
        # its 0.99 threshold, despite never completing the target rotation).
        if theta is not None and theta > state["max_theta"]:
            state["max_theta"] = theta
        if score is not None and score > state["score"]:
            state.update(scale=scale, score=score, result=result, fidelity=fid)
        return score if score is not None else -np.inf

    # --- Stage 1: analytic starting guess for a known rotation target ---
    target_theta = None
    if isinstance(ideal_gate, str) and ideal_gate.upper() in _ROTATION_TARGET_THETA:
        target_theta = _ROTATION_TARGET_THETA[ideal_gate.upper()]
    elif target_state is not None:
        n = qubit.as_qubit_model().n_levels
        import qutip as qt
        basis1 = qt.basis(n, 1)
        basis0 = qt.basis(n, 0)
        is0 = initial_state is None or (hasattr(initial_state, "full") and np.allclose(initial_state.full(), basis0.full()))
        is1 = hasattr(target_state, "full") and np.allclose(target_state.full(), basis1.full())
        if is0 and is1:
            target_theta = np.pi

    if target_theta is not None and target_theta > 0:
        source_ref = source_from_envelope_array(reference_shape, fs_envelope, carrier_ghz)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            result_ref = _engine.run(
                schematic, source_ref, nonlinear=nonlinear, noise=None,
                n_realizations=1, mode=mode,
            )
        v = np.asarray(result_ref.v_nl_qubit)
        t = np.arange(len(v)) / result_ref.fs
        if mode == "complex_baseband":
            env_i = np.real(v)
        else:
            env_i, _ = demodulate(v, t, result_ref.carrier_freq_hz, lpf_cutoff_hz)
        theta_ref = float(coupling_strength_per_volt * np.trapz(env_i, t))
        if abs(theta_ref) > 1e-12:
            initial_scale = target_theta / theta_ref
            if scale_bounds[0] <= initial_scale <= scale_bounds[1]:
                score = _consider(initial_scale)
                if not nonlinear:
                    # Linear chain: this scale is EXACT (classical pulse
                    # area scales exactly proportional to amplitude), not
                    # an approximation to refine -- whatever fidelity
                    # results here (e.g. real dispersion infidelity) IS
                    # the answer, and no further search over scale can
                    # improve it. Returning immediately also avoids the
                    # coarse scan's blind sweep up to scale_bounds[1]
                    # (which can wildly overdrive the qubit -- many
                    # multiples of a full rotation -- for a system with no
                    # compression to tame it, risking exactly the ODE
                    # solver blowup _run_at() now guards against, for no
                    # benefit since this scale was already correct).
                    return TuneupResult(
                        state["scale"], score is not None and score >= 1.0 - 1e-2,
                        state["result"], state["fidelity"],
                    )
                if score is not None and score >= 1.0 - 1e-6:
                    return TuneupResult(state["scale"], True, state["result"], state["fidelity"])

    # --- Stage 2: coarse fidelity scan across the full search range ---
    for scale in np.geomspace(scale_bounds[0], scale_bounds[1], n_scan):
        _consider(scale)

    # --- Stage 3: bounded local refinement around the best point found ---
    from scipy.optimize import minimize_scalar
    lo = max(scale_bounds[0], state["scale"] / 3.0)
    hi = min(scale_bounds[1], state["scale"] * 3.0)
    if hi > lo:
        minimize_scalar(
            lambda s: -_consider(s), bounds=(lo, hi), method="bounded",
            options={"xatol": max(state["scale"] * 1e-4, 1e-12)},
        )

    if target_theta is not None and target_theta > 0:
        # Ground-truth achievability: did ANY trial scale actually complete
        # the target rotation, not just "is fidelity high enough at the
        # best-found scale" -- a fidelity-only check can false-positive
        # right at a Saleh/Volterra achievability cliff (see _consider()'s
        # comment): a scale that gets close to but never reaches
        # target_theta can still score deceptively well on fidelity alone.
        achieved = state["max_theta"] >= target_theta * (1.0 - 1e-3)
    else:
        achieved = state["score"] >= 1.0 - 1e-2
    return TuneupResult(state["scale"], achieved, state["result"], state["fidelity"])
