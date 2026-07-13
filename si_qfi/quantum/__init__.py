"""
si_qfi.quantum
==============
QuTiP simulation backend.

Responsibilities
----------------
1. Qubit Hamiltonian definition (transmon analytic, manual, or scqubits).
2. Build QuTiP QobjEvo H(t) from I/Q waveform arrays via cubic spline coefficients.
3. Run sesolve (closed) or mesolve (open, with T1/T2) per realization.
4. Compute average gate fidelity from the propagator.
5. Demodulate real RF waveform to I/Q for the rotating frame (real-axis mode).

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
    against the plain [[0,1],[1,0]] X matrix).

  scqubits (optional):
    import scqubits as scq
    transmon = scq.Transmon(EJ=20.0, EC=0.2, ng=0.0, ncut=30)
    H0 = transmon.hamiltonian()   # returns QuTiP Qobj
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulation.engine import SimulationResult


# ---------------------------------------------------------------------------
# Qubit model definitions
# ---------------------------------------------------------------------------

@dataclass
class QubitModel:
    """
    User-supplied arbitrary qubit Hamiltonian.

    Parameters
    ----------
    H0 : QuTiP Qobj
        Bare qubit Hamiltonian (time-independent part), in units of rad/s.
    n_levels : int
        Hilbert space dimension (number of levels retained).
    drive_op : QuTiP Qobj, optional
        Drive coupling operator. Defaults to (a + a†) where a = destroy(n_levels).
    """
    H0: Any          # qt.Qobj
    n_levels: int
    drive_op: Any = None   # qt.Qobj or None → defaults to (a + a†)

    def __post_init__(self):
        if self.drive_op is None:
            try:
                import qutip as qt
                a = qt.destroy(self.n_levels)
                self.drive_op = a + a.dag()
            except ImportError:
                pass   # QuTiP not installed; Cursor will wire this up


@dataclass
class Transmon:
    """
    Analytic transmon qubit model (anharmonic oscillator approximation).

    H₀ = ω_q · a†a  -  (α/2) · a†a†aa
    where ω_q = 2π · f_q and α = anharmonicity.

    Parameters
    ----------
    Ej_GHz : float
        Josephson energy in GHz (sets qubit frequency approximately).
    Ec_MHz : float
        Charging energy in MHz (sets anharmonicity α ≈ -Ec).
    ng : float
        Offset charge (dimensionless). 0 for sweet spot.
    n_levels : int
        Truncation: number of energy levels to retain (default 5).
    """
    Ej_GHz: float
    Ec_MHz: float
    ng: float = 0.0
    n_levels: int = 5

    def qubit_freq_ghz(self) -> float:
        """Approximate qubit frequency (GHz) from transmon parameters."""
        return np.sqrt(8 * self.Ej_GHz * self.Ec_MHz / 1e3) - self.Ec_MHz / 1e3

    def anharmonicity_mhz(self) -> float:
        """Anharmonicity α ≈ -Ec (MHz)."""
        return -self.Ec_MHz

    def build_H0(self):
        """
        Build QuTiP Hamiltonian for analytic transmon.

        # --- CURSOR NOTE ---
        # Verify qt.destroy, qt.num exist in installed QuTiP version.
        # H0 should be in rad/s (multiply GHz frequencies by 2π·1e9).
        # -------------------
        """
        try:
            import qutip as qt
        except ImportError:
            raise ImportError("QuTiP is required. Install with: pip install qutip")

        n = self.n_levels
        a = qt.destroy(n)
        adag = a.dag()
        num = qt.num(n)

        omega_q = 2 * np.pi * self.qubit_freq_ghz() * 1e9    # rad/s
        alpha = 2 * np.pi * self.anharmonicity_mhz() * 1e6   # rad/s (negative)

        H0 = omega_q * num + (alpha / 2.0) * adag * adag * a * a
        return H0

    def as_qubit_model(self) -> QubitModel:
        """Convert to generic QubitModel with computed H0."""
        H0 = self.build_H0()
        try:
            import qutip as qt
            a = qt.destroy(self.n_levels)
            drive_op = a + a.dag()
        except ImportError:
            drive_op = None
        return QubitModel(H0=H0, n_levels=self.n_levels, drive_op=drive_op)


def from_scqubits(scq_qubit, n_levels: int = 5) -> QubitModel:
    """
    Build a QubitModel from a scqubits qubit object.

    # --- CURSOR NOTE ---
    # scqubits API (verify against installed version):
    #   transmon = scq.Transmon(EJ=20.0, EC=0.2, ng=0.0, ncut=30)
    #   H0_full = transmon.hamiltonian()   # large sparse Qobj
    #   # Truncate to n_levels:
    #   H0 = transmon.hamiltonian()[:n_levels, :n_levels]  # or use scq's truncation
    # Confirm the truncation API — scqubits may provide a direct method.
    # -------------------
    """
    H0_full = scq_qubit.hamiltonian()
    # Truncate to n_levels
    try:
        import qutip as qt
        H0_data = H0_full.full()[:n_levels, :n_levels]
        H0 = qt.Qobj(H0_data)
    except Exception as e:
        raise RuntimeError(f"scqubits → QubitModel conversion failed: {e}")
    return QubitModel(H0=H0, n_levels=n_levels)


# ---------------------------------------------------------------------------
# Demodulation: real RF → I/Q baseband
# ---------------------------------------------------------------------------

def demodulate(
    v_rf: np.ndarray,
    t: np.ndarray,
    carrier_freq_hz: float,
    lpf_cutoff_hz: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Demodulate a real RF waveform to I/Q baseband components.

    Used by real-axis mode to convert the final v_qubit(t) to the I/Q
    representation needed for the rotating-frame QuTiP Hamiltonian.

    Parameters
    ----------
    v_rf : np.ndarray, float64, shape (N,)
        Real RF waveform.
    t : np.ndarray, float64, shape (N,)
        Time array (seconds).
    carrier_freq_hz : float
        Carrier frequency used for demodulation (should match source carrier).
    lpf_cutoff_hz : float, optional
        Low-pass filter cutoff applied after mixing to remove 2ω₀ image.
        If None, no filtering is applied (user is responsible for ensuring
        the sample rate is low enough post-demodulation or uses the
        complex envelope directly).

    Returns
    -------
    I, Q : np.ndarray, float64, shape (N,) each
        In-phase and quadrature baseband components.
        ũ(t) = I(t) + j·Q(t)  is the complex envelope.
    """
    carrier = np.exp(-1j * 2 * np.pi * carrier_freq_hz * t)
    complex_env = v_rf * carrier * 2.0   # ×2 to correct for single-sideband

    if lpf_cutoff_hz is not None:
        from scipy.signal import butter, filtfilt
        fs = 1.0 / (t[1] - t[0])
        nyq = fs / 2.0
        b, a = butter(N=8, Wn=lpf_cutoff_hz / nyq, btype="low")
        complex_env = filtfilt(b, a, np.real(complex_env)) + \
                      1j * filtfilt(b, a, np.imag(complex_env))

    return np.real(complex_env), np.imag(complex_env)


# ---------------------------------------------------------------------------
# Hamiltonian builder
# ---------------------------------------------------------------------------

def build_hamiltonian(
    qubit_model: QubitModel,
    envelope_i: np.ndarray,
    envelope_q: np.ndarray,
    t_array: np.ndarray,
    coupling_strength_per_volt: float,
):
    """
    Build a QuTiP time-dependent Hamiltonian from I/Q waveform arrays.

    H(t) = H₀  +  η·I(t)·(a+a†)/2  +  η·Q(t)·i(a-a†)/2
    in the rotating frame -- H₀ must already be expressed in that frame
    (e.g. 0 for an exactly-resonant drive, or a small detuning term; RWA
    applied externally by the caller when constructing qubit_model.H0).
    There is currently no lab-frame path -- both complex_baseband and
    real_axis modes funnel into this same rotating-frame Hamiltonian
    (real_axis mode demodulates to I/Q first, see gate_fidelity()).

    The I/Q arrays are passed as numpy array coefficients to QobjEvo.
    QuTiP v5 applies cubic spline interpolation automatically.

    Parameters
    ----------
    qubit_model : QubitModel
        Qubit Hamiltonian and drive operator.
    envelope_i, envelope_q : np.ndarray, float64, shape (N,)
        In-phase and quadrature drive envelopes (volts).
    t_array : np.ndarray, float64, shape (N,)
        Time array corresponding to envelope samples (seconds).
    coupling_strength_per_volt : float
        η = drive coupling in rad/(s·V). Converts voltage to Hamiltonian coefficient.

    Returns
    -------
    H : list   [H0, [op_I, coeff_I], [op_Q, coeff_Q]]
        QuTiP time-dependent Hamiltonian in list format, ready for
        qt.propagator(H, T, c_ops=..., tlist=t_array) (or sesolve/mesolve
        directly). Verified against QuTiP 5.0.4: propagator's **kwargs are
        forwarded to the QobjEvo built from this list format, so tlist=
        controls the cubic-spline grid for envelope_i/envelope_q.
    """
    try:
        import qutip as qt
    except ImportError:
        raise ImportError("QuTiP is required.")

    H0 = qubit_model.H0
    n = qubit_model.n_levels
    a = qt.destroy(n)

    # Drive operators
    # σ_x-like coupling: (a + a†)/2  for I component
    # σ_y-like coupling: i(a† - a)/2 for Q component
    op_i = (a + a.dag()) * 0.5
    op_q = 1j * (a.dag() - a) * 0.5

    eta = float(coupling_strength_per_volt)
    coeff_i = eta * envelope_i   # shape (N,), float64
    coeff_q = eta * envelope_q   # shape (N,), float64

    H = [H0, [op_i, coeff_i], [op_q, coeff_q]]
    return H


# ---------------------------------------------------------------------------
# Gate fidelity result
# ---------------------------------------------------------------------------

@dataclass
class FidelityResult:
    """
    Fidelity computation result over an ensemble of noise realizations.

    propagators holds the raw per-realization channel (a QuTiP Qobj -- a
    plain unitary for a closed-system run, or a superoperator if T1_us/
    T2_us were given) that gate_fidelity() already had to compute to get
    F_avg -- kept here at zero extra cost so callers can inspect/reuse it
    (e.g. via final_states()) without re-solving anything.
    """
    n_realizations: int
    propagators: list = field(default_factory=list)   # per-realization qt.Qobj (unitary or superoperator) -- always populated
    fidelities: Optional[np.ndarray] = None            # shape (n_realizations,) -- average GATE fidelity; only set if ideal_gate was given
    F_avg: Optional[float] = None
    F_std: Optional[float] = None
    F_sem: Optional[float] = None
    ideal_gate: Optional[str] = None                   # gate name (or repr of a custom Qobj target), None if only target_state was requested
    state_fidelities: Optional[np.ndarray] = None       # shape (n_realizations,) -- only set if target_state was given
    state_F_avg: Optional[float] = None
    state_F_std: Optional[float] = None
    state_F_sem: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

    def final_states(self, initial_state=None) -> list:
        """
        Apply each stored propagator/channel to `initial_state` (a QuTiP
        ket or density matrix; defaults to the ground state |0><0|),
        returning one density matrix per realization -- the "raw density
        matrix" a caller wants, computed from the already-solved
        propagators without any new QuTiP solve.
        """
        if not self.propagators:
            raise ValueError(
                "No propagators stored on this FidelityResult -- "
                "gate_fidelity() should always populate this; if it's "
                "empty something upstream went wrong."
            )
        import qutip as qt
        if initial_state is None:
            n = self.propagators[0].dims[0][0]
            initial_state = qt.basis(n, 0)
        return [apply_channel(U, initial_state) for U in self.propagators]

    def plot_fidelity_hist(self, bins: int = 20) -> None:
        """Plot histogram of per-realization GATE fidelities. Raises if
        gate_fidelity() was called without ideal_gate (nothing to plot)."""
        if self.fidelities is None:
            raise ValueError(
                "No gate fidelities on this result -- gate_fidelity() was "
                "called with target_state only (no ideal_gate). Use "
                ".state_fidelities directly instead."
            )
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting.")
            return
        fig, ax = plt.subplots()
        ax.hist(self.fidelities, bins=bins, edgecolor="black")
        ax.axvline(self.F_avg, color="red", linestyle="--", label=f"Mean={self.F_avg:.5f}")
        ax.set_xlabel("Gate Fidelity")
        ax.set_ylabel("Count")
        ax.set_title(f"Gate Fidelity Distribution — {self.ideal_gate} gate")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        gate_part = (
            f"gate='{self.ideal_gate}', F_avg={self.F_avg:.5f}, "
            f"F_std={self.F_std:.5f}, F_sem={self.F_sem:.6f}"
            if self.F_avg is not None else "gate=None"
        )
        state_part = (
            f", state_F_avg={self.state_F_avg:.5f}" if self.state_F_avg is not None else ""
        )
        return f"FidelityResult({gate_part}{state_part}, N={self.n_realizations})"


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

    # --- CURSOR NOTE ---
    # qt.Qobj(matrix) builds a Qobj from a 2D numpy array.
    # Confirm dims parameter is not needed for square matrices.
    # -------------------
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
    qubit: "Transmon | QubitModel",
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
    qubit : Transmon or QubitModel
        Qubit Hamiltonian definition.
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
    lpf_cutoff_hz : float, optional
        Low-pass filter cutoff for demodulation in real_axis mode (removes
        the 2·f_carrier image after mixing down to baseband -- see
        demodulate()). Ignored in complex_baseband mode.

    Returns
    -------
    FidelityResult
        Always has .propagators populated (one qt.Qobj per realization --
        the raw channel, unitary or superoperator -- at zero extra solve
        cost); call .final_states() on the result to get actual density
        matrices from it for any initial state, including after the fact.
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

    v_qubit_ensemble = result.v_qubit_ensemble
    fs = result.fs
    mode = result.mode
    carrier_hz = result.carrier_freq_hz

    # Resolve qubit model
    if isinstance(qubit, Transmon):
        qmodel = qubit.as_qubit_model()
    else:
        qmodel = qubit

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

    # Shared time axis for every realization -- SimulationResult deliberately
    # doesn't store one (engine.py's arrays grow segment-to-segment and only
    # `fs` is assumed meaningful across the whole result), so it's derived
    # here from fs + the (shared, engine-guaranteed) ensemble length.
    n_samples = len(v_qubit_ensemble[0])
    t_array = np.arange(n_samples) / fs
    T_gate = t_array[-1]
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
    # nsteps must be raised alongside max_step -- capping the step size to
    # dt means the integrator may need on the order of n_samples internal
    # steps (e.g. ~4000 for a 100ns pulse at real_axis mode's 40 GSa/s
    # native rate), which exceeds QuTiP's default nsteps ceiling and raises
    # IntegratorException("Excess work done on this call...") otherwise.
    solver_options = {"max_step": dt, "nsteps": max(10_000, 20 * n_samples)}

    propagators = []
    fidelities = [] if U_ideal is not None else None
    state_fidelities = [] if target_state is not None else None
    n_real = len(v_qubit_ensemble)

    for v in v_qubit_ensemble:
        v = np.asarray(v)

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
        propagators.append(U_actual)

        if U_ideal is not None:
            fidelities.append(float(qt.average_gate_fidelity(U_actual, U_ideal)))
        if target_state is not None:
            rho_final = apply_channel(U_actual, initial_state)
            state_fidelities.append(float(qt.fidelity(rho_final, target_state)))

    result_kwargs = dict(n_realizations=n_real, propagators=propagators)

    if fidelities is not None:
        fidelities = np.array(fidelities)
        result_kwargs.update(
            fidelities=fidelities,
            F_avg=float(np.mean(fidelities)),
            F_std=float(np.std(fidelities)),
            F_sem=float(np.std(fidelities) / np.sqrt(n_real)),
            ideal_gate=ideal_gate if isinstance(ideal_gate, str) else repr(ideal_gate),
        )
    if state_fidelities is not None:
        state_fidelities = np.array(state_fidelities)
        result_kwargs.update(
            state_fidelities=state_fidelities,
            state_F_avg=float(np.mean(state_fidelities)),
            state_F_std=float(np.std(state_fidelities)),
            state_F_sem=float(np.std(state_fidelities) / np.sqrt(n_real)),
        )

    return FidelityResult(**result_kwargs)
