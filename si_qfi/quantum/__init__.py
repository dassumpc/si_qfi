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

# --- CURSOR NOTE ---
# This module uses QuTiP v5. Key API points to verify against installed version:
#
#   import qutip as qt
#
#   Operators:
#     qt.destroy(n)        — lowering operator, Fock space dim n
#     qt.create(n)         — raising operator
#     qt.num(n)            — number operator
#     qt.qeye(n)           — identity
#     qt.sigmaz()          — Pauli Z (two-level only)
#
#   Time-dependent Hamiltonian (QobjEvo with array coefficients):
#     H = [H0, [op, coeff_array]]   with tlist=t_array passed to solver
#     OR: H = qt.QobjEvo([H0, [op, coeff_array]], tlist=t_array)
#     The solver uses cubic spline interpolation on coeff_array by default.
#
#   Solvers:
#     qt.sesolve(H, psi0, tlist, ...)       — closed system (Schrödinger)
#     qt.mesolve(H, rho0, tlist, c_ops, ...)— open system (Lindblad master eq)
#
#   Propagator:
#     qt.propagator(H, T, c_ops=[], tlist=tlist)
#     Returns Qobj (unitary for closed, superoperator for open).
#
#   Fidelity:
#     qt.average_gate_fidelity(U_actual, U_ideal)
#     NOTE: in QuTiP v5 this may be qt.average_gate_fidelity or
#     qt.process_fidelity — check current docs and confirm function name.
#
#   scqubits (optional):
#     import scqubits as scq
#     transmon = scq.Transmon(EJ=20.0, EC=0.2, ng=0.0, ncut=30)
#     H0 = transmon.hamiltonian()   # returns QuTiP Qobj
# -------------------
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any


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
    rotating_frame: bool = True,
):
    """
    Build a QuTiP time-dependent Hamiltonian from I/Q waveform arrays.

    H(t) = H₀  +  η·I(t)·(a+a†)/2  +  η·Q(t)·i(a-a†)/2
    in the rotating frame (RWA applied externally by the user in H₀).

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
    rotating_frame : bool
        If True, H₀ is assumed to be in the rotating frame (subtract ω_q·n from H₀
        externally). If False, the full lab-frame H₀ is used with the carrier-frequency
        drive (not recommended; very stiff ODE).

    Returns
    -------
    H : list   [H0, [op_I, coeff_I], [op_Q, coeff_Q]]
        QuTiP time-dependent Hamiltonian in list format, ready for sesolve/mesolve.
        The tlist= argument must be passed as t_array when calling the solver.

    # --- CURSOR NOTE ---
    # In QuTiP v5, the preferred way is:
    #
    #   H = [H0, [drive_op_i, coeff_i_array], [drive_op_q, coeff_q_array]]
    #   result = qt.sesolve(H, psi0, t_array, ...)
    #
    # The solver infers tlist from the positional tlist argument and uses
    # cubic spline to interpolate the coeff arrays.
    #
    # If QobjEvo is needed explicitly:
    #   H_t = qt.QobjEvo([H0, [drive_op_i, coeff_i_array]], tlist=t_array)
    #
    # Confirm the exact calling convention for the installed QuTiP v5 version.
    # -------------------
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
    """
    fidelities: np.ndarray         # shape (n_realizations,)
    F_avg: float
    F_std: float
    F_sem: float
    ideal_gate: str
    n_realizations: int
    warnings: list[str] = field(default_factory=list)

    def plot_fidelity_hist(self, bins: int = 20) -> None:
        """Plot histogram of per-realization fidelities."""
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
        return (
            f"FidelityResult(gate='{self.ideal_gate}', "
            f"F_avg={self.F_avg:.5f}, F_std={self.F_std:.5f}, "
            f"F_sem={self.F_sem:.6f}, N={self.n_realizations})"
        )


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
    v_qubit_ensemble: list[np.ndarray],
    t_array: np.ndarray,
    qubit: "Transmon | QubitModel",
    ideal_gate: str,
    coupling_strength_per_volt: float,
    rotating_frame: bool = True,
    T1_us: Optional[float] = None,
    T2_us: Optional[float] = None,
    lpf_cutoff_hz: Optional[float] = None,
    mode: str = "complex_baseband",
) -> FidelityResult:
    """
    Compute average gate fidelity over an ensemble of noisy qubit waveforms.

    Parameters
    ----------
    v_qubit_ensemble : list of np.ndarray
        One waveform array per realization. Complex (baseband mode) or real (real-axis).
    t_array : np.ndarray, float64, shape (N,)
        Shared time axis for all realizations.
    qubit : Transmon or QubitModel
        Qubit Hamiltonian definition.
    ideal_gate : str
        Gate name string, e.g. 'X', 'Y', 'X/2'.
    coupling_strength_per_volt : float
        η: drive coupling constant in rad/(s·V).
    rotating_frame : bool
        If True, I/Q components drive the rotating-frame Hamiltonian.
    T1_us, T2_us : float, optional
        Intrinsic qubit T1 and T2 (microseconds). If supplied, mesolve is used
        with Lindblad collapse operators; otherwise sesolve (closed system).
    lpf_cutoff_hz : float, optional
        Low-pass filter cutoff for demodulation in real-axis mode.
    mode : str
        'complex_baseband' or 'real_axis'.

    Returns
    -------
    FidelityResult
    """
    try:
        import qutip as qt
    except ImportError:
        raise ImportError("QuTiP is required. pip install qutip")

    # Resolve qubit model
    if isinstance(qubit, Transmon):
        qmodel = qubit.as_qubit_model()
    else:
        qmodel = qubit

    n = qmodel.n_levels
    U_ideal = ideal_gate_unitary(ideal_gate, n)

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

    carrier_hz = None
    if mode == "real_axis":
        # We need carrier freq for demodulation; retrieve from first waveform's metadata
        # --- CURSOR NOTE ---
        # The engine should pass carrier_freq_hz explicitly; for now we raise.
        raise NotImplementedError(
            "Real-axis mode: pass v_qubit_ensemble as complex (pre-demodulated) arrays, "
            "or call demodulate() before gate_fidelity(). "
            "Direct real-RF → gate_fidelity path not yet implemented."
        )

    fidelities = []
    n_real = len(v_qubit_ensemble)

    for v in v_qubit_ensemble:
        v = np.asarray(v)

        # Extract I/Q
        if mode == "complex_baseband":
            env_i = np.real(v)
            env_q = np.imag(v)
        else:
            env_i, env_q = demodulate(v, t_array, carrier_hz, lpf_cutoff_hz)

        H = build_hamiltonian(qmodel, env_i, env_q, t_array, coupling_strength_per_volt)
        T_gate = t_array[-1]

        # --- CURSOR NOTE ---
        # qt.propagator signature in v5:
        #   qt.propagator(H, T, c_ops=[], tlist=tlist, options=...)
        # The tlist here tells the solver about the array coefficient grid.
        # Confirm this is the correct call signature for QuTiP 5.x.
        # -------------------
        U_actual = qt.propagator(H, T_gate, c_ops=c_ops, tlist=t_array)

        # --- CURSOR NOTE ---
        # In QuTiP v5 the fidelity function name may differ:
        #   qt.average_gate_fidelity(oper, target)
        # or
        #   qt.process_fidelity(oper, target)
        # Check which is present in the installed version and what the
        # argument order is (actual, ideal) vs (ideal, actual).
        # -------------------
        F_i = qt.average_gate_fidelity(U_actual, U_ideal)
        fidelities.append(float(F_i))

    fidelities = np.array(fidelities)
    return FidelityResult(
        fidelities=fidelities,
        F_avg=float(np.mean(fidelities)),
        F_std=float(np.std(fidelities)),
        F_sem=float(np.std(fidelities) / np.sqrt(n_real)),
        ideal_gate=ideal_gate,
        n_realizations=n_real,
    )
