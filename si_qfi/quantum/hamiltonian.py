"""
si_qfi.quantum.hamiltonian
===========================
Demodulation (real RF -> I/Q baseband, for real-axis mode) and the QuTiP
time-dependent Hamiltonian builder shared by gate_fidelity() and any direct
solver usage (e.g. examples/rabi_oscillation_demo.py).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .models import QubitModel


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
        Low-pass filter cutoff applied after mixing to remove the 2·f_carrier
        image. If None (default), defaults to `carrier_freq_hz` itself --
        halfway (linear scale) between the near-DC signal band and the
        2·f_carrier image, comfortably clear of both for any reasonably
        narrowband envelope (the whole premise complex-baseband/narrowband
        mode relies on -- see the narrowband-ratio diagnostic in
        simulation/engine.py). Every caller in this codebase that cared
        about the image previously had to hand-pick a cutoff (typically
        ~0.1·f_carrier); this default removes that boilerplate while still
        being overridable for a caller with an unusually wide envelope.

    Returns
    -------
    I, Q : np.ndarray, float64, shape (N,) each
        In-phase and quadrature baseband components.
        ũ(t) = I(t) + j·Q(t)  is the complex envelope.
    """
    if lpf_cutoff_hz is None:
        lpf_cutoff_hz = carrier_freq_hz

    carrier = np.exp(-1j * 2 * np.pi * carrier_freq_hz * t)
    complex_env = v_rf * carrier * 2.0   # ×2 to correct for single-sideband

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
