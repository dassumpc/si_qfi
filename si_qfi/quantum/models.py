"""
si_qfi.quantum.models
======================
Qubit Hamiltonian definitions: a common base (QubitBase) plus the concrete
QubitModel (user-supplied H0) and Transmon (analytic anharmonic oscillator)
implementations, and an scqubits bridge.

Every qubit type exposes `as_qubit_model() -> QubitModel` so gate_fidelity()
can resolve any qubit type identically, without knowing about specific
subclasses (see QubitBase docstring).
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------------

class QubitBase(ABC):
    """
    Common interface every qubit model implements: `as_qubit_model()`
    resolves any concrete qubit type (QubitModel, Transmon, future types
    like a Fluxonium or a Cooper-pair-box model) to the one canonical
    QubitModel (H0 + n_levels + drive_op) that build_hamiltonian()/
    gate_fidelity() actually consume -- so those functions never need an
    isinstance() check against a growing list of qubit subclasses; they
    just call qubit.as_qubit_model() unconditionally.
    """

    @abstractmethod
    def as_qubit_model(self) -> "QubitModel":
        ...


# ---------------------------------------------------------------------------
# Qubit model definitions
# ---------------------------------------------------------------------------

@dataclass
class QubitModel(QubitBase):
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

    def as_qubit_model(self) -> "QubitModel":
        """Already a QubitModel -- returns self."""
        return self


@dataclass
class Transmon(QubitBase):
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

        NOTE: this is the LAB-FRAME Hamiltonian (includes the full
        omega_q*num precession term). build_hamiltonian() (see
        hamiltonian.py) assumes H0 is already expressed in the frame
        ROTATING at the drive carrier -- feeding this lab-frame H0 directly
        into gate_fidelity() is WRONG unless the caller has separately
        transformed to that frame. See examples/transmon_leakage_demo.py's
        module docstring ("trap #1") for the full diagnosis and the correct
        rotating-frame construction (H0' = (alpha/2)*a+a+aa at exact
        resonance) used throughout that investigation instead of this
        method's output.
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
        """Convert to generic QubitModel with computed (lab-frame) H0 --
        see build_H0()'s docstring for the lab-frame caveat."""
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
