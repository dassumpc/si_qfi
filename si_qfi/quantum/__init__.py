"""
si_qfi.quantum
==============
QuTiP simulation backend.

Responsibilities
----------------
1. Qubit Hamiltonian definition (transmon analytic, manual, or scqubits) -- models.py
2. Build QuTiP QobjEvo H(t) from I/Q waveform arrays via cubic spline
   coefficients, and demodulate real RF to I/Q for real-axis mode -- hamiltonian.py
3. Run sesolve (closed) or mesolve (open, with T1/T2), compute gate and/or
   state fidelity -- fidelity.py

This file only re-exports the public API from the submodules above; see
their module docstrings (in particular fidelity.py's, which has the QuTiP
API notes) for implementation detail.
"""

from __future__ import annotations

from .models import QubitBase, QubitModel, Transmon, from_scqubits
from .hamiltonian import build_hamiltonian, demodulate
from .fidelity import (
    FidelityResult, SingleFidelity, NoiseEnsembleFidelity,
    gate_fidelity, apply_channel, ideal_gate_unitary,
    TuneupResult, tuneup_amplitude,
)

__all__ = [
    "QubitBase", "QubitModel", "Transmon", "from_scqubits",
    "build_hamiltonian", "demodulate",
    "FidelityResult", "SingleFidelity", "NoiseEnsembleFidelity",
    "gate_fidelity", "apply_channel", "ideal_gate_unitary",
    "TuneupResult", "tuneup_amplitude",
]
