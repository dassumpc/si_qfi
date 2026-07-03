"""
SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin
========================================================
Bridges SignalIntegrity drive-chain simulation with QuTiP gate fidelity analysis.

Typical usage
-------------
>>> import si_qfi as siq
>>> schematic = siq.load_schematic("driveline.si")
>>> source    = siq.SourceWaveform(carrier_freq_ghz=5.0, envelope=my_waveform)
>>> result    = siq.run(schematic=schematic, source=source,
...                     nonlinear=nl_nodes, noise=noise_nodes,
...                     n_realizations=200)
>>> qubit     = siq.quantum.Transmon(Ej_GHz=20.0, Ec_MHz=200.0, n_levels=5)
>>> fid       = siq.quantum.gate_fidelity(result, qubit, ideal_gate="X")
>>> print(fid.F_avg)
"""

from .schematic.loader import load_schematic
from .source.waveform import SourceWaveform
from .simulation.engine import run
from . import quantum
from . import sweep
from . import output
from .simulation.engine import compare_modes

__version__ = "0.1.0-dev"
__all__ = [
    "load_schematic",
    "SourceWaveform",
    "run",
    "quantum",
    "sweep",
    "output",
    "compare_modes",
]
