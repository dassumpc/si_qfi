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
>>> fid       = siq.quantum.gate_fidelity(result, qubit, coupling_strength_per_volt=2e7,
...                                       ideal_gate="X")
>>> print(fid.noise_free.F_avg)
>>> fid.noise_free.propagator      # the deterministic channel (unitary, or superoperator if T1_us/T2_us given)
>>> fid.noise_free.final_state()   # apply it to |0> (or any state you pass in) -> a density matrix
>>> # fid.noise is a NoiseEnsembleFidelity (with .F_avg/.F_std/.propagators/
>>> # .final_states(), plural) instead of None only if `noise_nodes` above
>>> # was non-empty -- see quantum/fidelity.py's FidelityResult docstring.

See also: `run(..., phase_noise={...})` for LO/oscillator phase noise (a
separate, multiplicative mechanism from `noise=`'s additive drive-line
noise -- see noise/psd.py's phase_noise_psd_from_spec()), and
`quantum.pulse_snr(result)` for the effective SNR of a noisy result. See
README.md (at the repo root) / INVESTIGATIONS.md (in this package
directory) for the full feature list and worked examples.
"""

from .schematic.loader import load_schematic
from .source.waveform import SourceWaveform
from .simulation.engine import run
from . import quantum
from . import output
from .simulation.engine import compare_modes

__version__ = "0.1.0.dev0"   # keep in sync with pyproject.toml's [project] version
__all__ = [
    "load_schematic",
    "SourceWaveform",
    "run",
    "quantum",
    "output",
    "compare_modes",
]
