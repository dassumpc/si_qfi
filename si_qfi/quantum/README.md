# si_qfi.quantum

The QuTiP simulation backend: turns a `SimulationResult` into a qubit
Hamiltonian and computes gate/state fidelity, plus calibration and SNR
utilities built on top of it.

- **`models.py`** — `QubitBase` (common interface, `as_qubit_model()`),
  `QubitModel` (arbitrary user-supplied H0), `Transmon` (analytic
  anharmonic-oscillator model), `from_scqubits()` (not yet exercised
  against a real scqubits install — see its own docstring).
- **`hamiltonian.py`** — `build_hamiltonian()` (I/Q envelope arrays -> a
  QuTiP time-dependent `H(t)`, rotating frame), `demodulate()` (real RF ->
  I/Q baseband, used by real-axis mode and by `compare_modes()`).
- **`fidelity.py`** — `gate_fidelity()` (average gate fidelity to a target
  unitary and/or state fidelity, over a noise ensemble; returns
  `.noise_free`/`.noise` separately — see `FidelityResult`'s own
  docstring), `tuneup_amplitude()` (searches a pulse's amplitude scale to
  hit a target fidelity through the actual, possibly nonlinear, chain).
- **`snr.py`** — `pulse_snr()`: effective SNR (signal power / noise power,
  windowed to where the signal is actually significant, bandwidth-matched
  across modes) of a noisy `SimulationResult` — see its own docstring for
  why it uses flat time-domain weighting rather than pulse-shape
  weighting, and Investigation 9 in `INVESTIGATIONS.md` for the filter-
  function theory behind that choice.

See `tests/test_quantum*.py` (one file per topic: dispersion, impedance
mismatch, noise density sweep, nonlinearity, SNR, T1/T2, transmon leakage).
