# si_qfi.simulation

The main entry point: `engine.run()` orchestrates schematic loading output,
nonlinearity, and both noise mechanisms into a `SimulationResult` ready for
`quantum.gate_fidelity()`.

- **`engine.py`** — `run()`: the deterministic nonlinear pass (propagate
  the source waveform through each segment, applying any annotated
  nonlinearity), then the noise pass. Additive noise (`noise=`) is cheap —
  drawn independently per realization and summed onto the *same*
  deterministic waveform. Phase noise (`phase_noise=`) is not — because
  it's injected at the source before the nonlinear pass, the nonlinear
  pass is re-run once per Monte Carlo realization whenever it's enabled
  (see the module docstring and Investigation 10 in `INVESTIGATIONS.md`
  for why that's physically necessary, not just more expensive). Also:
  `SimulationResult` (the shared output type), `compare_modes()` (RMS
  disagreement between a `complex_baseband` and `real_axis` run of the
  same schematic/pulse), and the isolation/harmonic-suppression/
  narrowband-ratio/gain-normalization diagnostic checks.

See `tests/test_engine.py`, `tests/test_engine_noise.py`,
`tests/test_engine_phase_noise.py`.
