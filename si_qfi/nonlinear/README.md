# si_qfi.nonlinear

Amplifier/mixer nonlinearity models, applied at annotated schematic nodes
during `engine.run()`'s nonlinear pass. All models are normalized to unity
small-signal gain by convention — the SI schematic itself supplies a
device's actual linear gain (see the module docstrings for the full
derivation of why, and `registry.py`'s small-signal-gain runtime check).

- **`base.py`** — `NonlinearNode` abstract base class.
- **`saleh.py`** — `SalehModel` (complex-baseband AM-AM/AM-PM, the classic
  bounded-rational `G[A]=alpha_a/(1+beta_a*A^2)` form) and
  `SalehRealAxisModel` (the same curve applied directly to the real
  waveform, real-axis mode only). Both build from exactly one of
  `op1db_amplitude`/`oip3_amplitude` via `.from_op1db_oip3()`.
- **`volterra.py`** — `VolterraModel` (real-axis mode only), three
  parameterizations: `describing` (from a single OP1dB/OIP3 point, the
  cubic-kernel describing-function result), `diagonal` (caller-supplied
  per-tap coefficients), `full_kernel` (a full 3-tap Volterra series).
- **`registry.py`** — parses the `nonlinear={...}` annotation dict into
  `NonlinearNode` objects, dispatched by mode; runs the small-signal-gain
  deviation check.

See `tests/test_nonlinear.py` for the describing-function math and
OP1dB/OIP3 calibration checks, and `examples/nonlinearity_fidelity_demo.py`
/ Investigation 2 in `INVESTIGATIONS.md` for how compression actually costs
gate fidelity.
