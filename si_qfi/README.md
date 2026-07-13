# SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin

Bridges **SignalIntegrity** drive-chain simulation with **QuTiP** gate fidelity analysis.

See `SI_Quantum_Fidelity_Plugin_PRD.md` for the full project definition, and
`SI_QFI_Cursor_Handoff.md` for implementation-status detail on the SI/QuTiP
integration points. **`INVESTIGATIONS.md` is a running report log of physics
investigations built on top of this bridge** (nonlinearity vs. gate fidelity,
bandwidth/dispersion, impedance mismatch/reflections, etc.) — read it for
worked examples of what this codebase is actually for.

**Status: the SI↔QuTiP bridge is fully wired up and tested.** Schematic
loading, transfer-function extraction, and the QuTiP fidelity backend are
all implemented, verified against real schematics, and covered by the test
suite — see `SI_QFI_Cursor_Handoff.md` for the detailed history of each
piece if you need it.

---

## Package structure

```
si_qfi/
├── __init__.py               Top-level API: load_schematic, SourceWaveform, run
├── source/waveform.py        SourceWaveform, DRAG/Gaussian envelope generators
├── nonlinear/
│   ├── saleh.py              SalehModel (baseband) + SalehRealAxisModel (real-axis)
│   ├── volterra.py           Volterra series (real-axis only)
│   └── registry.py           Parse annotation dict → NonlinearNode objects
├── noise/
│   ├── psd.py                Noise PSD from noise_figure / noise_density / thermal
│   ├── realization.py        Bandlimited Gaussian noise realization generation
│   └── propagation.py        NoisePropagator: per-node noise → qubit plane
├── simulation/engine.py      Two-pass NL + noise engine, compare_modes()
├── schematic/
│   ├── loader.py             SI schematic loading, incl. schematic-level
│   │                         variable overrides (load_schematic(variables=...))
│   └── transfer_function.py  SI S-parameter → transfer function extraction
├── quantum/__init__.py       QuTiP H(t) builder, gate_fidelity(), Transmon model
└── tests/                    Full test suite (nonlinear/noise/engine/schematic/
                              quantum, unit + SI/QuTiP-backed integration tests)

examples/                     Runnable investigation demos, one per INVESTIGATIONS.md section
INVESTIGATIONS.md             Running report log of physics findings from the examples/ demos
```

---

## What is implemented

- All nonlinear models: Saleh (baseband + real-axis variant), Volterra
- Noise PSD computation from all spec types
- Stochastic noise realization generation (baseband and real-axis)
- NoisePropagator (two-pass architecture)
- Simulation engine two-pass loop
- Diagnostic checks (isolation, harmonic suppression, narrowband)
- SI schematic loading (`schematic/loader.py`), including passing
  schematic-level `<Variables>` overrides through SI's own
  `OpenProjectFile(args=...)` mechanism
- SI transfer-function extraction (`schematic/transfer_function.py`), both
  baseband and real-axis impulse response conversion
- Quantum module: Transmon H₀, gate/state fidelity (`gate_fidelity()`),
  optional T1/T2 Lindblad decay, propagators and final states
- Unit tests for all of the above, plus SI/QuTiP-backed integration tests
  and the `examples/` investigation demos

---

## Setup

`setup.py` lives at the repository root (one level above this `si_qfi/` package
directory — note the repo root and the package directory share the same name,
which is easy to confuse). Install once, in editable mode, from the repo root:

```bash
cd ..            # to the repo root, where setup.py lives (not this si_qfi/ folder)
pip install -e .
```

After that, `import si_qfi` works from any directory, in any script or test
run — you do not need to `cd` into this folder or manipulate `sys.path`
yourself. (Running scripts/tests *without* this install step, from inside
this `si_qfi/` package folder, is the most common cause of
`ModuleNotFoundError: si_qfi` — there is no `si_qfi` package nested inside
itself for Python to find.)

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests that need SignalIntegrity or QuTiP `pytest.importorskip` themselves and
are skipped automatically if those packages aren't installed; the nonlinear/
noise/PSD unit tests run with only numpy/scipy.

---

## Quick usage

```python
import si_qfi as siq

schematic = siq.load_schematic("driveline.si")
source    = siq.SourceWaveform(carrier_freq_ghz=5.0, envelope=my_si_waveform)

result = siq.run(
    schematic      = schematic,
    source         = source,
    nonlinear      = {"NL_AMP1_OUT": {"model": "saleh", "alpha_a": 2.16,
                                       "beta_a": 1.15, "alpha_phi": 0.0, "beta_phi": 0.0}},
    noise          = {"NL_AMP1_OUT": {"type": "noise_figure",
                                       "noise_figure_db": 3.0, "temperature_k": 300.0}},
    n_realizations = 100,
    mode           = "complex_baseband",
)

qubit = siq.quantum.QubitModel(H0=..., n_levels=2)  # or siq.quantum.Transmon(...)

fid = siq.quantum.gate_fidelity(
    result,
    qubit,
    ideal_gate                 = "X",
    coupling_strength_per_volt = 2e7,
)
print(f"F = {fid.F_avg:.5f} ± {fid.F_sem:.6f}")
```

`load_schematic()` also accepts a `variables={...}` dict to override any
`<Variables>` declared in the `.si` schematic file itself (see
`tests/test_schematic_impedance_mismatch.si` and Investigation 5 in
`INVESTIGATIONS.md` for a worked example).

See `examples/` for complete, runnable end-to-end demos.
