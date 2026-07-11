# SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin

Bridges **SignalIntegrity** drive-chain simulation with **QuTiP** gate fidelity analysis.

See `SI_Quantum_Fidelity_Plugin_PRD.md` for the full project definition.

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
│   ├── loader.py             *** CURSOR: SI schematic loading API ***
│   └── transfer_function.py  *** CURSOR: SI S-parameter → transfer function ***
├── quantum/__init__.py       QuTiP H(t) builder, fidelity, Transmon model
│                             *** CURSOR: verify QuTiP v5 API calls ***
└── tests/test_nonlinear.py   Unit tests (no SI or QuTiP required)
```

---

## What is implemented (no SI/QuTiP needed)

- All nonlinear models: Saleh (baseband + real-axis variant), Volterra
- Noise PSD computation from all spec types
- Stochastic noise realization generation (baseband and real-axis)
- NoisePropagator (two-pass architecture)
- Simulation engine two-pass loop
- Diagnostic checks (isolation, harmonic suppression, narrowband)
- Quantum module: Transmon H₀, gate unitary library, FidelityResult
- Unit tests for all of the above

## What Cursor needs to complete

### 1. `schematic/loader.py` — SignalIntegrity API
All functions marked `# --- CURSOR NOTE ---`. Key tasks:
- `_extract_port_names(si_app)`: enumerate VoltageProbe labels from schematic
- `_check_voltage_source_present(si_app)`: find VoltageSource device

Nonlinear/noise node identity and order come from the `nonlinear`/`noise` dicts
passed to `siq.run()`, not from schematic scanning — see `validate_node_labels()`
in this file (already implemented, no SI API needed).

Start with the SI repo's `SignalIntegrityAppHeadless` class and inspect
`app.schematic.deviceList` to understand device enumeration.

### 2. `schematic/transfer_function.py` — SI S-parameter extraction
Key function: `_extract_single_tf(si_app, label_in, label_out, ...)`:
- Call `si_app.SParameters()` to get S-parameter matrix
- Map probe labels to port indices
- Extract H(f) = sp[freq][port_out][port_in]
- Call `_to_impulse_response(...)` (already implemented)

### 3. `quantum/__init__.py` — QuTiP v5 API verification
Marked with `# --- CURSOR NOTE ---`. Verify:
- `qt.propagator(H, T, c_ops=[], tlist=tlist)` calling convention
- `qt.average_gate_fidelity(U_actual, U_ideal)` function name and arg order
- `qt.QobjEvo([H0, [op, coeff_array]], tlist=t_array)` array coefficient interface

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

## Running tests (no SI or QuTiP required)

```bash
pip install pytest
pytest tests/test_nonlinear.py -v
```

---

## Quick usage (once SI and QuTiP are wired up)

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

qubit = siq.quantum.Transmon(Ej_GHz=20.0, Ec_MHz=200.0, n_levels=5)

fid = siq.quantum.gate_fidelity(
    v_qubit_ensemble           = result.v_qubit_ensemble,
    t_array                    = result.t_array,
    qubit                      = qubit,
    ideal_gate                 = "X",
    coupling_strength_per_volt = 2e7,
)
print(f"F = {fid.F_avg:.5f} ± {fid.F_sem:.6f}")
```
