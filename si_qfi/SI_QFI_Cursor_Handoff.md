# SI-QFI: Cursor Handoff Document
### Implementation Status & Integration Guide

---

## What This Document Is

This file is a companion to `SI_Quantum_Fidelity_Plugin_PRD.md`. It describes
what has already been implemented, what Cursor needs to complete, and the exact
API integration points with SignalIntegrity and QuTiP that require verification
against the installed library versions.

Drop both this file and the PRD into your Cursor project context.

---

## Repository Structure

```
si_qfi/
├── README.md                         Quick-start and structure overview
├── setup.py                          Package metadata and dependencies
├── SI_Quantum_Fidelity_Plugin_PRD.md Full design specification
│
├── __init__.py                       ✅ Top-level API: load_schematic, SourceWaveform, run
│
├── source/
│   ├── __init__.py                   ✅
│   └── waveform.py                   ✅ SourceWaveform, DRAG/Gaussian envelope generators
│
├── nonlinear/
│   ├── __init__.py                   ✅
│   ├── base.py                       ✅ NonlinearNode abstract base class
│   ├── saleh.py                      ✅ SalehModel (baseband) + SalehRealAxisModel (real-axis)
│   ├── volterra.py                   ✅ Volterra series (real-axis mode)
│   └── registry.py                   ✅ Annotation dict → NonlinearNode factory
│
├── noise/
│   ├── __init__.py                   ✅
│   ├── psd.py                        ✅ PSD from noise_figure / noise_density / thermal
│   ├── realization.py                ✅ Bandlimited Gaussian noise realizations
│   └── propagation.py                ✅ NoisePropagator (per-node → qubit plane)
│
├── simulation/
│   ├── __init__.py                   ✅
│   └── engine.py                     ✅ Two-pass NL + noise engine, compare_modes()
│
├── schematic/
│   ├── __init__.py                   ✅
│   ├── loader.py                     ⚠️  CURSOR: SI schematic loading (stubs)
│   └── transfer_function.py          ⚠️  CURSOR: SI S-parameter → transfer function (stubs)
│
├── quantum/
│   └── __init__.py                   ⚠️  CURSOR: verify QuTiP v5 API call signatures
│
├── output/
│   └── __init__.py                   🔲 Phase 3 stub
│
├── sweep/
│   └── __init__.py                   🔲 Phase 3 stub
│
└── tests/
    └── test_nonlinear.py             ✅ 17 unit tests, all passing, no SI/QuTiP required
```

Legend: ✅ complete and tested  ⚠️ needs Cursor  🔲 future phase stub

---

## Running Tests Without SI or QuTiP

The nonlinear math, noise generation, and PSD tests all run with only numpy/scipy:

```bash
pip install numpy scipy
cd si_qfi
python -c "
import sys; sys.path.insert(0, '.')
# paste contents of tests/test_nonlinear.py here, or run via pytest
"
```

All 17 tests currently pass. These cover:
- Saleh gain compression, zero-input safety, P1dB -1dB result
- The 3/4 describing function coefficient (PRD §5.1 core math)
- SalehRealAxisModel linear regime, compression, and 3rd-harmonic generation
- Volterra linear regime and compression
- Noise PSD from noise figure spec
- Baseband and RF noise realization statistics

---

## Cursor Task 1: schematic/loader.py — DONE (v0.7)

**File:** `si_qfi/schematic/loader.py`
**Status:** Fully implemented and verified against a real schematic
(`tests/test_schematic_basic.si`) — no Cursor work remains here.

### What it does now

Loads a `.si` SignalIntegrity schematic file and returns an `SISchematic`
dataclass with:
- List of all probe labels in the schematic (`port_names`)
- Verification that a source device named `source_label` exists (default
  `'VSource'`, overridable per schematic — `load_schematic(path,
  source_label=...)`)
- Confirmation that a probe named `qubit_probe_label` exists (default
  `'VQubit'`, overridable — `load_schematic(path, qubit_probe_label=...)`)

Real API details (verified against the installed SignalIntegrity source, not
guessed): devices live at `si_app.Drawing.schematic.deviceList`; properties
are read via `device['keyword'].GetValue()`; probes are identified by
`device['partname'].GetValue()` in a fixed set (`'Output'`,
`'DifferentialVoltageOutput'`, etc. — there is no `'VoltageProbe'` partname);
sources are identified by `device.netlist['DeviceName']` in
`('voltagesource', 'currentsource', 'networkanalyzerport')`, not by
partname. See the module docstring for the full list.

Note (v0.5 design change): the loader no longer scans for `NL_`-prefixed probes
or attempts a topological sort — `nl_probe_labels` and `_topological_sort_probes`
were removed. Nonlinear/noise node identity and NL propagation order now come
entirely from the `nonlinear` / `noise` dicts passed to `siq.run()` (PRD §3.2,
§3.5); the loader's only job is to expose the full `port_names` list so the
engine can validate those annotation keys against it via the already-implemented
`validate_node_labels()` (see bottom of `loader.py` — this function needs no
Cursor work, it's pure Python with no SI API calls).

### Functions to implement

**`_extract_port_names(si_app) -> list[str]`**

Return all VoltageProbe labels. Expected SI pattern:
```python
# Index the SI repo to verify exact attribute names
labels = []
for device in si_app.schematic.deviceList:
    if device.partname == 'VoltageProbe':
        labels.append(device.propertiesByName['ref'].value)
return labels
```
Verify: `partname`, `propertiesByName`, `ref`, `.value` — these may differ in the
current SI version.

**`_check_voltage_source_present(si_app, port_names)`**

Check that at least one VoltageSource device appears in `si_app.schematic.deviceList`.
Raise `ValueError` if not found.

### SI Headless App import pattern

```python
from SignalIntegrity.App.SignalIntegrityAppHeadless import SignalIntegrityAppHeadless
app = SignalIntegrityAppHeadless()
app.OpenProjectFile("path/to/schematic.si")
```

Verify this import path against the installed SI version. The repo has changed import
paths between versions.

---

## Cursor Task 2: schematic/transfer_function.py — DONE (v0.7)

**File:** `si_qfi/schematic/transfer_function.py`
**Status:** Fully implemented and verified against a real schematic
(`tests/test_schematic_basic.si`) — no Cursor work remains here.

### What it does now

There is no `si_app.SParameters()` method (the earlier stub guess was wrong).
The real hookup uses `si_app.TransferParameters()`, which returns a `Result`
(dict subclass) with `'source names'`, `'output waveform labels'`,
`'transfer matrices'` keys, plus a convenience `Result.FrequencyResponse
(from_name, to_name)` for by-name lookup — no port-index bookkeeping needed
at all, so `_label_to_port_index` was removed entirely. Segment H(f) between
two probes A→B (neither the source) is computed as
`H_{source→B}(f) / H_{source→A}(f)` (exact by linearity) — see the module
docstring for the full explanation, and `_extract_single_tf` /
`_source_referenced_response` / `_get_transfer_parameters` for the
implementation.

**v0.7 design change:** extraction (`extract_all_transfer_functions`,
`extract_noise_transfer_functions`, `_extract_single_tf`) no longer takes a
`SourceWaveform`, `mode`, or `carrier_hz` at all — it's purely
schematic-derived (`TransferFunction.h`/`.dt` are `None` until a separate
`compute_impulse_response(tf, fs, mode, carrier_hz)` call, made by the engine
once a concrete waveform is available). Real-axis mode now runs at the
schematic's own native sample rate (`native_sample_rate()`, 2× the top
frequency of its sweep) rather than interpolating H(f) onto the drive
waveform's fs — the waveform is resampled to match instead
(`SourceWaveform.rf_waveform_at()`). See PRD §3.3 for the full rationale.

### Already-implemented math (do not change)

- `compute_impulse_response(tf, mode, *, fs=None, carrier_hz=None)` — mode-dependent
  conversion, deferred to run() time (v0.7+): baseband uses
  `_tf_to_baseband_impulse()` and requires both `fs`/`carrier_hz` (raises
  `ValueError` if either is missing); real-axis reuses SI's own
  `FrequencyResponse.ImpulseResponse()` with no target rate at all (v0.10) —
  it ignores `fs` entirely and derives its own native rate from
  `TransferFunction.si_frequency_response` directly, rather than a
  hand-rolled IRFFT (v0.8). `freqs`/`H` are properties derived from
  `si_frequency_response` (v0.9), not stored fields.
- `_tf_to_baseband_impulse()` — shifts H(f+fc) to baseband, IFFTs to h̃(τ)
- `_interpolate_tf()` — magnitude + unwrapped phase interpolation onto new grid (baseband only)
- `compute_isolation_db()` — reverse transfer function isolation check

---

## Cursor Task 3: quantum/__init__.py — QuTiP v5 API Verification

**File:** `si_qfi/quantum/__init__.py`
**Status:** Logic complete. Three QuTiP API call sites need verification.

### Call site 1: propagator

Current code:
```python
U_actual = qt.propagator(H, T_gate, c_ops=c_ops, tlist=t_array)
```

In QuTiP v5 the signature is:
```python
qt.propagator(H, t, c_ops=[], args={}, options={}, **kwargs)
```
where `H` is a list `[H0, [op, coeff_array]]` and the solver needs to know the
time grid for array coefficients. Verify whether `tlist` is passed as a keyword
argument to `propagator` or whether the array coefficient is wrapped in `QobjEvo`
first.

The safest v5 pattern may be:
```python
H_evo = qt.QobjEvo([H0, [op_i, coeff_i], [op_q, coeff_q]], tlist=t_array)
U_actual = qt.propagator(H_evo, T_gate, c_ops=c_ops)
```

### Call site 2: average_gate_fidelity

Current code:
```python
F_i = qt.average_gate_fidelity(U_actual, U_ideal)
```

In QuTiP v5 this function exists but argument order may matter. Verify:
- Function name: `qt.average_gate_fidelity` or `qt.metrics.average_gate_fidelity`
- Argument order: (actual, target) or (target, actual)
- Whether it accepts superoperators (open system) or unitaries only

For open-system propagators (mesolve), `U_actual` is a superoperator (Liouville
space). The fidelity calculation differs. Check `qt.process_fidelity` as an
alternative for superoperator inputs.

### Call site 3: QobjEvo array coefficients

The Hamiltonian list format with numpy array coefficients:
```python
H = [H0, [op_i, coeff_i_array], [op_q, coeff_q_array]]
result = qt.sesolve(H, psi0, t_array)
```

In v5, the solver uses `tlist` (the time array passed to sesolve) to index into
the coefficient arrays. Confirm that when `tlist` is the same array as was used
to compute the waveform, the cubic spline interpolation is applied automatically
and no explicit `QobjEvo` wrapper is needed.

---

## Architecture Notes for Cursor

### Two-pass design (most important to understand)

The simulation engine (`simulation/engine.py`) runs two completely separate passes:

**Pass 1 — Nonlinear (deterministic, runs ONCE):**
```
source waveform
  → convolve with h_1(τ)   [SI segment 1 transfer function]
  → apply NL_1 model        [Saleh]
  → convolve with h_2(τ)   [SI segment 2 transfer function]
  → apply NL_2 model
  → ...
  → v_nl_qubit(t)           [distorted, noiseless waveform at qubit plane]
```

**Pass 2 — Noise (stochastic, runs N times per realization):**
```
For each noise node j (INDEPENDENT of NL segmentation):
  → generate noise realization v_noise_j(t) from S_v_j(f)
  → convolve with h_{j→qubit}(τ)   [full SI path from node j to qubit]
  → sum all j contributions
  → v_noise_qubit_i(t)

v_qubit_i(t) = v_nl_qubit(t) + v_noise_qubit_i(t)
```

This separation is valid because noise is linear — it can be propagated independently
of the nonlinear signal path. The `NoisePropagator` class in `noise/propagation.py`
implements Pass 2. The engine in `simulation/engine.py` orchestrates both.

### Nonlinear node naming convention (v0.5 design change)

NL nodes are **not** auto-detected from the schematic by prefix, and `NL_` is
**not** required or enforced anywhere in the code (`build_nonlinear_nodes` in
`registry.py` used to require it and no longer does — any existing probe label
works). The annotation dict key must exactly match an existing probe label in
the schematic, whatever that label is. `siq.run()` validates every
`nonlinear` / `noise` key against `schematic.port_names` via
`schematic.loader.validate_node_labels()` before doing anything else, and raises
a `ValueError` listing any labels that don't match a real probe. `NL_`-prefixed
names remain a reasonable convention for schematic readability, just not a
functional requirement.

```
Schematic probe label: "AMP1_OUT"       (no NL_ prefix required)
annotation key:        "AMP1_OUT"       ← must match exactly, checked at run() time
```

### Mode selection affects everything downstream

`mode='complex_baseband'` (default):
- Source waveform: complex envelope ũ(t)
- Transfer functions: shifted to baseband H̃(f) = H(f + fc)
- Nonlinear models: SalehModel
- QuTiP input: I/Q components of ũ(t) directly → rotating frame H(t)

`mode='real_axis'`:
- Source waveform: full RF v(t) = Re{ũ(t)·exp(j2πfc·t)}
- Transfer functions: full H(f) — no shift
- Nonlinear models: VolterraModel (`'volterra'`), or SalehRealAxisModel (built
  from the same `'saleh'` model string as the baseband case, dispatched by
  `mode` in `registry.py`)
- QuTiP input: demodulate v_qubit(t) to I/Q first, then rotating frame H(t)

### Diagnostic warnings

The engine emits Python `warnings.warn()` calls (not exceptions) for:
- Insufficient isolation between NL nodes (< -20 dB reverse coupling)
- Inadequate harmonic suppression at 3·fc (< 30 dB, baseband mode only)
- Memory regime: τ_reflection/T_pulse > 0.1 with memoryless model
- Memory regime: τ_reflection/T_pulse > 1.0 (reflection arrives post-pulse)
- Narrowband ratio BW/fc > 5% (baseband mode validity)
- Sample rate insufficient for harmonic tracking (real-axis mode)
- NL node small-signal gain deviates from unity by > 3 dB — likely double-counted
  amplifier gain against the schematic (PRD §3.6, new in v0.5; see
  `nonlinear/registry.py::_check_small_signal_gain`)

These are warnings not errors — the simulation proceeds. The user can escalate
them to errors with `warnings.filterwarnings('error', ...)` if desired.

---

## Key Math Reference (for implementing SI API calls)

### Transfer function to impulse response (already implemented)

For complex baseband mode, the SI-extracted H(f) (one-sided, at RF frequencies)
is shifted and IFFT'd:
```
H̃(f_bb) = H(f_bb + f_carrier)    # shift to baseband
h̃(τ) = IFFT[H̃]                   # complex baseband impulse response
```

For real-axis mode (v0.8): rather than IRFFT-ing `H(f)` ourselves, we call SI's
own `FrequencyResponse.ImpulseResponse()` on the native `FrequencyResponse`
object stashed on `TransferFunction.si_frequency_response` at extraction time
(v0.9 renamed this from the private-looking `_si_fr` — it's now the sole,
required source of `freqs`/`H`, not just an optional internal hint, since
TransferFunction is always schematic-backed). Its native sample rate
(`FrequencyList.TimeDescriptor()`'s `Fs = 2×top_frequency`) is verified to
match `native_sample_rate()` exactly, so there's no grid mismatch with the
drive waveform (resampled via `SourceWaveform.rf_waveform_at()`).

`_tf_to_baseband_impulse()` in `schematic/transfer_function.py` implements the
baseband path only — it needs the raw `H(f)`/`freqs` arrays and a carrier
frequency, since the frequency-shift `H(f+fc)` isn't something SI's own
`ImpulseResponse()` does.

### Noise propagation (already implemented)

For noise node j with PSD S_v_j(f) [V²/Hz]:
```python
noise_fft = sqrt(S_v_j(f) * df) * (randn(N) + 1j*randn(N))
v_noise_j = real(IFFT(noise_fft))                  # one realization
v_noise_j_at_qubit = convolve(h_{j→qubit}, v_noise_j)
```

### Saleh AM-AM describing function result (3/4 coefficient — key invariant)

For f(x) = x + a·x³ (cubic nonlinearity), after bandpass filtering:
```
A_out = A + (3a/4)·A³
```
The `(3/4)` factor is exact (from cos³θ = (3/4)cosθ + (1/4)cos3θ). It cancels
against the real-axis two-tone IP3 definition's `(4/3)` factor exactly, which
is why `SalehModel`'s (complex-baseband) beta_a-from-OIP3 formula is
`beta_a = 1/A_IP3,in²` (NOT `(4/3)/A_IP3,in²`) -- a real bug fixed in this
codebase after initially copying VolterraModel's real-axis formula wholesale
into the baseband derivation. `SalehRealAxisModel` (nonlinear/saleh.py),
which operates directly on the real waveform rather than an envelope, DOES
use the `(4/3)` factor, matching VolterraModel exactly -- see its module
docstring's "Real-axis variant" section for the full derivation of both
cases. All nonlinearity specs (`op1db_amplitude`/`oip3_amplitude`) are
OUTPUT-referred by convention (see nonlinear/volterra.py's module docstring).
The test `test_describing_function_coefficient` verifies the baseband case.

`SalehModel.from_op1db_oip3()`/`SalehRealAxisModel.from_op1db_oip3()` take no
`small_signal_gain`/gain argument at all -- alpha_a is always 1.0 (purely
output-referred nonlinearity, no gain-driven input/output conversion). The
general `SalehModel(alpha_a, beta_a, ...)` constructor is still the escape
hatch for a non-unity alpha_a in PRD §3.6's "when this convention does not
apply" case.

**Only ONE of op1db_amplitude/oip3_amplitude, never both:** `SalehModel`/
`SalehRealAxisModel.from_op1db_oip3()` and `VolterraModel(option='describing')`
each raise `ValueError` if given both (previously supported via a 5th-order
Volterra term / a `gamma_a` Saleh denominator term -- both removed to keep
the model surface small, PRD §5.1). Fitting from OIP3 alone therefore
*determines* (doesn't leave free) where the model's actual OP1dB falls --
different per model shape: ~10.1dB below OIP3 for Saleh's rational G[A],
~10.6dB for a plain cubic Volterra polynomial (PRD §5.1 table,
tests/test_nonlinear.py).

**`VolterraModel` also takes no `small_signal_gain` argument at all (2026-07-08):**
its `option='describing'`/`'diagonal'` k=1, m=0 coefficient (a1) is always
fixed at 1.0 -- not a constructor parameter, matching `SalehModel`'s
`alpha_a=1.0` precedent above. `nonlinear/registry.py::_build_volterra()`
raises `ValueError` if a spec dict includes `'small_signal_gain'`, same as
`_build_saleh()` already did. This was a real, if narrow, gap: the earlier
`from_op1db_oip3()` gain removal only touched Saleh's factory function;
`VolterraModel`'s *direct* constructor kept `small_signal_gain` (default
1.0) since it was a different code path and never explicitly asked about --
functionally it only ever needed to be 1.0 in practice (PRD §3.6's gain
convention, enforced at runtime by `_check_small_signal_gain`), so removing
it is a pure simplification, not a behavior change for any correctly-
configured node. `option='diagonal'` (caller supplies `coefficients`
directly) is unaffected -- still able to set a non-unity k=1 coefficient if
you construct it that way, still checked (warned, not rejected) at runtime.

---

## Dependencies

| Package | Version | Role | Required |
|---|---|---|---|
| numpy | ≥1.24 | Core numerical | Yes |
| scipy | ≥1.10 | FFT, spline, optimization | Yes |
| matplotlib | ≥3.7 | Plots | Yes |
| qutip | ≥5.0 | Quantum simulation | Yes (quantum module) |
| SignalIntegrity | latest | Schematic loading | Yes (schematic module) |
| scqubits | ≥4.0 | Transmon/fluxonium models | Recommended |

Install:
```bash
pip install numpy scipy matplotlib qutip
pip install scqubits
# SignalIntegrity: pip install SignalIntegrity or from source
```

---

## Phase Roadmap

**Phase 1 (current — complete the Cursor tasks above):**
- Wire up SI schematic loading and transfer function extraction
- Verify QuTiP v5 API calls
- Run end-to-end with a simple single-qubit schematic

**Phase 2:**
- Multi-segment propagation tests with real schematics
- scqubits Transmon/Fluxonium integration
- Lindblad T1/T2 secondary noise mode
- Rotating frame demodulation for real-axis mode

**Phase 3:**
- Multi-probe schematics (crosstalk analysis)
- Parameter sweep utilities (`sweep/parameter_sweep.py`)
- Fidelity budget decomposition (`sweep/budget.py`)
- Full output/plotting module (`output/plots.py`, `output/report.py`)
- Example notebooks

---

## Suggested First Integration Test

Once Cursor completes the SI API integration, use this minimal schematic to
validate end-to-end:

1. Create a SignalIntegrity schematic with:
   - `VoltageSource` → 50Ω coax 10cm → `VoltageProbe` labelled `QUBIT_PROBE`
   - No NL nodes, no noise (simplest case)

2. Run:
```python
import si_qfi as siq
import numpy as np

schematic = siq.load_schematic("test_coax.si")
# Build a simple Gaussian envelope SI Waveform (fill in SI Waveform API)
source = siq.SourceWaveform(carrier_freq_ghz=5.0, envelope=gaussian_si_waveform)

result = siq.run(schematic=schematic, source=source, n_realizations=1)

# Check: v_nl_qubit should be the Gaussian pulse attenuated and delayed by 10cm coax
print(result.v_nl_qubit[:10])
print("Warnings:", result.warnings)
```

3. Then add a Saleh NL node and verify the compression appears in the waveform.

4. Then add a noise node and verify ensemble spread in `result.v_qubit_ensemble`.

5. Finally wire up QuTiP fidelity for the X gate.

---

*Handoff document generated June 2026.*
*Companion files: SI_Quantum_Fidelity_Plugin_PRD.md, si_qfi_package.tar.gz*
