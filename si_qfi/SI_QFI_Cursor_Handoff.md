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
│   ├── saleh.py                      ✅ Saleh AM-AM/AM-PM model
│   ├── amam_ampm.py                  ✅ Tabulated AM-AM/AM-PM with cubic spline
│   ├── memory_polynomial.py          ✅ Memory polynomial model
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
- Tabulated AM-AM gain and phase accuracy
- Memory polynomial linear regime and zero-input
- Volterra linear regime and compression
- Noise PSD from noise figure spec
- Baseband and RF noise realization statistics

---

## Cursor Task 1: schematic/loader.py

**File:** `si_qfi/schematic/loader.py`
**Status:** All logic stubbed with `# --- CURSOR NOTE ---` markers.

### What it needs to do

Load a `.si` SignalIntegrity schematic file and return an `SISchematic` dataclass
with the following information extracted:
- List of all VoltageProbe labels in the schematic
- Verification that a VoltageSource device exists
- List of NL_ prefixed probe labels in topological signal-flow order
- Confirmation that `QUBIT_PROBE` label exists

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

**`_topological_sort_probes(si_app, nl_labels) -> list[str]`**

Sort NL_ probe labels in signal-flow order from VoltageSource to QUBIT_PROBE.
Options in increasing accuracy:
1. **Simple (good enough for well-drawn schematics):** sort by X coordinate of probe
   position in the schematic drawing.
2. **Robust:** trace connectivity from VoltageSource through the net graph and record
   the order NL_ probes are encountered. If SI exposes a netlist or connectivity graph,
   use it.

The warning in the current stub is acceptable for Phase 1 if topological sort is hard
— users can just name their probes in order and the annotation dict key order is used.

### SI Headless App import pattern

```python
from SignalIntegrity.App.SignalIntegrityAppHeadless import SignalIntegrityAppHeadless
app = SignalIntegrityAppHeadless()
app.OpenProjectFile("path/to/schematic.si")
```

Verify this import path against the installed SI version. The repo has changed import
paths between versions.

---

## Cursor Task 2: schematic/transfer_function.py

**File:** `si_qfi/schematic/transfer_function.py`
**Status:** Math helpers fully implemented. SI API calls stubbed.

### What it needs to do

For each pair of probe nodes (e.g. SOURCE → NL_AMP1_OUT, NL_AMP1_OUT → QUBIT_PROBE),
extract the voltage transfer function H(f) = V_out / V_in from the SignalIntegrity
schematic and convert it to a time-domain impulse response.

The math conversion (`_to_impulse_response`, `_interpolate_tf`,
`_tf_to_baseband_impulse`, `_tf_to_realaxis_impulse`) is already implemented and
does not need to change.

### Functions to implement

**`_extract_single_tf(si_app, label_in, label_out, fs, mode, carrier_hz)`**

This is the core SI API call. Needs to:
1. Call `si_app.SParameters()` to get the full S-parameter matrix
2. Map probe label strings to port indices
3. Extract H(f) = sp[k][port_out][port_in] for each frequency k
4. Call `_to_impulse_response(freqs, H, fs, mode, carrier_hz)` (already works)
5. Return a `TransferFunction` dataclass

Expected SI pattern:
```python
(sp, name) = si_app.SParameters()
freqs = np.array(sp.f())          # frequency list in Hz
# H_{out,in}(f) at each frequency:
H = np.array([sp[k][i_out][i_in] for k in range(len(freqs))], dtype=complex)
```

**Verify:** The S-parameter indexing order. In SI it may be:
- `sp[freq_index][port_out][port_in]` (most likely)
- `sp[port_out][port_in][freq_index]`

Check against the SI SParameters class definition.

**`_label_to_port_index(si_app, label) -> int`**

Map a probe label string to its integer port index in the S-parameter matrix.
Port ordering in SI's S-parameter output matches the order probes appear in the
schematic's port list. Build the mapping by enumerating:
```python
port_map = {}
for i, device in enumerate(si_app.schematic.deviceList):
    if device.partname == 'VoltageProbe':
        label = device.propertiesByName['ref'].value
        port_map[label] = i
```
Cache this mapping; don't call SI once per lookup.

### Important: SOURCE label handling

The engine uses `"SOURCE"` as a synthetic label for the VoltageSource input port.
`_label_to_port_index` needs to handle this by finding the VoltageSource device and
returning its port index. The VoltageSource is port 1 (input) of the schematic in
most SI setups — verify against the schematic.

### Already-implemented math (do not change)

- `_to_impulse_response()` — dispatches to baseband or real-axis conversion
- `_tf_to_baseband_impulse()` — shifts H(f+fc) to baseband, IFFTs to h̃(τ)
- `_tf_to_realaxis_impulse()` — irfft to real h(τ)
- `_interpolate_tf()` — magnitude + unwrapped phase interpolation onto new grid
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
  → apply NL_1 model        [Saleh / tabulated / memory polynomial]
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

### Nonlinear node naming convention

NL nodes are identified by probe labels starting with `NL_` in the SI schematic.
The annotation dict key must exactly match the probe label in the schematic.

```
Schematic probe label: "NL_AMP1_OUT"
annotation key:        "NL_AMP1_OUT"   ← must match exactly
```

### Mode selection affects everything downstream

`mode='complex_baseband'` (default):
- Source waveform: complex envelope ũ(t)
- Transfer functions: shifted to baseband H̃(f) = H(f + fc)
- Nonlinear models: Saleh, TabulatedAMAM, MemoryPolynomial
- QuTiP input: I/Q components of ũ(t) directly → rotating frame H(t)

`mode='real_axis'`:
- Source waveform: full RF v(t) = Re{ũ(t)·exp(j2πfc·t)}
- Transfer functions: full H(f) — no shift
- Nonlinear models: VolterraModel only
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

For real-axis mode:
```
h(τ) = IRFFT[H(f)]                # real impulse response
```

The functions `_tf_to_baseband_impulse()` and `_tf_to_realaxis_impulse()` in
`schematic/transfer_function.py` implement this. They need only the raw `H(f)`
array and `freqs` array from the SI SParameters call.

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
The `(3/4)` factor is exact (from cos³θ = (3/4)cosθ + (1/4)cos3θ).
The `MemoryPolynomial.from_p1db_ip3()` and `SalehModel.from_p1db_ip3()` both
use this result. The test `test_describing_function_coefficient` verifies it.

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
