# SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin
### Project Definition Document v0.4

---

## 1. Project Overview

**Project Name:** SI-QFI (Signal Integrity – Quantum Fidelity Impact)

**Summary:** SI-QFI is an open-source Python plugin that bridges classical microwave signal integrity (SI) analysis with quantum gate fidelity simulation. The user defines the entire drive chain as a single SignalIntegrity schematic. SI-QFI extracts transfer functions, propagates the drive waveform through each stage (applying nonlinear models at designated nodes), adds stochastic noise, and feeds the resulting waveform at the qubit plane into QuTiP to compute gate fidelity.

**Two simulation modes are supported:**
- **Complex baseband mode (default):** Propagates the complex envelope at baseband. Supports memoryless AM-AM/AM-PM and memory polynomial nonlinearity. Efficient sample rate; natural interface to QuTiP rotating frame. Valid when the narrowband assumption holds and channel harmonics are well-filtered.
- **Full real-axis mode:** Propagates the full real RF waveform on the complete frequency axis. Requires Volterra series nonlinearity. Exactly tracks harmonic generation and inter-harmonic mixing. Required when the narrowband assumption breaks down or when harmonic content reaching the qubit plane is non-negligible.

---

## 2. Core Design Principles

1. **SignalIntegrity is the single source of truth.** The entire drive chain is defined in a SignalIntegrity schematic. SI-QFI extracts transfer functions from it.

2. **Waveform propagation via transfer functions.** SI-QFI extracts H_k(ω) between adjacent probe nodes, converts to impulse responses h_k(τ), and propagates the waveform segment by segment via convolution.

3. **Two modes: complex baseband and full real-axis.** Mode selection determines the nonlinearity models available, the sample rate requirements, and the QuTiP interface. The modes are described in detail in §4.

4. **Noise and nonlinearity are computed in separate passes.** Noise nodes and nonlinear nodes are independent annotations with no requirement to coincide. Nonlinearity requires segmented propagation (order-dependent, non-commutative with the channel). Noise is a linear additive process: each noise source's contribution to the qubit plane is computed independently via the full transfer function from that node to QUBIT_PROBE, then superimposed at the qubit plane. Noise propagation runs once per realization; it does not need to respect or interleave with NL segmentation.

5. **Isolation is verified at runtime.** Reverse transfer functions between adjacent NL nodes are checked. Warnings are emitted if backward coupling is significant.

6. **Source waveform is a standard SignalIntegrity Waveform.** Duration and sample rate are encoded in the SI Waveform object. Only the carrier frequency is an additional parameter.

---

## 3. SignalIntegrity Schematic Contract

### 3.1 Required Elements

- **One voltage source** (`VoltageSource`): drive signal injection point.
- **One output probe** (`VoltageProbe`, labeled `QUBIT_PROBE`): qubit plane extraction point.

### 3.2 Nonlinear Node Probes

A nonlinear node is declared by placing a `VoltageProbe` labeled `NL_<name>` at the point in the schematic where nonlinearity is applied. SI-QFI uses these as cut points between propagation segments.

### 3.3 Transfer Function Extraction

For each segment between adjacent probes, SI-QFI extracts the voltage transfer function H_k(ω) = V_out(ω)/V_in(ω) from the SignalIntegrity schematic. In complex baseband mode this is shifted to baseband: H̃_k(f) = H_k(f + f_carrier). In real-axis mode the full one-sided transfer function H_k(f) is used directly.

### 3.4 Isolation and Harmonic Checks

**Isolation check:** Reverse transfer function H_reverse_k(ω) is computed between adjacent NL nodes. If max|H_reverse_k| over the signal band exceeds `isolation_threshold_db` (default: -20 dB), a warning is emitted recommending an isolator between those stages.

**Harmonic check (complex baseband mode only):** SI-QFI evaluates the transfer function at 3·f_carrier for each segment. If attenuation at 3·f_carrier relative to f_carrier is less than `harmonic_suppression_threshold_db` (default: 30 dB), a warning is emitted recommending real-axis mode.

**Inter-stage harmonic mixing check (complex baseband mode only):** If more than one NL node is present and isolation between them is less than `harmonic_suppression_threshold_db` at 3·f_carrier, a warning is emitted that harmonic re-mixing between stages may produce in-band products that complex baseband will not capture.

### 3.5 Node Ordering

Propagation order is determined by tracing signal flow from the voltage source to `QUBIT_PROBE`. If topology is ambiguous, SI-QFI raises an error.

---

## 4. Simulation Modes

### 4.1 Complex Baseband Mode (Default)

**What it propagates:** The complex envelope ũ(t) = A(t)·exp(jφ(t)), centered at DC after demodulation from the carrier. The real RF waveform is v(t) = Re{ũ(t)·exp(jω₀t)}.

**Sample rate requirement:** Determined by pulse bandwidth, not carrier frequency. For a 100 MHz bandwidth pulse: ~200–400 MSa/s. Typically two orders of magnitude lower than real-axis mode.

**Nonlinearity models available:** AM-AM/AM-PM (memoryless), memory polynomial. See §5.

**QuTiP interface:** I/Q components of ũ(t) feed directly into the rotating frame Hamiltonian. No demodulation step needed at the QuTiP boundary.

**Valid when:**
- Signal bandwidth << carrier frequency (narrowband assumption)
- Channel provides adequate harmonic suppression (verified by harmonic check, §3.4)
- No significant inter-stage harmonic re-mixing (verified by inter-stage check, §3.4)

**Mathematical justification:** See §5.1.

### 4.2 Full Real-Axis Mode

**What it propagates:** The full real RF waveform v(t) on the complete frequency axis, including carrier, harmonics, and intermodulation products at all frequencies.

**Sample rate requirement:** Must satisfy Nyquist for the highest significant harmonic. For a 5 GHz carrier with third-harmonic tracking: minimum 30 GSa/s, practically 40–60 GSa/s. For fifth harmonic: 60+ GSa/s. This is set by the SignalIntegrity Waveform sample rate.

**Nonlinearity models available:** Volterra series only. See §5.4. AM-AM/AM-PM is not available in real-axis mode because the envelope extraction it relies on is only defined for narrowband signals.

**QuTiP interface:** v(t) is passed as a real array coefficient to QobjEvo. The ODE solver must resolve the carrier oscillation — rotating frame demodulation is strongly recommended (SI-QFI performs this automatically before passing to QuTiP). Alternatively, QuTiP can be driven at full RF if the user explicitly disables demodulation, but integration will be slow.

**Valid when:**
- Narrowband assumption does not hold (wideband pulses, multi-tone drives)
- Harmonic content is not adequately suppressed before the qubit plane
- Two or more NL stages lack sufficient inter-stage filtering at harmonic frequencies
- Maximum physical accuracy is required regardless of computational cost

**Forced when:**
- More than one NL node is present AND harmonic check fails between any adjacent pair
- User explicitly selects `mode="real_axis"`

---

## 5. Nonlinearity Models

### 5.1 Mathematical Foundation: Why Complex Baseband AM-AM is Exact (Narrowband Case)

Consider a memoryless nonlinearity of the form:

```
f(x) = x + a·x³
```

applied to the bandpass signal x(t) = A(t)·cos(ω₀t + φ(t)), where A(t) and φ(t) are
slowly varying (narrowband assumption: their bandwidth << ω₀).

**Step 1: Expand the cubic term**

```
x³(t) = A³(t)·cos³(ω₀t + φ(t))
```

Apply the trigonometric identity cos³(θ) = (3/4)·cosθ + (1/4)·cos(3θ):

```
x³(t) = A³(t) · [(3/4)·cos(ω₀t + φ(t)) + (1/4)·cos(3ω₀t + 3φ(t))]
```

**Step 2: Collect terms by frequency**

```
y(t) = f(x(t))
     = [A(t) + (3a/4)·A³(t)] · cos(ω₀t + φ(t))     ← in-band (at ω₀)
     + (a/4)·A³(t) · cos(3ω₀t + 3φ(t))              ← third harmonic (at 3ω₀)
```

**Step 3: Apply bandpass filtering**

The channel after the nonlinearity attenuates the 3ω₀ component (verified by harmonic check).
After filtering, only the in-band term survives:

```
y_filtered(t) = [A(t) + (3a/4)·A³(t)] · cos(ω₀t + φ(t))
```

**Step 4: Read off the AM-AM function**

The output amplitude as a function of input amplitude is:

```
A_out(A) = A + (3a/4)·A³  =  G[A] · A
```

where G[A] = 1 + (3a/4)·A² is the amplitude-dependent gain (the AM-AM function).

In complex baseband, with ũ(t) = A(t)·exp(jφ(t)), the output is:

```
ũ_out(t) = [1 + (3a/4)·|ũ_in(t)|²] · ũ_in(t)
```

**This is exact** — not an approximation — given the three assumptions:
1. Narrowband: A(t), φ(t) vary slowly relative to 1/ω₀
2. Harmonic filtering: 3ω₀ content is removed by the subsequent channel
3. Memoryless: f(x) depends only on x(t), not x(t-τ)

Note the (3/4) describing function coefficient. It is **not** the polynomial coefficient a
directly — the cos³ identity splits energy between fundamental and harmonic, and only
(3/4) of the cubic energy returns to the fundamental. This is the classical describing
function result for a cubic nonlinearity.

**General odd polynomial nonlinearity:**

For f(x) = Σₙ aₙ·x^(2n+1), the describing function generalizes:

```
G[A] = Σₙ aₙ · D(2n+1) · A^(2n)
```

where D(2n+1) is the describing function coefficient for order (2n+1):

```
D(1)  = 1
D(3)  = 3/4
D(5)  = 5/8
D(7)  = 35/64
D(2n+1) = (1/4ⁿ) · C(2n+1, n)    where C(2n+1,n) is the binomial coefficient
```

The Saleh model and tabulated AM-AM curves are empirical representations of this same
function — they capture G[A] directly from measured data without requiring knowledge of
the underlying polynomial coefficients.

**Relation to IP3 and P1dB:**

For the cubic-only model f(x) = x + ax³, the third-order intercept (IP3) in terms of
input amplitude A is:

```
A_IP3 = sqrt(-4/(3a))     (requires a < 0 for compression)
```

The 1 dB compression point satisfies G[A_1dB] = 10^(-1/20) ≈ 0.891, giving:

```
A_1dB = sqrt(-0.109 · 4 / (3a))  ≈  0.383 · A_IP3
```

This 9.6 dB relationship between P1dB and IP3 is the well-known result for cubic-only
nonlinearity, and provides a direct way to fit the coefficient a from a single P1dB
measurement.

### 5.2 AM-AM / AM-PM Models (Complex Baseband Mode Only)

**Saleh Model (default for amplifiers):**

```
G[A] = α_a · A / (1 + β_a · A²)
Φ[A] = α_φ · A² / (1 + β_φ · A²)

ũ_out(t) = G[|ũ_in(t)|] · exp(j·Φ[|ũ_in(t)|]) · ũ_in(t) / |ũ_in(t)|
```

Fit parameters (α_a, β_a, α_φ, β_φ) from measured P1dB and phase-vs-power data via
`scipy.optimize.curve_fit`. Convenience fit from P1dB + IP3 scalars also provided.

**Tabulated AM-AM / AM-PM:**

Measured curves supplied as numpy arrays of (V_in, V_out) and (V_in, phase_rad) pairs.
Interpolated via `scipy.interpolate.CubicSpline`. No parametric fit required. This is
the most accurate representation of a measured amplifier.

### 5.3 Memory Polynomial (Complex Baseband Mode Only)

Extends the memoryless AM-AM to include finite memory depth M (in samples):

```
ũ_out(t) = Σ_{k∈{1,3,5}} Σ_{m=0}^{M} a_{km} · ũ(t-mT) · |ũ(t-mT)|^(k-1)
```

Odd orders only. M = 0 reduces to memoryless AM-AM (with Volterra coefficients). 
Valid when reflection delay τ is comparable to pulse width but harmonic suppression
still holds. Coefficients identified via least-squares from swept-tone measurements,
or set from a P1dB / IP3 parameterization with user-specified memory depth.

### 5.4 Volterra Series (Real-Axis Mode Only)

The full Volterra series for a real-valued signal, truncated to third order:

```
y(t) = ∫ h₁(τ)·x(t-τ)dτ
      + ∫∫ h₂(τ₁,τ₂)·x(t-τ₁)·x(t-τ₂) dτ₁dτ₂
      + ∫∫∫ h₃(τ₁,τ₂,τ₃)·x(t-τ₁)·x(t-τ₂)·x(t-τ₃) dτ₁dτ₂dτ₃
```

where x(t) is the full real RF waveform. This exactly captures:
- In-band distortion (same as AM-AM/AM-PM in the narrowband limit)
- Third harmonic generation at 3ω₀
- Inter-harmonic mixing products (harmonic of one stage mixing with fundamental
  of the next to produce in-band content)
- Even-order products at DC and 2ω₀

The h₁ kernel is the linear impulse response — the same transfer function used
for linear segment propagation. h₂ and h₃ are the nonlinear kernels.

**Practical parameterization options:**

Option A — Diagonal kernels (memory polynomial equivalent on real axis):
```
y(t) ≈ Σₙ Σₘ aₙₘ · x(t-mT)^n
```
Odd and even orders both present. Coefficients from swept-tone measurement or
fit from P1dB, IP3, second-order intercept (IP2) for even-order terms.

Option B — Full kernel specification:
User supplies h₁(τ), h₃(τ₁,τ₂,τ₃) as sampled arrays. h₂ can be set to zero
for systems with odd-symmetric nonlinearity (most amplifiers).

Option C — Measured describing function (default for real-axis mode):
SI-QFI measures h₁ from the SI transfer function and parameterizes h₃ from 
P1dB and IP3 measurements using the cubic kernel result (§5.1). This provides
a practical real-axis simulation without requiring full kernel identification.

**Relationship to complex baseband mode:**

In the narrowband limit, the Volterra series and the complex baseband AM-AM model
produce identical in-band output. The Volterra series additionally produces harmonic
content that AM-AM discards. Real-axis mode with Volterra and complex baseband mode
with AM-AM are therefore expected to agree at the qubit plane whenever the harmonic
suppression check passes — this provides a built-in cross-validation path.

---

## 6. Source Waveform Definition

```python
# The envelope is a standard SignalIntegrity Waveform.
# Duration and sample rate are already encoded in the SI Waveform object.
# Only carrier frequency is additionally required.

source = siq.SourceWaveform(
    carrier_freq_ghz = 5.0,
    envelope = si_waveform,    # SignalIntegrity Waveform: complex baseband envelope
)
```

In complex baseband mode, `si_waveform` carries the complex envelope directly. In
real-axis mode, SI-QFI modulates it onto the carrier:

```
v_source(t) = Re{ envelope(t) · exp(j·2π·f_carrier·t) }
```

The SI Waveform sample rate must satisfy Nyquist for the intended mode — SI-QFI checks
this at load time and raises an error if the sample rate is insufficient for real-axis
mode harmonic tracking (< 6·f_carrier for third-harmonic, < 10·f_carrier for fifth).

---

## 7. Noise Handling

### 7.1 Primary Mode: Stochastic Waveform Noise (Default)

Noise nodes and nonlinear nodes are independent annotations. A noise node can coincide
with an NL node, sit between NL nodes, or exist at a point in the schematic that has no
NL annotation — all cases are handled identically.

For each noise node j, SI-QFI precomputes the full transfer function h_{j→qubit}(τ)
from that node directly to QUBIT_PROBE (extracted from the SI schematic in setup, step 2
of the execution flow). This transfer function spans the entire remaining channel —
including any NL nodes between j and QUBIT_PROBE — but only its linear component, since
noise propagation is treated as a linear process (noise is assumed small relative to
signal, so it does not significantly drive the nonlinear elements).

For each realization, a noise voltage v_noise_j(t) is drawn at each annotated node and
propagated independently to the qubit plane:

```python
# PSD from noise specification
S_v_j(f) = k_B * T_eff * R_source        # noise figure spec
T_eff = T_phys * (10**(NF_dB/10) - 1)   # excess noise temperature

# One noise realization at node j
noise_fft = np.sqrt(S_v_j(freqs) * df) * (randn(N) + 1j*randn(N))
v_noise_j = np.real(np.fft.ifft(noise_fft))

# Propagate to qubit plane via precomputed transfer function
v_noise_j_at_qubit = np.convolve(h_j_to_qubit, v_noise_j)
```

All noise contributions are summed at the qubit plane (independent sources, linear
superposition) and added to the deterministic NL-distorted waveform:

```
v_qubit_i(t) = v_nl_qubit(t) + Σ_j  v_noise_j_at_qubit_i(t)
```

Ensemble average fidelity:
```
F_avg = (1/N) · Σᵢ F(v_qubit_i(t))
```

In complex baseband mode, noise is generated as a baseband complex process (bandlimited
around DC). In real-axis mode, noise is a real bandpass process centered at f_carrier.
In both cases the same independent-propagation structure applies.

```python
noise_nodes = {
    "NL_AMP1_OUT": {
        "type": "noise_figure",
        "noise_figure_db": 3.0,
        "temperature_k": 300.0,
    },
    "CRYO_INPUT": {
        "type": "noise_density",
        "single_sided_psd_v2_per_hz": 1.6e-20,
        "temperature_k": 0.02,
    },
    "COAX_CRYO": {
        "type": "thermal",
        "temperature_k": 4.0,
    },
}
```

### 7.2 Secondary Mode: Lindblad Collapse Operators

Optional intrinsic qubit decoherence (T1, T2) added as QuTiP Lindblad collapse operators
on top of the stochastic waveform noise. Each realization runs `mesolve` (open system)
instead of `sesolve` (closed system). Allows fidelity budget to separate SI-chain noise
from intrinsic qubit loss.

### 7.3 Ensemble Parameters

Default: 100 realizations. Configurable via `n_realizations`. Standard error on mean
fidelity is reported so the user can assess convergence.

---

## 8. SI-QFI Execution Flow

Nonlinearity and noise are computed in two separate passes. The nonlinear pass produces
a deterministic distorted waveform at the qubit plane. The noise pass independently
accumulates stochastic noise contributions from all annotated nodes at the qubit plane.
The two are summed per realization before the QuTiP simulation.

```
1. LOAD AND VALIDATE
   └─ Load SignalIntegrity schematic + nonlinear_nodes + noise_nodes annotations
   └─ Validate: VoltageSource present, QUBIT_PROBE present
   └─ Check mode: complex_baseband (default) or real_axis

2. SETUP: TRANSFER FUNCTION EXTRACTION
   └─ Extract H_k(ω) for each segment between adjacent NL probe pairs
      (Source → NL_1, NL_1 → NL_2, ..., NL_n → QUBIT_PROBE)
   └─ Extract H_{j→qubit}(ω) for each noise node j: full transfer function
      from node j to QUBIT_PROBE, regardless of NL segmentation
   └─ Mode conversion:
      - Complex baseband: shift all H(ω) → H̃(f) = H(f + f_carrier)
      - Real-axis: use H(ω) directly
   └─ Run checks: isolation between NL nodes, harmonic suppression (baseband mode),
      sample rate adequacy

3. SETUP: BUILD SOURCE WAVEFORM
   └─ Complex baseband: SI Waveform envelope used directly as ũ(t)
   └─ Real-axis: modulate envelope onto carrier → v(t) = Re{ũ(t)·exp(j2πf_c·t)}

4. NONLINEAR PASS  [deterministic — runs once]

   Propagate the source waveform through the segmented NL chain:

   Initialize: v_nl(t) = source waveform

   For each segment k in propagation order:
      i.  Convolve: v_nl(t) = h_k(τ) * v_nl(t)
      ii. If NL node k: apply nonlinearity
          - Complex baseband: AM-AM/AM-PM or memory polynomial on ũ_nl(t)
          - Real-axis: Volterra series on v_nl(t)

   Result: v_nl_qubit(t)  — deterministic distorted waveform at qubit plane

5. NOISE PASS  [stochastic — repeated N times per realization]

   Noise nodes are independent of NL segmentation. For each realization i:

   For each annotated noise node j:
      a. Generate noise realization at node j:
         - Compute S_v_j(f) from noise specification (NF, temperature, or PSD)
         - Draw bandlimited Gaussian realization v_noise_j(t) from S_v_j(f)
      b. Propagate to qubit plane:
         - v_noise_j_at_qubit(t) = h_{j→qubit}(τ) * v_noise_j(t)
         - h_{j→qubit} is the precomputed full transfer function from node j
           to QUBIT_PROBE (already extracted in step 2, used directly here)

   Sum all noise contributions at the qubit plane:
      v_noise_qubit_i(t) = Σ_j  v_noise_j_at_qubit_i(t)

   Combine with deterministic waveform:
      v_qubit_i(t) = v_nl_qubit(t) + v_noise_qubit_i(t)

6. QUTIP SIMULATION  [per realization]

   For each realization i:
      - Complex baseband: extract I/Q of ũ_qubit_i(t) directly
      - Real-axis: demodulate v_qubit_i(t) → I/Q
      - Build QobjEvo H(t) from I/Q array coefficients (cubic spline)
      - Run sesolve (closed) or mesolve with T1/T2 collapse ops (open)
      - Compute gate fidelity F_i via propagator + average_gate_fidelity

7. REPORT
   └─ F_avg = mean(F_i),  F_std = std(F_i),  F_sem = F_std / sqrt(N)
   └─ Fidelity budget: separate NL distortion contribution from noise contribution
      (run with noise disabled to isolate NL-only fidelity loss)
   └─ Emit all diagnostic warnings from step 2
```

**Why this separation is correct:**

Noise is a linear process. The voltage contribution of noise source j at the qubit plane
is h_{j→qubit}(τ) * v_noise_j(t), regardless of what nonlinear operations occur at
other points in the chain — because linearity means the noise contribution propagates
independently of the signal. This would not be valid if noise sources were located
inside a nonlinear element (where the NL would mix signal and noise), but for the
intended use case — noise injected at specific nodes in the chain, not within the NL
elements themselves — the separation is exact.

The fidelity budget in step 7 exploits this cleanly: running the nonlinear pass alone
(no noise) gives the NL-only fidelity loss; running with noise gives the combined loss;
the difference is the noise contribution.

---

## 9. QuTiP Simulation Backend

### 9.1 QobjEvo Array Coefficients with Cubic Spline

Both modes ultimately deliver I/Q baseband components to QuTiP. The v_qubit(t) waveform
(or its demodulated equivalent) is passed as a numpy array coefficient in QobjEvo. QuTiP
v5 applies cubic spline interpolation internally.

```python
# envelope_i, envelope_q: real np.ndarray (I and Q components)
# from complex baseband mode directly, or demodulated from real-axis mode

H = [H0,
     [qt.sigmax() / 2,  eta * envelope_i],
     [qt.sigmay() / 2,  eta * envelope_q]]

# Closed system
result = qt.sesolve(H, psi0, t_array)

# Open system (with T1/T2)
c_ops = [sqrt(1/T1) * qt.destroy(n), sqrt(1/T2) * qt.sigmaz()]
result = qt.mesolve(H, psi0, t_array, c_ops=c_ops)
```

### 9.2 Gate Fidelity

```python
U_actual  = qt.propagator(H, T_gate, c_ops=c_ops)
U_ideal   = ideal_gate_unitary(gate_name, n_levels)
F_i       = qt.average_gate_fidelity(U_actual, U_ideal)
```

### 9.3 Qubit Model

- **scqubits:** `siq.quantum.from_scqubits(transmon_obj)`
- **Manual:** `siq.quantum.QubitModel(H0_matrix, n_levels)`
- **Analytic transmon:** anharmonic oscillator, default convenience model

---

## 10. Module Structure

```
si_qfi/
├── schematic/
│   ├── loader.py              # Load and validate SI schematic
│   ├── transfer_function.py   # Extract H_k(ω) between probe pairs
│   ├── topology.py            # Determine NL node propagation order
│   └── checks.py              # Isolation, harmonic suppression, inter-stage checks
│
├── source/
│   └── waveform.py            # SourceWaveform: carrier + SI Waveform → modulated signal
│
├── nonlinear/
│   ├── base.py                # NonlinearNode base class
│   ├── saleh.py               # Saleh AM-AM/AM-PM (complex baseband)
│   ├── amam_ampm.py           # Tabulated AM-AM/AM-PM with spline (complex baseband)
│   ├── memory_polynomial.py   # Memory polynomial (complex baseband)
│   ├── volterra.py            # Volterra series, real-axis mode only
│   └── registry.py            # Map NL probe label → model
│
├── noise/
│   ├── annotation.py          # Parse noise_nodes dict
│   ├── psd.py                 # S_v(f) from noise figure / temperature / density
│   ├── realization.py         # Bandlimited Gaussian noise realization from S_v(f)
│   └── propagation.py         # Precompute h_{j→qubit} per noise node; propagate realizations independently
│
├── simulation/
│   ├── engine.py              # Two-pass main loop: NL pass (deterministic) + noise pass (stochastic)
│   ├── ensemble.py            # N realizations → F_avg, F_std, F_sem
│   └── diagnostics.py         # All runtime warnings and checks
│
├── quantum/
│   ├── qubit_model.py         # H0: scqubits, manual, analytic transmon
│   ├── hamiltonian.py         # QobjEvo H(t) from I/Q envelope arrays
│   ├── demodulation.py        # Demodulate real v_qubit(t) → I/Q for real-axis mode
│   └── fidelity.py            # propagator + average_gate_fidelity per realization
│
├── sweep/
│   ├── parameter_sweep.py     # Fidelity vs. schematic or annotation parameter
│   └── budget.py              # Per-impairment fidelity budget
│
├── output/
│   ├── plots.py               # Waveform, noise, fidelity distribution, sweep plots
│   └── report.py              # Summary: F_avg, F_std, warnings, budget
│
└── examples/
    ├── single_qubit_coax.py
    ├── amplifier_compression_baseband.py
    ├── amplifier_compression_realaxis.py    # Cross-validation example
    ├── two_qubit_crosstalk.py
    └── drag_pulse_dispersion.py
```

---

## 11. Target API

```python
import si_qfi as siq

# 1. Load schematic
schematic = siq.load_schematic("qubit_driveline.si")

# 2. Source waveform (SI Waveform carries duration and sample rate)
source = siq.SourceWaveform(carrier_freq_ghz=5.0, envelope=drag_si_waveform)

# 3. Nonlinear annotations
nonlinear_nodes = {
    # Complex baseband mode: Saleh model
    "NL_AMP1_OUT": {
        "model": "saleh",
        "alpha_a": 2.16, "beta_a": 1.15,
        "alpha_phi": 0.0, "beta_phi": 0.0,
    },
}

# Real-axis mode: Volterra
nonlinear_nodes_realaxis = {
    "NL_AMP1_OUT": {
        "model": "volterra",
        "kernel_order": 3,
        "p1db_dbm": 10.0,          # Parameterize h3 from P1dB + IP3
        "ip3_dbm": 20.0,
        "memory_depth": 5,
    },
}

# 4. Noise annotations (same for both modes)
noise_nodes = {
    "NL_AMP1_OUT": {"type": "noise_figure", "noise_figure_db": 3.0, "temperature_k": 300.0},
    "CRYO_INPUT":  {"type": "thermal", "temperature_k": 0.02},
}

# 5. Run simulation — complex baseband (default)
result = siq.run(
    schematic      = schematic,
    source         = source,
    nonlinear      = nonlinear_nodes,
    noise          = noise_nodes,
    n_realizations = 200,
    mode           = "complex_baseband",   # or "real_axis"
)

# result.v_qubit_ensemble: list of np.ndarray (complex envelope or real RF per mode)
# result.t_array:          np.ndarray (time axis from SI Waveform)
# result.warnings:         list of diagnostic strings

# 6. Gate fidelity
qubit = siq.quantum.Transmon(Ej_GHz=20.0, Ec_MHz=200.0, n_levels=5)

fidelity_result = siq.quantum.gate_fidelity(
    v_qubit_ensemble           = result.v_qubit_ensemble,
    t_array                    = result.t_array,
    qubit                      = qubit,
    ideal_gate                 = "X",
    coupling_strength_per_volt = 2e7,    # eta: rad/(s·V)
    T1_us = 50.0,    # Optional: intrinsic decoherence via Lindblad
    T2_us = 30.0,
)

print(f"Gate fidelity: {fidelity_result.F_avg:.5f} ± {fidelity_result.F_sem:.6f}")
fidelity_result.plot_waveform()
fidelity_result.plot_fidelity_hist()
fidelity_result.plot_budget()

# 7. Cross-validation: compare modes
result_real = siq.run(schematic=schematic, source=source,
                      nonlinear=nonlinear_nodes_realaxis, noise=noise_nodes,
                      n_realizations=200, mode="real_axis")
siq.compare_modes(result, result_real)   # Warns if modes disagree beyond tolerance
```

---

## 12. Dependencies

| Package | Role | Required |
|---|---|---|
| SignalIntegrity | Schematic simulation, transfer function extraction | Yes |
| numpy, scipy | Numerical core, FFT, spline, noise generation | Yes |
| qutip >= 5.0 | sesolve, mesolve, QobjEvo, propagator, fidelity | Yes |
| matplotlib | Plots | Yes |
| scqubits | Transmon/fluxonium Hamiltonians | Recommended |
| PyYAML | YAML annotation file support | Optional |
| qutip-jax | JAX-accelerated solvers for large systems | Optional |

---

## 13. Development Phases

### Phase 1 — Core Pipeline (MVP)
- Schematic loader and validator
- Transfer function extraction, single segment (no NL nodes)
- Complex baseband mode: Saleh AM-AM, stochastic noise, QobjEvo sesolve
- Single-qubit transmon, X/Y/DRAG fidelity
- Isolation, harmonic, and sample rate diagnostic checks

### Phase 2 — Full Nonlinearity
- Tabulated AM-AM/AM-PM, memory polynomial (complex baseband)
- Real-axis mode: Volterra series, full RF propagation, demodulation before QuTiP
- Multi-segment propagation (N NL nodes)
- Lindblad secondary mode (T1/T2)
- Cross-validation utility: compare_modes()

### Phase 3 — Multi-Qubit and Analysis
- Multi-probe crosstalk schematics
- scqubits integration
- Fidelity budget decomposition
- Parameter sweeps
- Example notebooks

---

## 14. Open Questions

1. **Volterra kernel parameterization from P1dB/IP3:** The diagonal-kernel approximation
   (Option C in §5.4) fits h₃ from P1dB and IP3 using the cubic kernel result. This
   assumes odd-symmetric nonlinearity (h₂ = 0). Confirm this is adequate for the
   amplifier types expected in superconducting qubit drive chains, or whether even-order
   terms need a separate IP2 specification.

2. **Demodulation phase reference for real-axis mode:** Demodulating v_qubit(t) to I/Q
   requires a phase reference at f_carrier. If the carrier phase drifts between the
   source and the qubit plane (due to dispersive channel), the demodulation must use
   the actual carrier phase at the qubit plane, not the source phase. Define whether
   SI-QFI extracts this automatically from the transfer function phase at f_carrier or
   requires user specification.

3. **Coupling constant η:** Voltage-to-Hamiltonian coupling (rad/s per volt) depends on
   qubit coupling geometry. User-specified in v1. Future: compute from lumped-element
   sub-circuit.

4. **Multi-qubit probes:** Multiple QUBIT_PROBE_n probes for crosstalk analysis. Defer
   to Phase 3.

5. **Noise correlation:** Noise sources assumed independent. Common-mode noise (shared
   ground return) would require a correlation matrix. Defer to future version.

---

## 15. References

- **SignalIntegrity** (github.com/TeledyneLeCroy/SignalIntegrity)
- **QuTiP v5** (arxiv.org/abs/2412.04705) — QobjEvo, sesolve, average_gate_fidelity
- **scqubits** — superconducting qubit Hamiltonians
- **Saleh (1981)** — AM-AM/AM-PM parametric model
- **Schetzen (1980)** — Volterra and Wiener Theories of Nonlinear Systems
- **DRAG pulse shaping** — Motzoi et al., PRL 103, 110501 (2009)
- **Describing function method** — Gelb & Vander Velde (1968)

---

*Document maintained by: [Author]*
*Last updated: June 2026*
*Status: Pre-development / Seed document v0.4*
