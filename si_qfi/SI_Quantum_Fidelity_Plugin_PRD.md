# SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin
### Design specification and derivations (originally "Project Definition Document v0.17")

---

## How to read this document

This is the **design specification**: the math behind the two simulation
modes, the derivations behind the nonlinearity models, and the reasoning
behind the noise-injection architecture. It is kept because the code cites
it heavily — roughly 44 docstring references point at section numbers here
(§3.6's gain convention alone is cited about a dozen times across
`nonlinear/`, and is quoted in a runtime warning users can actually see), so
the section numbering is a stable API of its own and does not get
renumbered.

**Where this document and the code disagree, the code is correct.** This was
written as a forward-looking spec, and implementation refined several
decisions after the fact. Two consequences worth stating plainly:

- **§7.1's Johnson-noise formula is wrong and was never implemented.** It
  writes `S_v = k_B * T_eff * R_source`; the correct, implemented form is
  `S_v = 4 * k_B * T * R`. See the inline correction at that section, and
  `noise/psd.py`'s module docstring, which documents the discrepancy and
  cross-checks the implemented version against SignalIntegrity's own native
  computation.
- **§10's file layout has been updated to match the shipped package.** Where
  this spec originally proposed a module that was later folded into another
  file (or not built at all), that is now noted there rather than listed as
  though it exists. The `sweep/` package it once proposed does not exist and
  ships no stub.

For what is actually implemented and how, see the top-level `README.md`
(feature list, runnable example) and `SI_QFI_Cursor_Handoff.md`
(implementation status, exact SignalIntegrity/QuTiP API resolutions). For
the physics this design has been used to investigate — and several real bugs
found along the way — see `INVESTIGATIONS.md` alongside this file.

---

## 1. Project Overview

**Project Name:** SI-QFI (Signal Integrity – Quantum Fidelity Impact)

**Summary:** SI-QFI is an open-source Python plugin that bridges classical microwave signal integrity (SI) analysis with quantum gate fidelity simulation. The user defines the entire drive chain as a single SignalIntegrity schematic. SI-QFI extracts transfer functions, propagates the drive waveform through each stage (applying nonlinear models at designated nodes), adds stochastic noise, and feeds the resulting waveform at the qubit plane into QuTiP to compute gate fidelity.

**Two simulation modes are supported:**
- **Complex baseband mode (default):** Propagates the complex envelope at baseband. Supports memoryless AM-AM/AM-PM nonlinearity (Saleh, tabulated). Efficient sample rate; natural interface to QuTiP rotating frame. Valid when the narrowband assumption holds and channel harmonics are well-filtered.
- **Full real-axis mode:** Propagates the full real RF waveform on the complete frequency axis. Requires a real-axis nonlinearity model (Volterra series, §5.3; or the real-axis Saleh variant, §5.4). Exactly tracks harmonic generation and inter-harmonic mixing. Required when the narrowband assumption breaks down or when harmonic content reaching the qubit plane is non-negligible.

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

- **One voltage/current source device**, whose `ref` matches `source_label` (default `'VSource'`) — the drive injection point, and the reference point every relative transfer function is computed against (§3.3).
- **One output probe**, whose `ref` matches `qubit_probe_label` (default `'VQubit'`) — the qubit plane extraction point.

Both labels are parameters of `siq.load_schematic(path, qubit_probe_label='VQubit', source_label='VSource')`, overridable per schematic — there is nothing special about the default strings beyond being the convention used by `tests/test_schematic_basic.si`.

### 3.2 Nonlinear Node Probes

**Nonlinear nodes are declared entirely by the `nonlinear` dict passed to `siq.run()` — not by any naming convention in the schematic itself.** A key in `nonlinear` names the `VoltageProbe` in the schematic where that nonlinearity is applied, and SI-QFI uses these as cut points between propagation segments. Probes are commonly named `NL_<name>` by convention for readability, but this is not required, auto-detected, or enforced anywhere in the code — any existing probe label works, `NL_` prefix or not.

Before use, SI-QFI validates that every key in `nonlinear` (and, separately, every key in `noise`) names a probe that actually exists in the loaded schematic, raising a `ValueError` listing the offending label(s) and the full set of valid probe names if not (`schematic.loader.validate_node_labels()`). This is a deliberate single-source-of-truth design: earlier revisions had the schematic loader independently scan for `NL_`-prefixed probes, which could silently diverge from the `nonlinear` dict the user actually passed to `run()` — e.g. a typo in an annotation key would mean that stage's nonlinearity was never applied, with no warning. The current design has one source of truth (the annotation dicts) and one validation step against the schematic.

### 3.3 Transfer Function Extraction

For each segment between adjacent probes, SI-QFI extracts the voltage transfer function H_k(ω) = V_out(ω)/V_in(ω) from the SignalIntegrity schematic. **Extraction is waveform-agnostic**: it depends only on the schematic, never on any particular drive waveform's sample rate, carrier frequency, or mode — it returns H_k(ω) purely as a function of frequency. Converting a raw H_k(ω) into a time-domain impulse response h_k(τ) — which does need a target sample rate, mode, and (for baseband) carrier frequency — is a separate, later step, done once a specific `SourceWaveform` is available (i.e. at `siq.run()` time, not at schematic-load time).

**Implementation note (source-referenced ratio):** SignalIntegrity's headless API only exposes transfer functions referenced to the source (`source_label` → each probe), never probe-to-probe directly. SI-QFI computes every segment H_k(ω) between two probes A → B (neither of which is `source_label`) as the ratio H_{source→B}(f) / H_{source→A}(f) — exact by linearity, since both are responses of the same linear network to the same source. When A *is* `source_label`, H_k is simply H_{source→B}(f) directly, no division needed. This is why `source_label` is required as an explicit reference for every extraction, not just for waveform injection. This division uses SI's own `FrequencyResponse` division operator directly (it's already implemented there), rather than SI-QFI dividing raw numpy arrays itself.

**Reuse of SI's own frequency-domain machinery, without a hard dependency:** `TransferFunction` stores the native SI `FrequencyResponse` object it was extracted from, as `si_frequency_response` (untyped — no SI import needed to declare the field). This lets real-axis mode's impulse response reuse SI's own `FrequencyResponse.ImpulseResponse()` rather than a hand-rolled IRFFT (see below). `TransferFunction` deliberately does **not** inherit from SI's `FrequencyResponse` class, even though it might seem natural to: that would require importing SignalIntegrity at module level in `schematic/transfer_function.py`, which is imported unconditionally by the engine and then by the top-level `si_qfi` package — making SignalIntegrity a hard dependency just to `import si_qfi`, and breaking the "core math works without SI installed" property the rest of the codebase preserves by importing SI lazily inside function bodies only where it's actually used.

**`freqs`/`H` are derived, not stored:** since `TransferFunction` is always constructed from a schematic (never manually, with independent frequency-domain data), `freqs` and `H` are `@property` methods computed from `si_frequency_response` on access, rather than separately-stored fields that could in principle drift out of sync with it. Every existing call site reads them as plain attributes (`tf.H`, `tf.freqs[-1]`, etc.), so this is transparent to callers.

**`compute_impulse_response(tf, mode, *, fs=None, carrier_hz=None)`:** `fs`/`carrier_hz` are keyword-only and optional, because only complex baseband mode actually needs them (and raises `ValueError` if either is missing). Real-axis mode calls `si_frequency_response.ImpulseResponse()` with no target rate at all — SI derives its own native rate directly from the FrequencyResponse's own frequency grid, ignoring any `fs` the caller passes (which, per `native_sample_rate()`, could only ever equal that same native rate anyway — passing it would be redundant, not just unused).

**Who adapts to whom — mode-dependent:** the two modes disagree on whose sample rate governs the impulse response, because only one of them has a schematic-intrinsic rate:

- **Real-axis mode:** the schematic has one natural, waveform-independent sample rate — 2× the top frequency of its own frequency sweep (matches SI's `CalculationProperties.UserSampleRate` for an evenly-spaced sweep from 0 to `EndFrequency` — verified against SI's own `FrequencyList.TimeDescriptor()` source, which derives the identical value). h_k(τ) is computed directly at that rate via SI's own `FrequencyResponse.ImpulseResponse()` — no interpolation of H(f), and no hand-rolled IRFFT either; SI's version additionally applies a fractional-delay correction around the FFT that a naive IRFFT skips — and the **drive waveform is resampled to match it** (mirroring SI's own `Waveform.Adapt()` idiom for fitting a waveform to a target time grid). Consequently the sample-rate-adequacy check (harmonic tracking, §6) runs against this native rate, not against the waveform's own `fs` — if it fails, increase the schematic's frequency sweep resolution, not the waveform's sample rate.
- **Complex baseband mode:** there is no schematic-intrinsic baseband rate — it's fundamentally a pulse-bandwidth choice (the whole point of §4.1 is running two orders of magnitude below real-axis rate). So baseband mode keeps interpolating H(f) onto the target grid implied by the envelope's own fs/carrier, as before — the **transfer function adapts to the waveform** here, the opposite direction from real-axis mode.

### 3.4 Isolation and Harmonic Checks

**Isolation check:** Reverse transfer function H_reverse_k(ω) is computed between adjacent NL nodes. If max|H_reverse_k| over the signal band exceeds `isolation_threshold_db` (default: -20 dB), a warning is emitted recommending an isolator between those stages.

**Harmonic check (complex baseband mode only):** SI-QFI evaluates the transfer function at 3·f_carrier for each segment. If attenuation at 3·f_carrier relative to f_carrier is less than `harmonic_suppression_threshold_db` (default: 30 dB), a warning is emitted recommending real-axis mode.

**Inter-stage harmonic mixing check (complex baseband mode only):** If more than one NL node is present and isolation between them is less than `harmonic_suppression_threshold_db` at 3·f_carrier, a warning is emitted that harmonic re-mixing between stages may produce in-band products that complex baseband will not capture.

### 3.5 Node Ordering

Propagation order is the `nonlinear` dict's key insertion order: `SOURCE → nonlinear.keys()[0] → nonlinear.keys()[1] → ... → QUBIT_PROBE`. The user is responsible for listing keys in signal-flow order; SI-QFI does not trace schematic topology to infer or verify it (automatic topological ordering from the schematic graph was considered but dropped in favor of the single-source-of-truth design in §3.2 — the annotation dict already needs to exist and be ordered for the model configuration itself, so reusing its order avoids a second, potentially conflicting mechanism).

### 3.6 Linear/Nonlinear Gain Split Convention

**The SI schematic is the sole source of a device's linear (small-signal) gain, phase, and frequency response — including active devices.** SignalIntegrity performs generic linear network analysis and can import an active device's measured or datasheet small-signal S-parameters (e.g. an amplifier's low-power S2P file) exactly as it would a passive component; the device does not need to be passive or reciprocal. When a nonlinear element such as an amplifier appears in the drive chain:

1. Import its small-signal (linear, low-power) S-parameters as a block in the SI schematic.
2. Place the `NL_<name>` probe at that device's physical output node.
3. Configure the corresponding nonlinear model (§5) to represent **only the amplitude-dependent deviation** from that already-captured linear response — not the device's gain itself.

**Every nonlinear node model must therefore be normalized to unity small-signal gain**: as input amplitude A → 0, the model's output must satisfy G[A] → 1 (not the device's real gain). Concretely:

| Model | Unity small-signal gain means... |
|---|---|
| Saleh (complex baseband) | `alpha_a = 1.0` |
| Saleh (real-axis variant) | `alpha_a = 1.0` |
| Volterra (describing) | order-1, tap-0 coefficient (a1), always fixed `= 1.0` -- not a constructor parameter |
| Volterra (diagonal) | order-1, tap-0 coefficient, should be `≈ 1.0` -- caller-supplied, checked at runtime |
| Volterra (full kernel) | `h1` = unit impulse — no additional filtering unless intentionally modeling dispersion beyond what the schematic already captures |

This is why `SalehModel.from_op1db_oip3()`/`SalehRealAxisModel.from_op1db_oip3()` take no gain argument at all and always build `alpha_a=1.0` — there is no non-unity case to default away from; op1db_amplitude/oip3_amplitude alone fully determine a purely output-referred nonlinearity. `VolterraModel`'s `option='describing'`/`'diagonal'` constructor takes no `small_signal_gain` argument either (removed) — its k=1, m=0 coefficient (a1) is always fixed at 1.0 for the same reason. The general `SalehModel(alpha_a, beta_a, ...)` constructor still accepts an arbitrary alpha_a for the rare case below where this convention doesn't apply, and `VolterraModel(option='diagonal', coefficients=...)` still accepts an arbitrary k=1 coefficient if you supply `coefficients` yourself (checked at runtime, not enforced at construction).

**Runtime check:** `siq.run()` warns if any nonlinear node's small-signal gain deviates from 0 dB by more than 3 dB (§8, step 2).

**When this convention does not apply:** if a nonlinear element has no separate linear representation available (e.g. a passive mixer diode with no S-parameter model), the nonlinear node's small-signal gain may legitimately be non-unity — in that case the device's full response, gain included, belongs in the nonlinear annotation, and the SI schematic segment leading to that node must not also model the device.

---

## 4. Simulation Modes

### 4.1 Complex Baseband Mode (Default)

**What it propagates:** The complex envelope ũ(t) = A(t)·exp(jφ(t)), centered at DC after demodulation from the carrier. The real RF waveform is v(t) = Re{ũ(t)·exp(jω₀t)}.

**Sample rate requirement:** Determined by pulse bandwidth, not carrier frequency. For a 100 MHz bandwidth pulse: ~200–400 MSa/s. Typically two orders of magnitude lower than real-axis mode.

**Nonlinearity models available:** AM-AM/AM-PM (memoryless). See §5.

**QuTiP interface:** I/Q components of ũ(t) feed directly into the rotating frame Hamiltonian. No demodulation step needed at the QuTiP boundary.

**Valid when:**
- Signal bandwidth << carrier frequency (narrowband assumption)
- Channel provides adequate harmonic suppression (verified by harmonic check, §3.4)
- No significant inter-stage harmonic re-mixing (verified by inter-stage check, §3.4)

**Mathematical justification:** See §5.1.

### 4.2 Full Real-Axis Mode

**What it propagates:** The full real RF waveform v(t) on the complete frequency axis, including carrier, harmonics, and intermodulation products at all frequencies.

**Sample rate requirement:** Must satisfy Nyquist for the highest significant harmonic. For a 5 GHz carrier with third-harmonic tracking: minimum 30 GSa/s, practically 40–60 GSa/s. For fifth harmonic: 60+ GSa/s. This is set by the SignalIntegrity Waveform sample rate.

**Nonlinearity models available:** Volterra series, or the real-axis Saleh variant (a bounded rational AM-AM-style curve applied directly to the instantaneous real waveform, rather than to an envelope amplitude). See §5.3–5.4. The complex-baseband AM-AM/AM-PM models are not available in real-axis mode because the envelope extraction they rely on is only defined for narrowband signals — the real-axis Saleh variant sidesteps this by not extracting an envelope at all.

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

All models in this section are used under the gain convention defined in §3.6: each is normalized to unity small-signal gain, so the SI schematic supplies a device's linear response and these models supply only the amplitude-dependent deviation from it. This is also why the cubic model in §5.1 below is written as `f(x) = x + a·x³` rather than `f(x) = G₀·x + a·x³` — the linear term's coefficient is already fixed at 1 by convention.

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
A_1dB = sqrt((1 - 10^(-1/20)) · 4 / (3a) · (-1))  =  sqrt(1 - 10^(-1/20)) · A_IP3  ≈  0.330 · A_IP3
```

i.e. **A_1dB/A_IP3 ≈ -9.6 dB** (input-referred, single-tone CW fundamental) — the
well-known result for cubic-only nonlinearity: with only one free coefficient (a)
besides the fixed linear term, a single cubic term can match ONE of P1dB/IP3, not both
independently — P1dB then comes out wherever the cubic implies it, not as a
separately-controllable input. Output-referred (OP1dB vs. OIP3, i.e. including the 1dB
of compression itself), this becomes **≈ -10.6 dB** — see "OIP3-only implies a specific
OP1dB" below.

**OP1dB/OIP3 naming and output-referred convention:** all P1dB/IP3-style constructor
parameters in the code (`op1db_amplitude`, `oip3_amplitude`) are named with an explicit
`o` prefix and are **output-referred** — the actual output amplitude at that point, not
the input amplitude that produced it (converted internally to input-referred amplitudes
before applying the formulas above). The bare "IP3"/"P1dB" terminology above refers to
the general mathematical relationship, which is referencing-convention-agnostic.

**Only ONE of OP1dB or OIP3 may be specified — never both.** `VolterraModel
(option='describing')` and `SalehModel`/`SalehRealAxisModel.from_op1db_oip3()` each
have exactly one free shape parameter (`a3` for Volterra's cubic, `β_a` for Saleh's
rational `G[A] = α_a/(1+β_a·A²)`), which can only be calibrated from ONE point. An
earlier version of this codebase added a second free parameter to each model (a 5th-order
Volterra term, a `γ_a` Saleh denominator term) specifically to fit both points
simultaneously — removed, to keep the model surface small (only one nonlinearity
"shape" per model now, not a family of them). Both constructors raise `ValueError` if
given neither or both of op1db_amplitude/oip3_amplitude.

**OIP3-only implies a specific OP1dB (and vice versa) — NOT a free choice:** since each
model has only one free shape parameter, fitting from OIP3 alone *determines* where the
model's actual OP1dB falls (it isn't independently choosable). This value is
**different for each model**, because they're different nonlinearity *shapes* that only
agree asymptotically (same leading-order/IP3 behavior), not at the compression point:

| Model | OIP3-only implied OP1dB (output-referred) |
|---|---|
| SalehModel (baseband, bounded rational `G[A]`) | ≈ **-10.14 dB** below OIP3 |
| VolterraModel (real-axis, plain cubic — single-tone CW fundamental) | ≈ **-10.64 dB** below OIP3 |
| SalehRealAxisModel (real-axis, bounded rational) | ≈ **-10.1 dB** below OIP3 (matches baseband closely — see below) |

Verified in `tests/test_nonlinear.py` — closed-form for Saleh/Volterra, and via actual
single-tone simulation (FFT-extracted fundamental) for the real-axis models, since the
raw polynomial/rational curve evaluated at a *constant* input is a different, non-
physical quantity for a real-axis model (see nonlinear/volterra.py's and
nonlinear/saleh.py's module docstrings — a real-axis memoryless nonlinearity produces
harmonics when driven by an actual waveform, so "where does the fundamental compress by
1dB" and "where does the raw curve itself cross -1dB" are different questions).

**Domain-dependent OIP3 factor:** the β_a-from-OIP3 formula above is only exact for the
complex-baseband/envelope case (`β_a = 1/A_IP3,in²`, no 4/3 factor — the baseband
envelope's own (3/4) in-band reduction factor exactly cancels the (4/3) that appears in
the real-axis derivation in this section). The real-axis Saleh variant (§5.4) uses
`β_a = (4/3)/A_IP3,in²`, matching VolterraModel exactly, since it operates directly on
the real waveform. See nonlinear/saleh.py's module docstring for the full derivation of
both cases, and TestSalehBasebandRealAxisEquivalence in tests/test_nonlinear.py for the
verification that this factor makes the two domains describe the same physical
amplifier (to good — not exact — approximation, since the two rational curves aren't
algebraically identical).

### 5.2 Saleh AM-AM / AM-PM Model (Complex Baseband Mode Only)

The classic 2-parameter Saleh form:

```
G[A] = α_a / (1 + β_a · A²)
Φ[A] = α_φ · A² / (1 + β_φ · A²)

ũ_out(t) = G[|ũ_in(t)|] · exp(j·Φ[|ũ_in(t)|]) · ũ_in(t)
```

Built from EXACTLY ONE of OP1dB or OIP3 (output-referred; see §5.1's "Only ONE of OP1dB
or OIP3 may be specified") via `from_op1db_oip3()`, which takes no gain argument at all
-- alpha_a is always 1.0, a purely output-referred nonlinearity (see the gain-convention
table in §3.6). There is no fit-from-raw-measurement or tabulated-curve path in this
codebase currently -- `SalehModel(alpha_a, beta_a, ...)`'s general constructor is the
escape hatch for a non-unity alpha_a in the rare case where §3.6's convention doesn't
apply.

Even this classic form has a genuine breakdown amplitude: raw output y(A) =
α_a·A/(1+β_a·A²) peaks at A=1/sqrt(β_a) and DECLINES beyond that (never true for a real
amplifier short of hard clipping), even though gain itself compresses monotonically the
whole time. `max_monotonic_amplitude` = 1/sqrt(β_a); `apply_baseband()` warns if driven
past it -- see nonlinear/saleh.py module docstring.

### 5.3 Volterra Series (Real-Axis Mode Only)

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

Option A — Diagonal kernels (memoryless-per-tap polynomial on real axis):
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
EXACTLY ONE of a P1dB or IP3 measurement using the cubic kernel result (§5.1)
-- never both (see §5.1's "Only ONE of OP1dB or OIP3 may be specified"). This
provides a practical real-axis simulation without requiring full kernel
identification.

**Relationship to complex baseband mode:**

In the narrowband limit, the Volterra series and the complex baseband AM-AM model
produce identical in-band output. The Volterra series additionally produces harmonic
content that AM-AM discards. Real-axis mode with Volterra and complex baseband mode
with AM-AM are therefore expected to agree at the qubit plane whenever the harmonic
suppression check passes — this provides a built-in cross-validation path.

### 5.4 Real-Axis Saleh Variant (Real-Axis Mode Only)

`SalehRealAxisModel` (in nonlinear/saleh.py) applies the same bounded rational G[A]
curve as the complex-baseband Saleh model (§5.2), but directly to the instantaneous
real waveform x(t) rather than to an envelope magnitude:

```
G[A] = α_a / (1 + β_a · A²)

y(t) = G[x(t)] · x(t)
```

Since G[A] only uses A², this is well-defined for signed x(t) and automatically
odd-symmetric (matching a real amplifier's AM-AM curve). Applying it directly to the
real waveform means it generates genuine harmonic/intermodulation content via ordinary
waveform distortion — the same mechanism the Volterra series exploits — while staying
bounded/saturating everywhere (β_a > 0 guarantees no pole, unlike a truncated polynomial
which will eventually turn over -- see §5.2's max_monotonic_amplitude note, which
applies here too) (see the "Domain-dependent OIP3 factor" note in §5.1 for why β_a's
OIP3 formula differs by a factor of 4/3 between this model and the complex-baseband
Saleh model). No AM-PM: there is no separate envelope phase to modulate when operating
directly on a real, signed waveform.

`from_op1db_oip3()` works identically to the complex-baseband case (same spec-dict keys
via `registry.py`'s `'saleh'` model string, dispatched by `mode`), and the same
`max_monotonic_amplitude` diagnostic and overdrive warning apply.

This provides a second, bounded alternative to Volterra's `option='describing'` for
real-axis mode — useful when the truncated-polynomial breakdown risk described in §5.1
is a concern, at the cost of not being able to independently tune individual
harmonic/IMD amplitudes the way a general Volterra kernel can.

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
# CORRECTED. This section originally read `S_v_j(f) = k_B * T_eff * R_source`,
# which is too small by a factor of 4 and was NEVER implemented that way.
# The implemented (and correct) Johnson-noise form is 4*k_B*T*R -- the same
# convention SignalIntegrity's own native noise computation uses, verified in
# tests/test_engine_noise.py::test_johnson_psd_matches_4kTR_formula. Had the
# original line been implemented, every noise-figure-derived PSD would have
# been undersized by exactly 4x relative to every other noise source in the
# codebase. See noise/psd.py's module docstring.
S_v_j(f) = 4 * k_B * T_eff * R_source    # noise figure spec
T_eff = T_ref * (10**(NF_dB/10) - 1)    # excess noise temperature;
                                        # T_ref = 290 K (IEEE reference T0),
                                        # not the device's physical temperature

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
        # temperature_k is OPTIONAL here and defaults to 290 K -- the IEEE
        # noise-figure reference temperature T0, NOT the device's physical
        # operating temperature. Override it only if the figure was
        # specified against a non-standard reference.
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

### 7.4 Oscillator Phase Noise (Future Feature — Not Yet Implemented)

Everything in §7.1 relies on noise being **additive and small-signal linear**: drawn once
per node, propagated to the qubit plane through a precomputed *linear* transfer function,
and summed onto the deterministic NL-distorted waveform after the nonlinear pass has
already run (§8's "two passes, one deterministic, one stochastic" split). That assumption
is stated explicitly in §7.1 ("noise is assumed small relative to signal, so it does not
significantly drive the nonlinear elements") and holds fine for circuit/thermal noise
sources referred in as a voltage at some node.

Oscillator/LO phase noise does not fit this model, for two reasons:

1. **It's multiplicative, not additive.** A noisy carrier is `env(t)·cos(2π·f_c·t + φ(t))`,
   not `env(t)·cos(2π·f_c·t) + noise(t)`. It scales with the instantaneous drive envelope
   rather than existing independently of it.
2. **It originates upstream of the nonlinear chain, not at an arbitrary mid-chain node.**
   The LO sets the carrier the *entire* drive is built from, so a phase-noisy drive
   waveform is what actually enters every downstream NL stage — the nonlinearity sees and
   reshapes the noisy waveform, it doesn't get noise added onto its output afterward. This
   is true in both simulation modes; there is no version of "add it in post" that's exact
   once any real (AM-AM or AM-PM) nonlinearity sits between the LO and the qubit plane —
   including in real-axis mode, where a pure post-hoc multiplicative phase term on the
   final qubit-plane waveform still can't reproduce how that phase perturbation would have
   mixed with harmonic content generated by an upstream nonlinear stage.

**Planned approach:** treat phase noise as a Monte Carlo input to the nonlinear pass
itself, not as a third kind of node in the noise pass. Concretely: draw one phase-noise
realization φ_i(t) per Monte Carlo trial from its target PSD (reusing the existing
spectral-shaping machinery in `noise/realization.py` — drawing white Gaussian noise,
shaping by `sqrt(S_φ(f)·df)`, IFFT — the same recipe already used for §7.1's noise, just
producing a phase process instead of a voltage process), apply it to the source waveform
before the nonlinear pass (`v_i(t) = env(t)·exp(j·(2π·f_c·t + φ_i(t)))` conceptually, or
the equivalent baseband-phase-rotation for complex-baseband mode), and run the *full*
deterministic NL pass once per realization instead of once total. Existing per-node
additive noise (§7.1) still only needs drawing once per realization and propagating
linearly as before — it composes with this unchanged, it's just no longer the only
per-realization cost.

This means the nonlinear pass moves from "compute once, reuse for every realization"
to "compute once per Monte Carlo realization" whenever phase noise is enabled. Evaluating
the NL models themselves is cheap (they're simple pointwise/polynomial operations on an
already-computed array — the segment convolutions dominate, and those don't change), so
the expected cost increase is roughly proportional to `n_realizations`, not qualitatively
worse — but it does mean `n_realizations` becomes a cost driver for phase-noise runs in a
way it currently isn't for NL-only or additive-noise-only runs (where the NL pass is a
fixed one-time cost regardless of `n_realizations`). Should be benchmarked once built
rather than assumed.

Deliberately rejected alternatives, for the record:
- **Post-hoc multiplicative noise on `v_nl_qubit`** (cheapest, reuses the current
  architecture as-is): rejected as incomplete — only exact when no nonlinearity sits
  between the phase-noise injection point and the qubit plane, which defeats the purpose
  for exactly the schematics (amplifier chains) this tool is built around.
  Also unclear how to even define this consistently in real-axis mode (see point 2 above).
- **Multiplicative noise injected at one mid-chain node, additive-noise-style** (only
  re-run the *remaining* linear segments downstream of that node): rejected as still
  incomplete in general — only exact if nothing nonlinear sits downstream of the chosen
  injection node, which isn't guaranteed by the schematic contract and would need to be
  either assumed or explicitly checked.

Not scheduled; captured here so the design is settled before implementation is picked up
(see §14, Open Question 6, and §13 Phase 3).

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
   └─ Validate: every nonlinear_nodes / noise_nodes key names a probe that
      exists in the schematic (§3.2) — raises ValueError listing any that
      don't, before any transfer functions are extracted
   └─ Check mode: complex_baseband (default) or real_axis

2. SETUP: TRANSFER FUNCTION EXTRACTION (waveform-agnostic, §3.3)
   └─ NL_1..NL_n order = nonlinear_nodes dict key order (§3.5)
   └─ Extract raw H_k(ω) for each segment between adjacent NL probe pairs
      (Source → NL_1, NL_1 → NL_2, ..., NL_n → QUBIT_PROBE) — frequency
      domain only, no sample rate/carrier/mode involved yet
   └─ Extract raw H_{j→qubit}(ω) for each noise node j: full transfer function
      from node j to QUBIT_PROBE, regardless of NL segmentation
   └─ Run checks that only need frequency-domain data: isolation between
      NL nodes, harmonic suppression (baseband mode), small-signal gain
      normalization of each NL node (§3.6)

3. SETUP: BUILD SOURCE WAVEFORM AND CONVERT TO IMPULSE RESPONSES (§3.3)
   └─ Determine the sample rate + waveform this run actually convolves at:
      - Complex baseband: source.fs (the envelope's own rate); v(t) = ũ(t)
        used directly
      - Real-axis: the schematic's own native rate (2× top frequency of its
        sweep); the drive envelope is resampled to match it and modulated
        onto the carrier → v(t) = Re{ũ_resampled(t)·exp(j2πf_c·t)}; sample
        rate adequacy (harmonic tracking) is checked against this native
        rate, not the envelope's original fs
   └─ Convert each raw H_k(ω)/H_{j→qubit}(ω) to a time-domain impulse
      response h_k(τ)/h_{j→qubit}(τ) at that rate:
      - Complex baseband: shift H(ω) → H̃(f) = H(f + f_carrier), interpolated
        onto the envelope's grid, then IFFT
      - Real-axis: IFFT directly at the schematic's native rate — no
        interpolation needed

4. NONLINEAR PASS  [deterministic — runs once]

   Propagate the source waveform through the segmented NL chain:

   Initialize: v_nl(t) = source waveform

   For each segment k in propagation order:
      i.  Convolve: v_nl(t) = h_k(τ) * v_nl(t)
      ii. If NL node k: apply nonlinearity
          - Complex baseband: AM-AM/AM-PM on ũ_nl(t)
          - Real-axis: Volterra series or the real-axis Saleh variant on v_nl(t)

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

This is the **as-built** layout. Several modules this spec originally
proposed were folded into neighbouring files during implementation rather
than shipped separately; those are noted inline rather than listed as if
they exist. The top-level `README.md` carries the same tree with one-line
descriptions, and each subpackage has its own `README.md`.

```
si_qfi/
├── schematic/
│   ├── loader.py              # Load and validate SI schematic, incl. <Variables> overrides
│   ├── transfer_function.py   # Extract H_k(ω) between probe pairs; impulse response per mode
│   └── noise.py               # SI statistical-noise-source PSD + its transfer function
│                              # (proposed topology.py / checks.py were not built separately:
│                              #  NL propagation order and the isolation / harmonic-suppression
│                              #  checks both live in simulation/engine.py)
│
├── source/
│   └── waveform.py            # SourceWaveform: carrier + SI Waveform → modulated signal
│
├── nonlinear/
│   ├── base.py                # NonlinearNode base class
│   ├── saleh.py               # SalehModel (complex baseband) + SalehRealAxisModel (real-axis)
│   ├── volterra.py            # Volterra series, real-axis mode only
│   ├── tabulated.py           # TabulatedModel: generic AM-AM/AM-PM from a caller-supplied table
│   └── registry.py            # Map NL probe label → model; small-signal-gain check (§3.6)
│
├── noise/
│   ├── psd.py                 # S_v(f) from noise figure / temperature / density / colored
│   │                          # override, plus phase-noise PSD specs
│   │                          # (proposed annotation.py was not built: parsing the noise dict
│   │                          #  is part of psd.py)
│   ├── realization.py         # Bandlimited Gaussian noise realization from S_v(f); phase noise
│   └── propagation.py         # Precompute h_{j→qubit} per noise node; propagate independently
│
├── simulation/
│   └── engine.py              # Two-pass main loop: NL pass (deterministic) + noise pass
│                              # (stochastic); phase-noise injection; compare_modes()
│                              # (proposed ensemble.py / diagnostics.py were not built
│                              #  separately: ensemble statistics live in quantum/fidelity.py,
│                              #  and all runtime warnings/checks live here)
│
├── quantum/
│   ├── models.py              # H0: scqubits, manual, analytic transmon (spec'd as qubit_model.py)
│   ├── hamiltonian.py         # QobjEvo H(t) from I/Q envelope arrays, plus demodulate()
│   │                          # (proposed demodulation.py was folded in here)
│   ├── fidelity.py            # propagator + average_gate_fidelity per realization; ensemble
│   │                          # statistics; tuneup_amplitude()
│   └── snr.py                 # pulse_snr(): effective SNR of a noisy result
│
├── output/
│   └── __init__.py            # plot_waveform(), plot_nonlinearity()
│                              # (proposed plots.py / report.py split was not built; the richer
│                              #  report generation of §12 remains unimplemented)
│
└── examples/                  # One runnable demo per INVESTIGATIONS.md section
                               # (the illustrative filenames this spec originally listed were
                               #  superseded by the actual investigation demos; see
                               #  INVESTIGATIONS.md for the current set)
```

**Not built.** The `sweep/` package proposed by earlier drafts (parameter
sweeps and per-impairment fidelity budgets, §12) does not exist and no stub
ships for it. Parameter sweeps today are written as plain loops over
`siq.run()` — see `examples/noise_density_sweep_demo.py` for a worked one.

---

## 11. Target API

```python
import si_qfi as siq

# 1. Load schematic. qubit_probe_label / source_label default to
#    'VQubit' / 'VSource' (§3.1) -- pass overrides if your schematic uses
#    different probe/source ref names.
schematic = siq.load_schematic("qubit_driveline.si")
# schematic = siq.load_schematic("qubit_driveline.si",
#                                 qubit_probe_label="MyQubitProbe",
#                                 source_label="MySource")

# 2. Source waveform (SI Waveform carries duration and sample rate)
source = siq.SourceWaveform(carrier_freq_ghz=5.0, envelope=drag_si_waveform)

# 3. Nonlinear annotations
# NL_AMP1_OUT sits at AMP1's physical output. AMP1's small-signal S-parameters
# (its linear ~20 dB gain) are already a block in the .si schematic (§3.6), so
# alpha_a is normalized to 1.0 — this model supplies only AMP1's
# amplitude-dependent compression on top of that linear response.
nonlinear_nodes = {
    # Complex baseband mode: Saleh model
    "NL_AMP1_OUT": {
        "model": "saleh",
        "alpha_a": 1.0, "beta_a": 1.15,
        "alpha_phi": 0.0, "beta_phi": 0.0,
    },
}

# Real-axis mode: Volterra
# a1 (order-1 coefficient) is always fixed at 1.0 -- not a constructor
# parameter -- consistent with §3.6.
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
# result.fs, result.mode, result.carrier_freq_hz: enough for gate_fidelity()
#   to derive its own time axis internally -- there is no result.t_array
#   (engine.py deliberately doesn't track one; see SimulationResult's
#   docstring). result.warnings: list of diagnostic strings.

# 6. Gate fidelity -- pass `result` itself, not its individual fields; this
# is the one true call signature (gate_fidelity() derives t_array/mode/
# carrier_freq_hz from `result` directly, so they can't drift out of sync
# with what siq.run() actually simulated).
qubit = siq.quantum.Transmon(Ej_GHz=20.0, Ec_MHz=200.0, n_levels=5)

fidelity_result = siq.quantum.gate_fidelity(
    result                     = result,
    qubit                      = qubit,
    coupling_strength_per_volt = 2e7,    # eta: rad/(s·V)
    ideal_gate                 = "X",    # str, or a custom target unitary Qobj/ndarray
    T1_us = 50.0,    # Optional: intrinsic decoherence via Lindblad
    T2_us = 30.0,
)

# fidelity_result.noise_free (a SingleFidelity) is ALWAYS populated -- one
# solve on the deterministic result.v_nl_qubit, no noise realizations
# involved. fidelity_result.noise (a NoiseEnsembleFidelity) is populated
# here specifically because noise_nodes above was non-empty -- it's None
# whenever engine.run() was called with noise=None (or an empty dict).
print(f"Noise-free gate fidelity: {fidelity_result.noise_free.F_avg:.5f}")
if fidelity_result.noise is not None:
    print(f"Ensemble gate fidelity:   {fidelity_result.noise.F_avg:.5f} "
          f"± {fidelity_result.noise.F_sem:.6f}  (N={fidelity_result.noise.n_realizations})")
siq.output.plot_waveform(result)
# Fidelity budget decomposition is NOT implemented and no siq.sweep package
# ships -- see §10's "Not built" note. To budget contributions today, run
# siq.run() with and without each impairment enabled and compare; every
# investigation in INVESTIGATIONS.md does exactly that.

# 7. Calibration helper: tuneup_amplitude() searches an amplitude scale
# factor for a reference envelope shape to maximize the requested fidelity
# (gate or state -- same ideal_gate/target_state contract as gate_fidelity()
# above), replacing hand-rolled per-script calibration loops. Optimizes the
# noise-free fidelity only; call gate_fidelity() yourself on the returned
# .result for a final noisy/decohered number, as above.
tuned = siq.quantum.tuneup_amplitude(
    schematic, reference_shape=my_envelope_array, fs_envelope=2e9, carrier_ghz=5.0,
    qubit=qubit, coupling_strength_per_volt=2e7, ideal_gate="X",
)
print(f"Tuned scale={tuned.scale:.4f}  achieved={tuned.achieved}  "
      f"F_avg={tuned.fidelity.noise_free.F_avg:.5f}")

# fidelity_result.propagators is always populated (one QuTiP Qobj per
# realization -- a unitary, or a superoperator if T1_us/T2_us were given)
# at zero extra solve cost. fidelity_result.final_states() applies those
# channels to a given initial state (default |0>) to get the actual
# per-realization density matrices, with no re-solving needed:
rho_list = fidelity_result.final_states()   # or .final_states(my_initial_state)

# gate_fidelity() can also compare the evolved state directly against a
# target STATE (a ket or density matrix -- state fidelity, qt.fidelity())
# instead of, or alongside, ideal_gate (channel fidelity):
state_fid = siq.quantum.gate_fidelity(
    result, qubit, coupling_strength_per_volt=2e7,
    target_state=my_target_density_matrix,   # initial_state defaults to |0>
)
print(state_fid.state_F_avg)

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
- Real-axis mode: Volterra series and the real-axis Saleh variant, full RF propagation, demodulation before QuTiP
- Multi-segment propagation (N NL nodes)
- Lindblad secondary mode (T1/T2)
- Cross-validation utility: compare_modes()

### Phase 3 — Multi-Qubit and Analysis (partially complete)

Phases 1 and 2 above are done. Phase 3 was never completed as a unit; its
current status, item by item:

- **Oscillator phase noise (§7.4)** — ✅ done. Monte Carlo over the nonlinear
  pass, one full NL evaluation per realization, exactly as specified here.
  See `noise/psd.py`'s `phase_noise_psd_from_spec()`, `engine.run()`'s
  `phase_noise=` parameter, and Investigation 10 in `INVESTIGATIONS.md`.
- **Example notebooks** — ✅ done, though as `examples/` scripts (one per
  investigation) plus two standalone derivation notebooks in `notebooks/`,
  rather than as notebooks throughout.
- **scqubits integration** — ⚠️ written but unverified. `from_scqubits()`
  exists in `quantum/models.py` and has never been exercised against a real
  scqubits install; its own docstring says so.
- **Fidelity budget decomposition** — 🔲 not built. No `sweep/` package
  ships (see §10). Budget contributions by running with and without each
  impairment and comparing.
- **Parameter sweeps** — 🔲 not built as a utility. Written as plain loops
  over `siq.run()`; see `examples/noise_density_sweep_demo.py`.
- **Multi-probe crosstalk schematics** — 🔲 not built.

Two capabilities were added that this spec never anticipated: `pulse_snr()`
(`quantum/snr.py`), and `TabulatedModel` (`nonlinear/tabulated.py`), a
generic table-driven AM-AM/AM-PM model for devices whose response fits
neither the Saleh nor Volterra functional form.

---

## 14. Open Questions

1. **Volterra kernel parameterization from P1dB/IP3:** The diagonal-kernel approximation
   (Option C in §5.3) fits h₃ from P1dB and IP3 using the cubic kernel result. This
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

6. **Oscillator phase noise:** design settled in §7.4 (Monte Carlo over a per-realization
   nonlinear pass, rejecting a cheaper post-hoc-multiplicative approach as physically
   incomplete once real nonlinearity sits between the LO and the qubit plane), not yet
   implemented. Open sub-questions once picked up: what PSD parameterization to expose in
   the noise-node spec dict (power-law per IEEE 1139 vs. a tabulated `L(f)` curve, the
   latter matching how most real datasheets actually report it); whether the per-
   realization NL re-evaluation cost is acceptable at the default `n_realizations=100` or
   needs its own smaller default; and whether AWG/sample-clock timing jitter (a related
   but physically distinct effect — perturbs the envelope's own time base rather than the
   carrier phase) belongs in the same mechanism or needs separate treatment.

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
*Last updated: July 2026*
*Status: Pre-development / Seed document v0.10*
