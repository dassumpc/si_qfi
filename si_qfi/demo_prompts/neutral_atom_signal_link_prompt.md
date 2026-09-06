# Demo brief: agent-designed signal link for a neutral-atom/trapped-ion Raman gate

You are starting a **new, standalone git repository** (not a fork or subdirectory
of `si_qfi`). Its purpose is to demonstrate that an AI agent, given a physics
toolkit and a target, can autonomously scope a real RF/optical drive-chain
design, select representative components for it, simulate the chain, and
report the gate fidelity it expects to achieve — a "signal link design" demo,
not a one-off calculation.

## What `si_qfi` is and how to get it

`si_qfi` is a Python package that bridges **SignalIntegrity** (a classical
RF/microwave circuit simulator) with **QuTiP** (a quantum dynamics library).
You describe a drive chain — amplifiers, transmission lines, noise sources,
mismatches — as an ordinary SignalIntegrity `.si` schematic. `si_qfi` extracts
real transfer functions from that schematic, propagates a control pulse
through it (applying whatever nonlinearity and noise you've annotated), and
hands the resulting waveform to QuTiP to compute gate/state fidelity. It
supports two propagation modes (`complex_baseband`, the efficient default;
`real_axis`, exact but slower), several nonlinearity models (Saleh AM-AM/AM-PM,
Volterra, and a generic interpolated-table model — see below), three
independent noise mechanisms (additive drive-line noise, multiplicative
LO/oscillator phase noise, and intrinsic T1/T2 decoherence), and calibration
helpers (`tuneup_amplitude()`) so you don't have to hand-tune pulse amplitudes.

Install it into this new repo with:

```bash
pip install "<FILL IN: si_qfi source location — local path or GitHub URL>[quantum,si]"
```

Both backends are optional extras in si_qfi's packaging, so you need the
`[quantum,si]` extras (QuTiP + SignalIntegrity) for anything in this brief.

Read `si_qfi`'s own `README.md` first — it sits at that repo's **root** and
gives the package structure, feature list, and a runnable quick-usage
example. It is the authoritative reference for exact API signatures; treat
anything below as a design brief, not a spec, and verify call signatures
against the installed package before relying on them. Also skim
`si_qfi/INVESTIGATIONS.md` (note: inside the package directory, not the repo
root) — it documents real physics findings, and real bugs/gotchas, from using
this exact toolkit, several of which are directly relevant below.

## The physical picture

A neutral-atom or trapped-ion qubit is not driven directly by an RF/microwave
line the way a superconducting qubit is. Instead, the standard architecture
for a **Raman-driven hyperfine qubit gate** is:

```
RF synthesizer/AWG -> RF driver amplifier -> coax -> AOM (RF port)
                                                       |
                                                  (diffracted light,
                                                   frequency-shifted by
                                                   the RF drive)
                                                       |
                                                     atom/ion
```

An acousto-optic modulator (AOM) diffracts an incident laser beam using an
RF-driven acoustic wave; the diffracted beam's amplitude and phase track the
RF drive envelope. The diffracted light then drives the atom's hyperfine
transition (directly, or via a second AOM-driven beam for a two-photon Raman
scheme — see "Two-photon Raman" below).

## Where `si_qfi` stops, and why that's the right boundary

`si_qfi` models the **electrical/RF chain** faithfully: the synthesizer's
envelope, driver-amplifier compression/noise, coax dispersion and mismatch,
and — the point that makes this modality tractable — **the AOM's own
RF-power-to-diffracted-amplitude response**, using the generic model
`TabulatedModel` (registry string `"table"`). Everything *downstream* of the
diffracted light (free-space propagation, the atom-light dipole coupling
itself) is **not** modeled by `si_qfi` — it's absorbed into a single
Hamiltonian coupling constant, exactly the same abstraction
`coupling_strength_per_volt` already provides for superconducting qubits
(there too, the actual capacitive/inductive coupling geometry is not modeled
from first principles — it's a single externally-supplied number). Treat the
final "effective drive amplitude" coming out of `si_qfi`'s chain as directly
proportional to the Rabi rate driving the atom's qubit transition, scaled by
whatever `coupling_strength_per_volt` you choose. This is not a shortcut
specific to this demo — it is how `si_qfi`'s Hamiltonian-building boundary
already works for every modality it supports.

### The AOM nonlinearity, specifically

An AOM's diffraction efficiency (Bragg regime) follows roughly
`A_out = A_max * sin(kappa * sqrt(P_RF))`, where `P_RF` is the RF drive
power. Since `si_qfi` propagates a *voltage* waveform and `P_RF ∝ V^2`, this
becomes `A_out(V) = A_max * sin(kappa' * V)` for some effective `kappa'` —
sinusoidal and genuinely **non-monotonic** past the first diffraction maximum
(unlike Saleh's bounded-rational curve or a Volterra polynomial, which are
the wrong shape for this). `si_qfi`'s `nonlinear.TabulatedModel`
(`nonlinear={"NODE": {"model": "table", "amplitude": [...],
"output_amplitude": [...]}}`) exists for exactly this: it interpolates a
caller-supplied amplitude-in -> amplitude-out table, with no assumption of
monotonicity. Build a small table (a few dozen points) sampling
`A_max * sin(kappa' * V)` over `V` from 0 up to comfortably past your intended
operating point, and annotate the AOM's RF-input node with it. Note the table
must start at exactly `(0.0, 0.0)` and its `amplitude` column must be
strictly ascending. Pick `kappa'` so your nominal operating drive lands well
below the first turnover (i.e. on the efficiency curve's rising,
near-monotonic part) — driving into the turnover region is a legitimate thing
to *show* (e.g. as a "don't overdrive this AOM" finding) but not something
you want as your baseline operating point.

### Two-photon Raman (optional — judge whether it's worth the complexity)

If you want a genuine two-photon Raman scheme (two AOM-driven beams, Rabi
rate proportional to the *product* of both diffracted field amplitudes,
divided by a large intermediate-state detuning `Delta`): run `si_qfi` once
per AOM/RF chain (two separate `siq.run()` calls, two separate schematics or
variable overrides), multiply the two resulting envelopes together (with the
`1/Delta` factor) **outside** `si_qfi`, and feed the single combined envelope
into the Hamiltonian-building step directly rather than through the
automatic `gate_fidelity()` pipeline. You'll likely need to construct a
`SimulationResult` by hand for the combined envelope — `si_qfi`'s own
`si_qfi/examples/phase_noise_case_study_demo.py` has a helper
(`_posthoc_phase_noise_result()`) that does something structurally similar
(building a `SimulationResult` by hand to feed into `gate_fidelity()`); use
it as a pattern, not a literal template. **This adds real complexity for a
demo whose point is the RF/EE chain, not Raman physics** — a single-AOM,
single-photon-equivalent drive is a perfectly good, simpler fallback if the
two-photon glue code isn't paying for itself. Your call.

Worth knowing if you go this route: real experiments (e.g. recent
Harvard/QuEra neutral-atom work) often generate the two Raman tones with an
**EOM** (electro-optic phase modulator) driven near the hyperfine splitting,
rather than two independent AOM chains — specifically because both tones then
share one optical carrier, so laser phase noise is common-mode and cancels to
first order. `si_qfi` does **not** model EOM sideband generation (it's a
multi-tone spectral operation, not an amplitude-domain nonlinearity, and sits
outside the narrowband envelope picture `complex_baseband` assumes). If you
model an EOM-based architecture, say plainly in the report that the tone
generation itself is assumed rather than simulated.

## Explicitly out of scope

Do not attempt to model, and do not report fidelity numbers that implicitly
assume you've captured: AC Stark shifts, off-resonant scattering from the
intermediate state, Rydberg-state decay, or motional/Doppler dephasing. These
are real error sources for some neutral-atom/ion gates (especially Rydberg
gates, where they typically *dominate* the error budget) but they live
entirely outside the RF/electrical domain `si_qfi` models — pulling them in
would require hand-added QuTiP Lindblad/Hamiltonian terms with no
`si_qfi` machinery behind them, and would muddy the demo's actual point
(that an agent can design and simulate the *drive chain*). Scoping to a
Raman-driven hyperfine qubit gate specifically (not Rydberg) keeps these
effects small enough to legitimately ignore.

## What to actually build

1. **Scope and select parts** for the electrical chain: an RF
   synthesizer/AWG, an RF driver amplifier (with realistic OP1dB/OIP3/noise
   figure), coax/transmission line, and the AOM itself (RF input impedance,
   drive power range, diffraction efficiency curve). You do not need live
   web access — representative, plausible datasheet-range numbers are fine,
   clearly labeled as illustrative unless you actually have real datasheet
   access. Write down *why* each part was chosen (this reasoning is a
   deliverable, not just the final numbers).
2. **Build the `.si` schematic** (or generate one programmatically) wiring
   these parts together, with the AOM's RF input annotated with a
   `TabulatedModel` ("table") nonlinearity as described above, and realistic
   noise sources (thermal noise on lossy elements at their physical
   temperature; synthesizer phase noise via `phase_noise=` if you want to
   show its effect on gate fidelity).
3. **Calibrate a pulse** (Gaussian or DRAG-shaped envelope) via
   `tuneup_amplitude()` to hit a target single-qubit gate through the actual
   (nonlinear) chain — do not hand-pick an amplitude.

   > **Known bug to watch for**: `tuneup_amplitude()`'s calibration search has
   > a documented robustness gap for a band of intermediate `op1db_amplitude`
   > values, where it can land on a spurious, wildly-wrong solution (scale
   > ~100x too large) while still reporting `achieved=True`. Sanity-check the
   > calibrated scale against the pulse's own expected order of magnitude
   > before trusting it; if it looks wrong, perturb your part choice (e.g.
   > the driver amplifier's OP1dB) and retry rather than silently accepting
   > a bad calibration.

4. **Run the full simulation** (`siq.run()` + `quantum.gate_fidelity()`) with
   noise enabled, and report `.noise_free` vs. `.noise` fidelity separately
   so the drive-chain-noise contribution is visible on its own.
5. **Write a report** (this repo's own `README.md` is fine) presenting: the
   chosen parts and why, the resulting noise budget, the calibrated pulse,
   and the predicted gate fidelity — the actual deliverable of this demo.

## A note on honesty

If something doesn't work as expected — a calibration search fails, a
nonlinearity model doesn't behave the way the physics should, a number looks
suspicious — report that directly rather than smoothing it over or quietly
picking parameters that avoid the problem. `si_qfi`'s own `INVESTIGATIONS.md`
documents several real dead ends and bugs found this way (a saturating
amplifier masking a real effect, a non-convergent test design, the
`tuneup_amplitude()` bug mentioned above) — that kind of honest reporting is
part of what makes this demo credible, not something to avoid.

## A minimal code skeleton to start from (verify against the installed API)

```python
import qutip
import si_qfi as siq
from si_qfi.source.waveform import build_gaussian_envelope, source_from_envelope_array

schematic = siq.load_schematic("chain.si")   # your generated/authored schematic
qubit = siq.quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

# coupling_strength_per_volt here stands in for the AOM's optical output ->
# Rabi-rate conversion, exactly as it stands in for capacitive coupling in
# the superconducting case -- pick an illustrative value consistent with a
# realistic hyperfine Raman gate speed (e.g. a few hundred kHz to a few MHz
# Rabi rate) and justify it in your report.
eta = 2 * 3.14159265 * 1e6

ref_shape = build_gaussian_envelope(duration_s=..., sigma_s=..., sample_rate_hz=..., amp=1.0)
tuned = siq.quantum.tuneup_amplitude(
    schematic, ref_shape, fs_envelope=..., carrier_ghz=...,   # RF carrier driving the AOM
    qubit=qubit, coupling_strength_per_volt=eta, ideal_gate="X",
)

source = source_from_envelope_array(tuned.scale * ref_shape, fs=..., carrier_ghz=...)
result = siq.run(
    schematic=schematic, source=source, nonlinear=None,   # nonlinearity comes from the schematic annotation
    noise={...}, n_realizations=100, mode="complex_baseband", seed=42,
)
fid = siq.quantum.gate_fidelity(result, qubit, coupling_strength_per_volt=eta, ideal_gate="X")
print(fid.noise_free.F_avg, fid.noise.F_avg, fid.noise.F_sem)
```

Keep the drive's bandwidth/carrier ratio under ~0.05 or `siq.run()` will warn
that `complex_baseband` may be inaccurate — see the narrowband-ratio
diagnostic in si_qfi's README.
