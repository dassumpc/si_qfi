# Demo brief: agent-designed signal link for a superconducting-qubit XY control line

You are starting a **new, standalone git repository** (not a fork or
subdirectory of `si_qfi`). Its purpose is to demonstrate that an AI agent,
given a physics toolkit and a target, can autonomously scope a real
room-temperature-to-cryostat drive-chain design, select representative
components for it, simulate the chain, and report the gate fidelity it
expects to achieve — a "signal link design" demo, not a one-off calculation.

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
Volterra, and a generic interpolated-table model), three independent noise
mechanisms (additive drive-line noise, multiplicative LO/oscillator phase
noise, and intrinsic T1/T2 decoherence), and calibration helpers
(`tuneup_amplitude()`) so you don't have to hand-tune pulse amplitudes.

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

A transmon's XY control line runs from room-temperature electronics down
through a dilution refrigerator to the chip, with attenuation stages at each
physical temperature stage (to break the thermal-noise path from the warm
electronics down to the ~10 mK stage the qubit lives at):

```
RT synthesizer/AWG -> RT driver amplifier -> coax
   -> 4K stage attenuator -> still stage attenuator
   -> mixing-chamber stage attenuator -> qubit XY line
```

Each attenuator is a *lossy* element at a *specific physical temperature* —
each one is a genuine Johnson-noise source in its own right (this is exactly
what `si_qfi`'s `type="thermal"` noise override is for: it takes a physical
temperature directly, not just an abstract PSD number), and the whole chain's
composite noise figure/temperature is what ultimately sets the drive-line
noise floor reaching the qubit.

## Where `si_qfi`'s modeling boundary is

`si_qfi` models the **entire electrical chain shown above**, faithfully,
node by node — this modality does not need any new physics beyond what
already exists in the package (unlike the neutral-atom/trapped-ion case,
which needed a generic tabulated nonlinearity for the AOM boundary). The one
thing still external to `si_qfi` is the **qubit-coupling geometry** itself —
turning the drive voltage at the chip into an actual Rabi rate depends on
the qubit's capacitive/inductive coupling to its drive line, which is not
computed from first principles here. That's exactly what
`coupling_strength_per_volt` is for: pick a physically reasonable value (a
typical transmon XY coupling gives Rabi rates in the tens-of-MHz range for
drive amplitudes of order 1V at the chip) and state your reasoning in the
report.

## What to actually build

1. **Scope and select parts** for the full chain: an RT synthesizer/AWG
   (with realistic phase-noise datasheet figures, `dBc/Hz` vs. offset
   frequency), an RT driver amplifier (OP1dB/OIP3), coax, and 2-3 discrete
   cryostat attenuator stages with realistic attenuation values and physical
   temperatures (e.g. ~4K, ~0.1K/still, ~0.01K/mixing-chamber — pick values
   consistent with a real dilution-refrigerator wiring diagram). You do not
   need live web access — representative, plausible datasheet-range numbers
   are fine, clearly labeled as illustrative unless you actually have real
   datasheet access. Write down *why* each part and its attenuation/
   placement was chosen (this reasoning is a deliverable, not just the final
   numbers) — e.g. trading off more attenuation (better thermal isolation,
   worse SNR at the qubit) against less.
2. **Build the `.si` schematic** wiring these stages together, each
   attenuator annotated with `noise={"NODE": {"type": "thermal",
   "temperature_k": ...}}` at its own physical stage temperature, and the
   driver amplifier annotated with a Saleh or Volterra nonlinearity built
   from its OP1dB/OIP3.
3. **Calibrate a DRAG-shaped pulse** (`build_drag_envelope()`) via
   `tuneup_amplitude()` to hit a target single-qubit gate (e.g. `"X"`)
   through the actual (nonlinear) chain — do not hand-pick an amplitude.
   DRAG matters here specifically because a transmon has real leakage to its
   second excited state; a bare Gaussian would show that leakage as fidelity
   loss that DRAG is designed to suppress — this is a real, demonstrable
   effect worth showing in the report, not just using DRAG as a default.

   > **Known bug to watch for**: `tuneup_amplitude()`'s calibration search has
   > a documented robustness gap for a band of intermediate `op1db_amplitude`
   > values, where it can land on a spurious, wildly-wrong solution (scale
   > ~100x too large) while still reporting `achieved=True`. Sanity-check the
   > calibrated scale against the pulse's own expected order of magnitude
   > before trusting it; if it looks wrong, perturb your part choice (e.g.
   > the driver amplifier's OP1dB) and retry rather than silently accepting
   > a bad calibration.

4. **Add the synthesizer's phase noise** via `phase_noise={"dbc_hz": ...,
   "bandwidth_hz": ...}` and show its effect on gate fidelity alongside the
   additive thermal-noise-only case, so the two noise mechanisms' relative
   contributions are visible separately (`si_qfi`'s `INVESTIGATIONS.md`
   Investigation 10 has the relevant background on why these two mechanisms
   are architecturally distinct and are injected differently). Note
   `bandwidth_hz` is required — there is no default.
5. **Run the full simulation** (`siq.run()` + `quantum.gate_fidelity()`,
   optionally with `T1_us`/`T2_us` set to add intrinsic qubit decoherence on
   top of drive-chain noise) and report `.noise_free` vs. `.noise` fidelity
   separately.
6. **Build a small cost-vs-fidelity comparison**: pick 2-3 alternative part
   choices (e.g. a cheaper vs. a lower-noise-figure driver amplifier, or a
   different attenuation budget across the cryostat stages) and compare
   their resulting fidelity and rough relative cost — a small table or plot
   is enough. This is the "design tradeoff" part of the story, not just a
   single fixed answer.
7. **Write a report** (this repo's own `README.md` is fine) presenting: the
   chosen parts and why, the resulting noise budget, the DRAG-vs-Gaussian
   leakage comparison, the cost/fidelity tradeoff, and the predicted gate
   fidelity for your final chosen design.

## Explicitly out of scope

`si_qfi` currently has no way to inject σz (dephasing-axis) noise directly —
only same-axis (drive-line) noise and, separately, `T1_us`/`T2_us` as an
aggregate intrinsic-decoherence knob. If you want to represent flux noise or
another dephasing-specific channel, use `T2_us` as the aggregate stand-in
rather than trying to build a σz-noise mechanism that doesn't exist yet —
note this limitation explicitly in your report rather than working around it
silently. A two-qubit/flux-line extension is a reasonable stretch goal but
is out of scope for the base demo for the same reason.

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
from si_qfi.source.waveform import build_drag_envelope, source_from_envelope_array

schematic = siq.load_schematic("chain.si")   # your generated/authored schematic
qubit = siq.quantum.Transmon(Ej_GHz=..., Ec_MHz=..., n_levels=...)   # see si_qfi/quantum/README.md

eta = 2 * 3.14159265 * 20e6   # rad/(s*V) -- illustrative XY coupling, justify in your report

ref_shape = build_drag_envelope(duration_s=..., sigma_s=..., sample_rate_hz=..., amp=1.0, anharmonicity_hz=...)
tuned = siq.quantum.tuneup_amplitude(
    schematic, ref_shape, fs_envelope=..., carrier_ghz=...,
    qubit=qubit, coupling_strength_per_volt=eta, ideal_gate="X",
)

source = source_from_envelope_array(tuned.scale * ref_shape, fs=..., carrier_ghz=...)
result = siq.run(
    schematic=schematic, source=source, nonlinear=None,   # nonlinearity comes from the schematic annotation
    noise={...}, phase_noise={...}, n_realizations=100, mode="complex_baseband", seed=42,
)
fid = siq.quantum.gate_fidelity(
    result, qubit, coupling_strength_per_volt=eta, ideal_gate="X",
    T1_us=..., T2_us=...,
)
print(fid.noise_free.F_avg, fid.noise.F_avg, fid.noise.F_sem)
```

Keep the drive's bandwidth/carrier ratio under ~0.05 or `siq.run()` will warn
that `complex_baseband` may be inaccurate — see the narrowband-ratio
diagnostic in si_qfi's README. Note that enabling `phase_noise=` makes the
engine re-run the nonlinear pass once per Monte Carlo realization, so it is
substantially slower than additive noise alone; start with a small
`n_realizations` while iterating.
