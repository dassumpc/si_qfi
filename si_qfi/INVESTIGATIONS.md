# SI-QFI Investigations

A running log of physics questions answered using this codebase's own
SI → nonlinearity → QuTiP pipeline, each backed by a runnable demo script
(`examples/`), a regression test file (`tests/`) that locks the finding in,
and a generated figure. This document is intended as source material for a
report to be published alongside the codebase — each entry is written to
stand alone: motivation, method, result, and what would need to change to
extend it.

Conventions used throughout: a plain 2-level qubit (`quantum.QubitModel`,
no `Transmon`/leakage), resonant drive (`H0 = 0` in the rotating frame), a
100 ns Gaussian I-only pulse, and — for any investigation involving a
nonlinearity — self-calibration of the drive amplitude to hit exactly the
target rotation angle *through* the actual (possibly nonlinear) chain,
rather than assuming linear gain. This isolates genuine *distortion* from
simple *miscalibration*, which are different questions.

---

## 1. Rabi oscillation, no impairments (baseline)

**Script:** `examples/rabi_oscillation_demo.py` · **Tests:** `tests/test_quantum.py` · **Figure:** `examples/rabi_oscillation_demo.png`

**Motivation.** Before asking whether any impairment degrades gate
fidelity, first confirm the SI-QFI → QuTiP link itself is correct: does a
resonant π-pulse, propagated through a real (if trivial) SI schematic with
zero nonlinearity and zero noise, reproduce the textbook Rabi result and
reach fidelity arbitrarily close to unity?

**Method.** Sweep drive pulse area θ from 0 to 2.2π through
`tests/test_schematic_basic.si` (a lossless, flat-2.5×-gain, matched
drive line) in both `complex_baseband` and `real_axis` simulation modes,
solving the resulting qubit state with QuTiP at each point.

**Result.** Population traces the analytic sin²(θ/2) curve to within
simulation noise for both modes; average gate fidelity to the ideal X gate
peaks at θ=π. At the exact calibration point, **F = 0.9999993
(complex_baseband) and F = 1.0000000 (real_axis)** at a modest 2 GSa/s
envelope grid, improving to the float64 noise floor (~1e-12) at 8 GSa/s.
Both modes agree with each other and with the analytic prediction across
the whole sweep.

**A genuine solver bug this caught, not just a validation result:**
QuTiP's default adaptive ODE step size can silently step clean over an
entire drive pulse when the qubit is exactly resonant (no timescale to
size steps against), returning an output indistinguishable from the
identity — i.e. fidelity ≈ 1/3 to a nontrivial target — with **no error or
warning**. This did not appear on a hand-written smooth test array; only
the real, numerically-noisy simulated waveform triggered it. Fixed by
capping the integrator's `max_step`/`nsteps` in `quantum.gate_fidelity()`
to the drive's own sample spacing — see the "CRITICAL" comment there.

---

## 2. Does amplifier nonlinearity limit gate fidelity — AM-AM, AM-PM, or compression depth?

**Script:** `examples/nonlinearity_fidelity_demo.py` · **Tests:** `tests/test_quantum_nonlinear.py` · **Figure:** `examples/nonlinearity_fidelity_demo.png`

**Motivation.** A real drive amplifier has both AM-AM (gain compression)
and AM-PM (phase shift vs. drive amplitude). Which one actually limits
achievable gate fidelity, and does that depend on how close to the 1dB
compression point the amplifier is driven?

**Method.** Single amplifier stage (`DriverOutput` in
`test_schematic_basic.si`), self-calibrated to a π-pulse. Three sweeps:
(A) `complex_baseband`, Saleh AM-AM only, vs. compression severity
(`op1db_amplitude`); (B) `complex_baseband`, Saleh AM-AM + AM-PM, vs.
AM-PM peak phase at three compression depths; (C) `real_axis`, Saleh
real-axis and Volterra (both AM-AM only — see below), vs. compression
severity.

**Result.**

- **Pure AM-AM never limits fidelity, once recalibrated.** A memoryless
  real-gain nonlinearity multiplies a single-axis (I-only) drive by a
  real, non-negative scalar at every instant — it reshapes the *total*
  integrated rotation angle but can never rotate energy into the
  orthogonal (Q) axis. A 2-level qubit with no other timescale (no
  detuning, no leakage) only cares about that total angle. Panel A sits
  flat at the ~1e-11 float64 floor across the entire achievable
  compression range, in both baseband and real-axis mode (Panel C, for
  both the Saleh rational curve and the Volterra cubic — two different
  model shapes, same conclusion).
- **This has a hard limit, and it's a cliff, not a slope.** A Saleh/
  Volterra AM-AM curve's raw output eventually turns over (declines for
  large enough drive), so a given (compression severity, pulse shape,
  duration) combination has a maximum *achievable* total rotation angle.
  Below a critical `op1db_amplitude`, no drive amplitude — however large —
  completes a full π pulse. It is binary: exactly right, or flatly
  impossible with that pulse; there is no partial-degradation regime for
  AM-AM alone acting on a single stage.
- **AM-PM is the actual culprit.** Panel B shows infidelity growing
  smoothly and monotonically with AM-PM peak phase (up to ~7% at 30°),
  worse at deeper compression for the same nominal AM-PM spec (since the
  phase shift itself grows with drive amplitude). Unlike AM-AM,
  recalibrating the I-axis pulse area cannot undo this — AM-PM distorts
  the *axis* of rotation, not just its magnitude.
- **Scope limitation of this codebase:** AM-PM is only implemented for
  `SalehModel` in `complex_baseband` mode. Neither `SalehRealAxisModel`
  nor `VolterraModel` has any phase-distortion mechanism (both are
  memoryless real functions of the instantaneous waveform — there is no
  separate "envelope phase" to modulate). `real_axis` mode can therefore
  only ever show the AM-AM "recalibrate, or hard wall" pattern from this
  investigation; it cannot represent a real amplifier's own real-axis
  phase distortion if one is present. Complex-baseband mode with Saleh's
  AM-PM is the only tool in this codebase for that today.

**A numerical caveat worth carrying forward:** at a coarse 2 GSa/s
envelope sample rate, deep AM-AM compression alone (no AM-PM) produced a
spurious ~1e-6 "infidelity" that looked like real physics but wasn't — a
cubic-spline/ODE under-resolution artifact from the sharper, more
flat-topped pulse shape compression produces. It vanished to the float64
floor at 8 GSa/s. Every subsequent investigation in this document uses
8 GSa/s for this reason.

---

## 3. Two cascaded amplifiers, real-axis mode: does AM-AM alone limit fidelity via harmonic remixing?

**Script:** `examples/two_amp_harmonic_remixing_demo.py` · **Tests:** `tests/test_quantum_nonlinear.py` (two-amp section) · **Figure:** `examples/two_amp_harmonic_remixing_demo.png`

**Motivation.** Investigation 2 established that pure AM-AM never limits
fidelity for a *single* amplifier stage. But a real, odd, memoryless
nonlinearity driven by a narrowband signal generates a 3rd-harmonic image
near 3·f_carrier alongside its in-band distortion (the same mechanism
behind the 4/3 OIP3 factor documented in `nonlinear/saleh.py`). For one
amplifier, that image is far out of band and irrelevant. With a **second**
amplifier downstream, could that 3rd-harmonic content mix back down into
the qubit's own operating band?

**The mechanism, worked out explicitly:** expanding a cubic nonlinearity's
response to (near-f_c signal `x1` + near-3f_c image `x3`), the cross term
`3·x1²·x3` contains a component near `3f_c − 2f_c = f_c` — a genuine
third-order intermodulation product landing back in-band at the *second*
stage. This does not contradict the "cascaded odd nonlinearities driven by
a single CW tone only ever produce odd harmonics" proof used elsewhere in
this codebase (`tests/test_engine.py`'s two-amplifier cascade tests) —
`f_c` itself is the (odd, order-1) fundamental; that proof constrains which
harmonics can appear, not what physically contributes to the fundamental's
own in-band content.

**Method.** `tests/test_schematic_lossy_T_line_2_amplifier.si` (two
amplifiers, `DriverOutput` and `DriverOutput2`, with lossy/dispersive
transmission lines in between and after — the dispersion matters, since it
gives the fundamental and its 3rd-harmonic image different relative phase
by the time they reach the second stage). Same `op1db_amplitude` Saleh
AM-AM spec at both nodes (no AM-PM — real-axis has none, see Investigation
2), self-calibrated to a π pulse. Compared: single-stage (only
`DriverOutput2` nonlinear) vs. two-stage (both nonlinear), and
`real_axis` vs. `complex_baseband`, all through the *same* schematic.

**A methodological correction made during this investigation, kept here
deliberately because it's a real lesson, not just a fixed number:** the
first version of this demo used a 100 ns pulse and reported infidelity
"floors" around 1e-5 for op1db values well away from either cliff — the
same value in single- and two-stage, in both modes. That immediately
looked suspicious once questioned directly, and on inspection it wasn't
the effect under study at all: `engine.run()` with `nonlinear=None` on
this exact schematic gives the identical ~1e-5 infidelity. It's ordinary
linear dispersion from this schematic's lossy, frequency-dependent
transmission lines (its phase response isn't perfectly flat across the
pulse's own bandwidth) — confirmed by sweeping pulse duration from 100 ns
to 1600 ns and observing the "floor" shrink roughly with bandwidth²,
exactly as dispersion-limited distortion should. At 100 ns this baseline
sat uncomfortably close to the smallest genuine gray-zone values, which
would have made the two effects hard to tell apart. **Fixed by widening
the pulse to 400 ns** (¼ the bandwidth, baseline infidelity ~13× lower,
down to ~5×10⁻⁷) and by plotting the no-NL baseline explicitly as a
reference line on every panel below, rather than leaving a reader to
guess whether a given point is "floor" or "effect." The numbers below are
all from the corrected (400 ns) run; `tests/test_quantum_nonlinear.py`
locks in both the corrected op1db thresholds and a dedicated check that
the no-NL baseline itself stays under 1e-6.

**Result — the hypothesis is correct, in a more specific and more
interesting form than "cascading makes it worse":**

- **Cascading extends the achievable range.** Splitting a given total
  compression across two stages reaches a lower `op1db_amplitude` (deeper
  compression) than either stage could reach alone — true in *both*
  modes: baseband's achievability cliff moves from `op1db ≈ 0.25`
  (single-stage) to `≈ 0.18` (two-stage); real-axis's moves from `≈ 0.20`
  to `≈ 0.15`.
- **Real-axis reaches further than baseband, and only real-axis shows a
  genuine partial-infidelity gray zone.** Between real-axis's own
  single-stage cliff (`op1db ≈ 0.20`) and its two-stage cliff (`≈ 0.15`),
  the two-stage chain is achievable with **real, bounded, growing
  infidelity** — e.g. 1.4×10⁻⁵ at `op1db=0.17`, rising toward `≈3×10⁻⁵` at
  `0.15` — clearly above the ~5×10⁻⁷ no-NL baseline floor at this pulse
  duration, and a genuine gray zone that neither `complex_baseband`
  (structurally blind to harmonic content, at either stage count) nor
  `real_axis` single-stage (a clean binary cliff, per Investigation 2)
  ever shows. This gray zone *is* the harmonic-remixing signature: it
  exists precisely where (a) real RF harmonics exist to remix
  (`real_axis` mode) **and** (b) there is a second nonlinear stage
  downstream to do the remixing. An FFT of the qubit-plane waveform in
  this regime shows clearly visible 3rd-harmonic content at 3·f_c
  alongside the fundamental.
- **Where both are achievable at mild compression, real-axis is actually
  slightly *more* forgiving than baseband**, not less (infidelity
  5.3×10⁻⁷ vs. 5.9×10⁻⁷ at `op1db=0.5`, both already close to their
  respective no-NL baselines) — a small, secondary effect, not the
  headline result, but worth noting since a naive "harmonics can only
  hurt" prior would not have predicted it.

**Practical takeaway:** for a single amplifier, `complex_baseband` mode's
AM-AM treatment is not just convenient but *complete* — it will not miss
anything `real_axis` mode would show (Investigation 2). Once a design has
**two or more** cascaded nonlinear stages, `complex_baseband` mode can be
both *overly pessimistic* about the achievable operating range (its own
cliff comes sooner) and *silently blind* to a real, bounded infidelity
contribution in the newly-achievable-in-real_axis region — `real_axis`
mode is required to see it at all.

---

## 4. Pulse bandwidth vs. gate fidelity, from pure linear channel dispersion

**Script:** `examples/bandwidth_dispersion_fidelity_demo.py` · **Tests:** `tests/test_quantum_dispersion.py` · **Figure:** `examples/bandwidth_dispersion_fidelity_demo.png`

**Motivation.** A byproduct of Investigation 3: while chasing down the
"is the floor too high?" question there, the cause turned out to be
ordinary linear dispersion, not anything nonlinear. That effect is
interesting and useful on its own terms — it's the answer to "how wide a
drive pulse can I use before this specific drive line's own dispersion
costs me measurable gate fidelity" — so it's characterized here directly,
with **no nonlinearity anywhere in the chain** (`nonlinear=None`
throughout).

**Method.** Three schematics, same carrier/coupling/qubit as every other
investigation in this document: `test_schematic_basic.si` (lossless,
matched — a true zero-dispersion control), `test_schematic_lossy_T_line.si`
(one lossy transmission-line segment), and
`test_schematic_lossy_T_line_2_amplifier.si` (two lossy segments — see
Investigation 3). Self-calibrated π-pulse gate fidelity swept across pulse
duration 50–1600 ns (envelope bandwidth ~20 MHz down to ~0.6 MHz).

**Result.**

- **The lossless control stays at the float64 noise floor (~10⁻⁹ to
  10⁻¹⁵, no trend) at every bandwidth tested** — confirming this is
  genuinely a dispersion effect tied to the lossy lines specifically, not
  a generic artifact of self-calibrating a pulse through any schematic.
- **Both lossy schematics show infidelity scaling almost exactly as
  bandwidth² — fitted exponents 1.95 (one segment) and 1.92 (two
  segments)** — over more than a decade of bandwidth. This is the
  textbook scaling for a channel whose response isn't perfectly flat
  across the signal's own bandwidth.
- **Dispersion accumulates through additional lossy segments**: the
  two-segment schematic is consistently ~2.4–2.6× worse than the
  one-segment schematic at every bandwidth tested (not exactly 2× — the
  two segments aren't identical — but a stable multiplicative factor
  across the whole sweep).
- **Physical origin (Panel B): predominantly a gain TILT across the pulse
  band, not phase curvature.** Group delay measured essentially flat at
  this schematic's own 10 MHz frequency-sweep resolution, but |H(f)|
  shows a clear, close-to-linear slope near the carrier (~1–2%
  peak-to-peak differential attenuation over ±50 MHz for the one-segment
  case, more for two) — this particular SI transmission-line loss model
  (`ldbperhzpers`, loss scaling with frequency) predominantly tilts the
  passband rather than curving its phase. A reader modeling a different
  physical line (e.g. one dominated by genuine group-delay ripple instead)
  should expect this specific mechanism to differ, even if the
  bandwidth² scaling itself is generic.
- **Both simulation modes agree** (real_axis tracks complex_baseband to
  within ~10–30%, same order of magnitude and trend) — unlike
  Investigation 3's harmonic-remixing effect, ordinary linear dispersion
  is *not* real_axis-only: complex_baseband mode's own `H(f)`
  representation already carries in-band linear channel effects exactly,
  since there's no nonlinearity generating out-of-band content for
  baseband mode to miss.

**Practical takeaway:** before attributing any small infidelity floor to
a nonlinear or otherwise exotic effect, check the schematic's OWN
dispersion at the bandwidth in use — this is a real, quantifiable,
bandwidth² cost that exists independent of any amplifier nonlinearity,
and for a sufficiently wide pulse on a sufficiently lossy line it is not
negligible (it was mistaken for a nonlinear effect once already in this
document's own history — see Investigation 3's methodological note).

---

## 5. Impedance mismatch / reflections: when does it cost gate fidelity?

**Script:** `examples/impedance_mismatch_demo.py` · **Tests:** `tests/test_quantum_impedance_mismatch.py` · **Figure:** `examples/impedance_mismatch_demo.png` · **New schematic:** `tests/test_schematic_impedance_mismatch.si`

**Motivation.** A real drive line is never perfectly 50Ω end to end. Does
an impedance mismatch between the amplifier and the qubit — and the
reflections it causes — actually cost gate fidelity, or only under
specific conditions? Hypothesis going in: degradation should require
*both* a genuine impedance mismatch *and* a propagation delay long enough
that the reflection doesn't just look like part of the original pulse.

**A new capability needed for this one.** Every prior investigation swept
a *nonlinear model parameter* (Python-side), never a *schematic-level*
quantity — sweeping impedance and delay meant the SI schematic itself
needed to be parametrized and re-solved per sweep point. SignalIntegrity
already has a mechanism for this (`SignalIntegrityAppHeadless.
OpenProjectFile(filename, args={...})`, matched against a `<Variables>`
section declared in the .si file itself), verified directly against the
installed SI source before use — nothing needed adding to SI itself, only
`si_qfi.schematic.loader.load_schematic()` gained a new `variables=`
keyword threading straight through to it. See `loader.py`'s docstring for
the full contract.

**The new schematic**, based on `test_schematic_basic.si` (lossless,
single amplifier, so reflections are the *only* effect present — no
dispersion or loss to confound the picture): the amplifier's output
impedance and the VQubit-side termination are tied to one shared variable
`Zmismatch` (both ends of the line between them mismatched together, 50.0
reproducing the original matched baseline exactly), and the line's own
propagation delay is a second variable `Tprop`. Confirmed the resulting
transfer function shows the textbook reflection signature (frequency-
domain ripple) once mismatched, and is symmetric in `|Z-50|` around 50Ω as
physically required.

**A methodological requirement worth flagging for anyone extending this:**
SI's frequency sweep has a *discrete* grid (spacing `df = EndFrequency /
FrequencyPoints`), and the real-axis impulse response derived from it (an
IFFT) is only valid up to a time window of `1/df` — energy arriving later
wraps around and lands on top of the start of the window instead. A severe
mismatch's reflection ladder (successive round-trip bounces, amplitude
decaying by `Γ²` per bounce) can take many round trips — several
microseconds, for `Γ≈0.7` — to actually die out. This schematic's
`EndFrequency=7GHz`/`FrequencyPoints=32000` gives `df≈218.75kHz`, i.e. a
`1/df≈4.57µs` unambiguous window, comfortably past every bounce that
matters for the `Tprop` range swept below (max 100ns one-way). Verified
directly by inspecting the raw impulse response: with this resolution the
full multi-bounce ladder is captured at its correct, well-separated
arrival times (every ~400ns, decaying by a measured factor of ~0.51 per
bounce — matching `Γ²=0.714²=0.51` for the `Zmismatch=300` case almost
exactly) rather than folding onto the ~100ns pulse window.

**Result, Part 1 — Panels A-C, a coarse (whole-nanosecond) `Tprop` grid:**

- **Matched impedance (`Γ=0` exactly) costs nothing, at any delay** —
  infidelity stays at the numerical floor (~10⁻⁷) even at the longest
  delay tested. No reflection coefficient, no reflection, full stop.
- **Mismatch alone, at short delay, costs nothing either** (Panel B,
  green): even the most severe mismatch tested (300Ω, `Γ≈0.71`) stays
  within `[-8.9×10⁻⁷, 7.5×10⁻⁷]` — indistinguishable from the floor —
  when the round-trip delay (`2×Tprop`) is much shorter than the pulse
  duration. *Both* conditions are genuinely required.
- **Both together, on this grid, produce only a small effect** — Panel A
  spans `-8.9×10⁻⁷` to `8.3×10⁻⁶` over round-trip-delay/pulse-duration
  ratios 0.04 to 8; Panel B's long-delay curve spans `1.9×10⁻⁷` to
  `7.0×10⁻⁶` across `Γ=0` to `0.71`; Panel C's 2D sweep looks smooth and
  unremarkable throughout.

**This picture is an artifact of the sweep grid, not the physics — Part 2,
Panel D, is the actual headline finding.** Every `Tprop` value used above
(and in every regression test) is a whole number of nanoseconds. That is
not a neutral choice: the carrier is exactly 5GHz, so a whole-nanosecond
delay is *always* an exact integer number of carrier cycles. The reflected
echo's carrier phase on arrival is `2π×5GHz×Tprop`, which is therefore
*always* 0 (mod `2π`) at every single point Panels A-C sample — the safest
possible phase, by construction, every time.

Resolving `Tprop` at the sub-carrier-period scale (the carrier period is
`1/5GHz = 0.2ns`) at a fixed severe mismatch (`Zmismatch=300`, `Γ=0.71`)
shows what that coarse grid structurally cannot see: **infidelity swings
from the numerical floor (`~7×10⁻⁶`) up to `0.54` within picoseconds of
`Tprop`, periodic with the 0.2ns carrier period** (Panel D). Every
whole-nanosecond `Tprop` sits exactly in one of the narrow safe notches;
a cable a fraction of a millimeter different in length would not.

**Is this a calibration gap or genuine distortion?** Tested directly, not
assumed. The standard calibration (`tuneup_amplitude`, used everywhere
above) picks a single *real* amplitude so the total X-quadrature
(in-phase) rotation area hits `π`. A strictly stronger calibration was
also tried: a *complex* (amplitude **and** phase) launch scale, chosen so
the total `(X, Y)` rotation area lands exactly on `(π, 0)` — verified to
land there to machine precision. If the sub-carrier-period sensitivity
were just "wrong total area," this would cure it. **It does not** — Panel
D's complex-scale curve tracks the real-scale curve within about a factor
of 2 almost everywhere, including the same catastrophic peaks. The cause
is that the reflected echo, at comparable amplitude to the direct pulse
and overlapping its tail, makes the *instantaneous* drive axis wobble
between X and Y during that overlap — rotations about different axes at
different instants don't commute, so no single global rescale of a fixed
pulse shape (real or complex) can undo a time-dependent axis wobble, only
correct its net integrated total. This is genuine waveform distortion, in
the same family as the nonlinearity-driven distortion elsewhere in this
series — not a miscalibration that a better calibration routine would
erase.

**Practical takeaway:** whether a given impedance mismatch is dangerous
depends on the round-trip delay's *phase relative to the carrier* at a
sub-picosecond (sub-millimeter cable-length) precision that essentially no
real setup controls or even measures — a comparably-sized reflection can
be almost free or can badly scramble a gate depending on a cable length
difference far below any practical tolerance. Critically, this cannot be
fixed by a better amplitude or amplitude+phase calibration — the standard
kind of calibration any real setup does — since the damage is in the
pulse's shape, not its net rotation. A drive-line reflection with a
comparable-amplitude echo overlapping the pulse should be treated as a
potentially severe, effectively uncontrolled hazard, not the "forgiving,
sub-percent impairment" that a coarse (or unlucky) parameter sweep would
suggest.

---

## 6. Transmon leakage: does a real (anharmonic) qubit model change the fidelity story?

**Script:** `examples/transmon_leakage_demo.py` · **Tests:** `tests/test_quantum_transmon_leakage.py` · **Figure:** `examples/transmon_leakage_demo.png`

**Motivation.** Every investigation so far (1–5) used an idealized 2-level
qubit (`H0=0`, exactly resonant, no third level to leak into) — a
deliberate scope boundary, flagged as an open question after Investigation
5. Does swapping in a real `quantum.Transmon` (finite anharmonicity,
`n_levels=3`) change anything, independent of any drive-chain impairment?
No NL, no noise, no channel impairment anywhere in this demo — the
schematic (`test_schematic_basic.si`) is lossless and perfectly matched, so
any infidelity here comes purely from the qubit Hamiltonian, not the drive
chain.

**Trap #1 — which frame is `Transmon.build_H0()` in?** `Transmon.
build_H0()` returns the *lab-frame* Hamiltonian (`omega_q·num + (alpha/2)·
a†a†aa`), but `quantum.build_hamiltonian()` (used by every `gate_fidelity()`
call) assumes `H0` is already expressed in the frame *rotating* at the
drive carrier (its own docstring: "H0=0 for an exactly-resonant drive").
Naively calling `Transmon(...).as_qubit_model()` would add the qubit's full
~2π·5GHz precession term on top of a drive built for a rotating frame —
nonsense, not leakage. Fixed by constructing the rotating-frame model
directly (the standard DRAG-paper starting point, Motzoi et al. PRL 103,
110501 (2009)): at exact resonance the precession term cancels exactly,
leaving only `H0' = (alpha/2)·a†a†aa`.

**Trap #2 — average GATE fidelity is the wrong metric here.** The first
version of this demo used `gate_fidelity(ideal_gate="X")` on the full
3-level propagator and got nonsense: infidelity ~0.3–0.7 that barely
improved even 100× past the leakage timescale, where a direct state check
showed population transfer already >99.99% complete with ~0 population in
`|2⟩`. The propagator itself, inspected directly, was essentially perfect
(`-i·σx` on `{|0⟩,|1⟩}` plus a near-unit-magnitude phase on `|2⟩`). The
cause: that `|2⟩` phase is *unpopulated-level free evolution* under `H0'`
— physically unobservable for a qubit that starts and ends in `{|0⟩,|1⟩}`
— but `qt.average_gate_fidelity()` penalizes it anyway, because it's a
*relative* phase between the `{0,1}` block and the `|2⟩` block, and that
metric is only phase-invariant when the *same* global phase applies to the
*whole* Hilbert space (true for `n_levels=2`, not once a differently-
evolving third level exists). Fixed by switching to `gate_fidelity(
target_state=...)` — state fidelity from `|0⟩` to target `|1⟩`, which is
insensitive to this artifact — plus reading raw leakage population directly
off `FidelityResult.final_states()`. A permanent caution about this now
lives in `gate_fidelity()`'s own docstring.

**Result:**

- **Panel A.** At fixed anharmonicity (`alpha = -200 MHz`, typical for a
  transmon), the 3-level model shows real leakage-driven state infidelity
  that grows sharply as the pulse shortens relative to `1/|alpha|` —
  `2.9×10⁻⁵` at 320ns down to (i.e. up to) `0.29` at 5ns — while the
  idealized 2-level model stays at the numerical floor (`<10⁻⁹`) throughout,
  since it has no third level to leak into. This is the first investigation
  in the series where the *qubit model itself*, not the drive chain, is
  what limits fidelity.
- **Panel B.** DRAG (`build_drag_envelope()`) suppresses the actual leakage
  *population* into `|2⟩` dramatically — measured up to ~950× at 40ns, and
  consistently >7× even at the shortest (5ns) duration tested — confirming
  the derivative-removal correction genuinely works in this codebase's own
  solver, not just in the docstring's claim. But overall *state infidelity*
  improves only modestly at short duration (e.g. 22% vs. 29% at 5ns) —
  because this codebase's `build_drag_envelope()` implements only the
  leading-order I/Q-quadrature correction, with no companion frequency-
  detuning term, so it leaves behind the AC-Stark-shift-driven rotation-
  angle error that full DRAG implementations correct separately. **"DRAG
  suppresses leakage" and "DRAG fixes the gate" are not quite the same
  claim** with this codebase's current DRAG implementation.

**Practical takeaway:** an idealized 2-level qubit model is a genuinely
different physical assumption from a real transmon, not just a
simplification of degree — it hides an entire class of fast-pulse error
(leakage) that a real device experiences. Fast pulses (approaching
`1/|alpha|`) need either enough margin from the anharmonicity or an
explicit leakage-suppression scheme; a simple single-quadrature DRAG
correction is a genuine, measurable, but partial fix.

---

## 7. T1/T2 decoherence: how much does intrinsic qubit decoherence cost gate fidelity vs. gate time?

**Script:** `examples/t1_t2_decoherence_demo.py` · **Tests:** `tests/test_quantum_t1_t2.py` · **Figure:** `examples/t1_t2_decoherence_demo.png`

**Motivation.** `gate_fidelity()` has supported intrinsic T1/T2 (Lindblad
collapse operators, `T1_us`/`T2_us`) since the QuTiP backend was first
wired up, but it had only ever been exercised by a single unit test —
never its own investigation. How much does a qubit's *own* decoherence cost
gate fidelity, as a function of gate time, independent of any drive-chain
impairment? Stays on the idealized 2-level qubit (T2's `qt.sigmaz()`
collapse operator is only meaningful for `n_levels=2`, see
`gate_fidelity()`'s docstring — combining this with Investigation 6's
leakage physics needs a generalized dephasing operator this codebase
doesn't have yet). `T2_us` here follows this codebase's existing
convention of deriving a pure-dephasing rate as `1/Tφ = 1/T2 - 1/(2·T1)` —
i.e. it means the same thing as a measured "T2*" (free-induction-decay)
time in an experiment, not a Hahn-echo T2.

**A real trap, found and fixed while building this.** The first version of
this demo swept nominal pulse duration and plotted infidelity vs.
`duration/T1`, expecting curves at different T1 to collapse onto one line.
They didn't — the effective per-ns cost dropped by ~100× between short and
long nominal durations, for *every* T1 tested. Cause: `result.
v_qubit_ensemble`'s array is *longer* than the nominal pulse — convolving
the drive envelope with `test_schematic_basic.si`'s own impulse response
pads it with a roughly *fixed* ~99ns tail (confirmed directly: `n_out -
n_in` corresponds to ~99ns essentially independent of input pulse length),
because that tail's length is set by the schematic's own frequency-domain
resolution, not by the drive pulse. Harmless for closed-system fidelity (no
drive there, so the propagator segment over it is ~identity) — but
`gate_fidelity()` uses the *full* array length as `T_gate` for the Lindblad
solve, so with `T1_us`/`T2_us` given, decoherence keeps acting through that
extra ~99ns of essentially-zero-drive tail, inflating reported infidelity
by an amount that has nothing to do with the nominal gate. Fixed by
measuring and plotting against the *actual* simulated gate time
(`len(result.v_qubit_ensemble[0]) / result.fs`), not the nominal requested
duration. A permanent caution about this now lives in `gate_fidelity()`'s
own `T1_us`/`T2_us` docstring.

**Result:**

- **Panel A (T1-limited, `T2=2·T1`).** Once corrected to use the *true*
  gate time, `infidelity / (T_gate_true/T1)` is close to a single constant
  (~0.33) across T1 ∈ {10, 40, 100} µs and a ~400× range of gate times —
  confirming clean, expected linear-in-time perturbative decoherence
  physics, with curvature appearing only at the largest ratio tested
  (`T_gate/T1 ≈ 0.4`, entering the non-perturbative regime as expected).
  Using nominal duration instead of true gate time, by contrast, gives
  coefficients differing by >12× across the same duration range — the
  negative control confirming the trap was real.
- **Panel B (fixed gate time and T1, T2 swept).** Pure dephasing beyond T1
  can dominate total decoherence-driven infidelity even at a long, "good"
  T1: at fixed `T1=30µs` and `gate_time≈300ns`, infidelity climbs
  monotonically from `3.3×10⁻³` (`T2=2·T1`, T1-limited, no extra dephasing)
  to `0.164` (`T2≈0.94µs`, heavy extra dephasing) — a ~50× range from T2
  alone, T1 held fixed throughout.

**Practical takeaway:** decoherence-driven infidelity budgeting genuinely
needs the *true* simulated gate-time window, not the nominal pulse
duration you asked for — any schematic with delay/dispersion pads the
propagation window, and that padding is invisible in a closed-system
(no-T1/T2) result but directly inflates any T1/T2-inclusive fidelity
number. Separately: matches real experimental experience that T2* is often
the practically-limiting number even when T1 looks fine.

---

## Open questions for future investigations

- Investigation 6 answered the "does AM-AM's result hold for a multi-level
  qubit" question from Investigation 2/3's scope boundary: yes for total
  pulse *area* determining rotation, but a genuinely new, separate leakage
  channel opens up that 2-level analysis can't see at all. Follow-on: does
  full (I/Q + detuning) DRAG close the residual state-infidelity gap this
  investigation found for the simple I/Q-only correction?
- Investigation 6/7 both used the idealized 2-level or 3-level *closed vs.
  open* system in isolation — does leakage (Investigation 6) interact with
  intrinsic decoherence (Investigation 7) when combined (e.g. T1 decay
  competing with a fast, leakage-heavy pulse)? Not yet tested — needs a
  generalized T2 dephasing operator for `n_levels > 2` first (`gate_
  fidelity()`'s T2 branch is `n_levels=2`-only today).
- Noise (the noise/ subpackage: PSD, stochastic realizations,
  NoisePropagator) is fully implemented and unit-tested but has never been
  exercised by an investigation — how does drive-chain-injected noise
  (amplifier noise figure, etc.) compare in scale to the intrinsic T1/T2
  decoherence characterized in Investigation 7?
- Does the two-amplifier gray zone (Investigation 3) grow or shrink with
  the inter-stage channel's dispersion (i.e. is a *less* dispersive line
  between amplifiers more forgiving, or does any nonzero delay/dispersion
  suffice)? Not yet swept.
- AM-PM's real-axis blind spot (Investigation 2): if a real-axis phase-
  distortion mechanism is added to this codebase in the future, does it
  interact with the harmonic-remixing effect from Investigation 3, or are
  they independent contributions?
- Investigation 4 found the dominant distortion mechanism for THIS
  codebase's lossy transmission-line model is a gain tilt, not group-delay
  curvature — is that a property of the specific `ldbperhzpers` loss model
  used in these test schematics, or would a different SI line model (e.g.
  one with explicit dispersion/velocity-vs-frequency behavior) show the
  bandwidth² scaling dominated by phase curvature instead? Not yet tested.
- Investigation 5 only mismatches ONE segment (amplifier output <-> qubit
  termination), with the amplifier's own input side, and both source-side
  impedances, left fixed and matched. Does a mismatch on the DRIVE side
  (before the amplifier) show the same round-trip-delay-vs-pulse-duration
  crossover, or does the amplifier's own gain/isolation change the
  picture (e.g. by attenuating a reflection headed back toward the
  source before it can reflect again)? Not yet tested. Also: this
  investigation used `complex_baseband` mode only (reflections are linear,
  already established mode-independent by Investigation 4) — an explicit
  `real_axis` cross-check for THIS specific schematic/effect was not run.
