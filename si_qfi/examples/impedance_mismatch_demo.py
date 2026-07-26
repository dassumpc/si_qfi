"""
examples/impedance_mismatch_demo.py
====================================
Investigation: when does an impedance mismatch (amplifier output impedance
and/or qubit-plane termination not exactly 50 ohm) between the amplifier
and the qubit actually cost gate fidelity, via reflections?

Hypothesis under test (as posed): degradation should only show up when
BOTH (a) the impedance is far from 50 ohm AND (b) the propagation delay
between the mismatched points is long enough to matter relative to the
pulse length. Confirmed on a coarse (whole-nanosecond) delay grid -- but
that grid turns out to hide a much larger, sub-carrier-period-sensitive
effect entirely. See THE FINDING below, and Panel D in particular.

New schematic for this investigation: tests/test_schematic_impedance_mismatch.si
-- based on test_schematic_basic.si (lossless, single amplifier), but with
THREE quantities pulled out as SI project-level Variables instead of fixed
values, so they can be swept from Python without hand-editing the .si file
per point:
  - Zmismatch: the amplifier's OWN output impedance (D1's `zo`) AND the
    VQubit-side termination impedance (R2's `r`) -- deliberately tied to
    the SAME variable, so both ends of the line between them are equally
    mismatched (matches how the user framed the question). 50.0 exactly
    reproduces the original matched, reflection-free baseline.
  - Tprop: T2's propagation delay (the line between DriverOutput, the
    amplifier's physical output node, and VQubit, the qubit plane) --
    everything else in the schematic (R1, T1, D1's zi, the drive-side
    impedances) stays fixed at the original matched 50 ohm/1ns values, so
    the ONLY mismatch in the whole chain is the one under study.

SI mechanism used (new to this codebase, verified directly against the
installed SignalIntegrity source before use -- see
SignalIntegrity/App/SignalIntegrityAppHeadless.py):
`SignalIntegrityAppHeadless.OpenProjectFile(filename, args={...})` matches
`args` keys against the schematic's own declared `<Variables>` names and
sets their values before the network is ever solved -- this is SI's own,
already-existing parametrized-schematic mechanism; nothing new had to be
added to SI itself. si_qfi.schematic.loader.load_schematic() now exposes
it as a `variables=` keyword, threaded straight through to that `args`
parameter -- see its docstring for the exact contract (applied once,
before any transfer function is extracted, since extraction is cached per
SI app instance).

A real numerical trap, load-bearing for this whole investigation: SI's
frequency-domain sweep (CalculationProperties EndFrequency/FrequencyPoints)
has a DISCRETE frequency grid with spacing df = EndFrequency/FrequencyPoints,
and the real-axis impulse response derived from it (an IFFT) is only valid
up to a MAXIMUM time window of 1/df -- energy arriving later than that wraps
around and lands on top of whatever's near the start of the window instead.
A severe mismatch's reflection ladder (successive round-trip bounces,
amplitude decaying by Gamma^2 per bounce) can take MANY round trips --
several microseconds, for the Gamma~0.7 case tested here -- to actually die
out, not just one. THIS schematic's EndFrequency=7GHz/FrequencyPoints=32000
gives df=218.75kHz, i.e. a 1/df~4.57us unambiguous window -- comfortably
past every bounce that matters for the Tprop range swept below (max 100ns
one-way here). EndFrequency itself is kept at 7GHz rather than SI's usual
20GHz (only needs to comfortably cover the 5GHz carrier + pulse bandwidth,
and reflections are a purely linear phenomenon, already established
mode-independent for linear effects in bandwidth_dispersion_fidelity_demo.
py's Investigation 4, so real_axis mode's wider sweep is never needed here)
to keep the per-point SI solve fast enough for a real parameter sweep.

THE FINDING (Panels A-C -- coarse, whole-nanosecond Tprop grid):
  - At Zmismatch=50 (Gamma=0) EXACTLY, infidelity stays at the numerical
    floor (~1e-7) at ANY delay, including very long ones -- no reflection
    coefficient, no reflection, full stop, regardless of timing.
  - At a SHORT delay (round-trip 2*Tprop << pulse duration), infidelity
    stays at the numerical floor EVEN AT THE MOST SEVERE impedance
    mismatch tested (300 ohm, Gamma=0.71) -- confirming the "delay must
    matter too" half of the hypothesis directly: a mismatch with nowhere
    for its reflection to meaningfully separate from the forward wave in
    time does essentially nothing to gate fidelity.
  - At a LONG delay, infidelity does grow with the reflection coefficient
    Gamma = |Z-50|/|Z+50| (confirmed symmetric in Z around 50 ohm -- Z=25
    and Z=100, both Gamma=0.333, give IDENTICAL infidelity, as expected
    since the physics only depends on |Gamma|) and with delay itself, but
    STAYS MODEST (parts in 1e-6 to 1e-5 across the whole range tested).
  - Sweeping delay at FIXED severe mismatch (Gamma=0.5): infidelity rises
    smoothly (not a hard cliff) with round-trip-delay/pulse-duration ratio.

WHY THAT PICTURE IS INCOMPLETE, AND WHAT PANEL D SHOWS INSTEAD:
Every Tprop value in Panels A-C above (and in every regression test) is a
whole number of nanoseconds. That's not a neutral choice: the carrier is
exactly 5GHz, so a whole-nanosecond delay is ALWAYS an exact integer number
of carrier cycles -- the reflected echo's carrier phase on arrival is
2*pi*5GHz*Tprop, which is therefore ALWAYS 0 (mod 2*pi) at every point
Panels A-C sample. Resolving Tprop at the sub-carrier-period scale (the
carrier period is 1/5GHz = 0.2ns) reveals what that coarse grid structurally
cannot see: infidelity swings from the numerical floor up to order-1 (>50%
observed) within picoseconds of Tprop, periodic with the 0.2ns carrier
period -- Panels A-C were, by construction, sampling only the safest phase
at every single point.

Is this a miscalibration artifact, or genuine distortion? Tested directly
(see complex_calibrated_infidelity_at()): even a STRONGER calibration that
launches with the exact complex (amplitude AND phase) scale needed to hit
the target total rotation area exactly -- verified to land on it to machine
precision -- does NOT fix it. Infidelity stays comparably large. This proves
it isn't "wrong total area" (which a 2-parameter calibration would cure);
it's that the reflected echo, at comparable amplitude to the direct pulse
and overlapping its tail, makes the INSTANTANEOUS drive axis wobble between
X and Y during the overlap -- no single global rescale of a fixed pulse
shape (real or complex) can undo a time-dependent axis wobble, only correct
its net integrated total. Practically: any calibration no more sophisticated
than picking a launch amplitude (or amplitude+phase) -- which is what real
setups typically do -- cannot correct this, and its severity depends on
cable length at the sub-picosecond scale, which real setups do not control.
Panels A-C's optimistic numbers hold only at the specific (lucky) delays
tested, not as a general statement about impedance mismatch.

Run: python examples/impedance_mismatch_demo.py
Requires: SignalIntegrity, QuTiP, matplotlib.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import qutip

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope, source_from_envelope_array
from si_qfi.simulation import engine as _engine
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_impedance_mismatch.si").resolve()

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6
DURATION_S = 25e-9
SIGMA_S = DURATION_S / 6
FS_ENVELOPE = 8e9


def infidelity_at(zmismatch: float, tprop_s: float, qmodel, mode="complex_baseband") -> float:
    """
    tuneup_amplitude()-calibrated pi-pulse infidelity for a given
    (Zmismatch, Tprop) pair. No nonlinearity anywhere -- this is a purely
    linear reflection effect, so tuneup_amplitude()'s analytic-guess fast
    path (one reference run + exact rescale) handles every point here in 2
    engine.run() calls, same cost as the hand-rolled version this replaces.
    """
    schematic = si_loader.load_schematic(
        SCHEMATIC_PATH, variables={"Zmismatch": zmismatch, "Tprop": tprop_s},
    )
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=ETA, ideal_gate="X", mode=mode,
    )
    return 1.0 - tuned.fidelity.noise_free.F_avg


def complex_calibrated_infidelity_at(zmismatch: float, tprop_s: float, qmodel) -> float:
    """
    A DELIBERATELY stronger calibration than infidelity_at()/
    tuneup_amplitude(): instead of rescaling the reference pulse by a single
    REAL amplitude, rescale it by a COMPLEX number (amplitude AND phase),
    chosen so the TOTAL (I-quadrature + i*Q-quadrature) integrated area at
    the qubit lands exactly on the target (pi, 0) -- i.e. this corrects not
    just "wrong overall amplitude" but also "wrong overall launch phase",
    which infidelity_at() cannot (its calibration only ever targets the
    real/X-quadrature area, see tuneup_amplitude()'s own docstring).

    This exists to answer a specific question directly rather than by
    argument: is the extreme delay-phase sensitivity seen at sub-carrier-
    period Tprop resolution (see the fine sweep in main()) just a
    miscalibration artifact that a better calibration would erase, or is it
    genuine waveform distortion? If it were the former, this function would
    recover near-unit fidelity everywhere infidelity_at() does not. It does
    not (confirmed directly, see main()'s printed comparison) -- proving the
    effect is real distortion (the echo overlaps the drive's own tail at
    comparable amplitude, so the instantaneous drive axis wobbles between X
    and Y during the overlap; no single global rescale of a fixed pulse
    shape, real or complex, can undo a time-dependent axis wobble, only its
    net integrated total).
    """
    schematic = si_loader.load_schematic(
        SCHEMATIC_PATH, variables={"Zmismatch": zmismatch, "Tprop": tprop_s},
    )
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    src_ref = source_from_envelope_array(ref_shape, FS_ENVELOPE, CARRIER_GHZ)
    result_ref = _engine.run(schematic, src_ref, nonlinear=None, noise=None, n_realizations=1, mode="complex_baseband")
    v_ref = np.asarray(result_ref.v_nl_qubit)
    t_ref = np.arange(len(v_ref)) / result_ref.fs
    area_ref = ETA * np.trapz(v_ref, t_ref)   # complex: theta_x_ref + i*theta_y_ref

    complex_scale = np.pi / area_ref   # exact for a linear channel -- no search needed
    source_shape = complex_scale * ref_shape.astype(complex)
    src = source_from_envelope_array(source_shape, FS_ENVELOPE, CARRIER_GHZ)
    result = _engine.run(schematic, src, nonlinear=None, noise=None, n_realizations=1, mode="complex_baseband")
    fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
    return 1.0 - fid.noise_free.F_avg


def reflection_coefficient(z, z0=50.0):
    return abs(z - z0) / (z + z0)


def main():
    warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")
    qmodel = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

    # ------------------------------------------------------------------
    # Panel A: fixed severe mismatch (Zmismatch=150 -> Gamma=0.5), sweep
    # round-trip-delay/pulse-duration ratio.
    # ------------------------------------------------------------------
    print("Panel A: infidelity vs delay ratio, at Gamma=0.5...")
    tprops_ns = np.array([100, 0.5, 1, 5, 10, 25, 50, 75, 100,])
    ratio_a = (2 * tprops_ns * 1e-9) / DURATION_S
    infid_a = np.array([infidelity_at(150.0, t * 1e-9, qmodel) for t in tprops_ns])

    # ------------------------------------------------------------------
    # Panel B: sweep Gamma at two fixed delays -- one deliberately SHORT
    # (round-trip << pulse duration) and one LONG (round-trip > pulse
    # duration) -- the direct test of "does mismatch alone matter".
    # ------------------------------------------------------------------
    print("Panel B: infidelity vs Gamma, short vs long delay...")
    z_values = np.array([50.0, 55.0, 60.0, 70.0, 80.0, 100.0, 125.0, 150.0, 200.0, 300.0])
    gammas = reflection_coefficient(z_values)
    tprop_short_s, tprop_long_s = 5e-9, 100e-9
    infid_b_short = np.array([infidelity_at(z, tprop_short_s, qmodel) for z in z_values])
    infid_b_long = np.array([infidelity_at(z, tprop_long_s, qmodel) for z in z_values])

    # ------------------------------------------------------------------
    # Panel C: 2D heatmap over the full (Zmismatch, Tprop) grid -- the
    # combined picture. Modest grid (7x7) to keep runtime reasonable at
    # ~2-5s/point.
    # ------------------------------------------------------------------
    print("Panel C: 2D sweep (this takes a few minutes)...")
    z_grid = np.array([50.0, 65.0, 80.0, 100.0, 130.0, 170.0, 220.0])
    tprop_grid_ns = np.array([1, 5, 15, 40, 80, 100])
    infid_c = np.zeros((len(tprop_grid_ns), len(z_grid)))
    for i, t_ns in enumerate(tprop_grid_ns):
        for j, z in enumerate(z_grid):
            infid_c[i, j] = infidelity_at(z, t_ns * 1e-9, qmodel)

    # ------------------------------------------------------------------
    # Panel D: EVERY Tprop used in Panels A-C above (and in the regression
    # tests) is a whole number of nanoseconds -- and because the carrier is
    # exactly 5GHz, a whole-nanosecond delay is ALWAYS an exact integer
    # number of carrier cycles. That's not a neutral choice of sweep grid:
    # the reflected echo's carrier phase on arrival is 2*pi*carrier*Tprop,
    # so every whole-ns Tprop lands EXACTLY on a phase-zero point where the
    # echo's contribution is purely real (X-axis only, no Y leakage) --
    # coincidentally the SAFEST possible point. This panel resolves Tprop at
    # the sub-carrier-period scale (carrier period = 1/5GHz = 0.2ns) to show
    # what Panels A-C's coarse, whole-ns grid never samples: infidelity
    # swinging from the numerical floor up to order-1 within a few
    # picoseconds of Tprop, periodic with the 0.2ns carrier period.
    #
    # Two curves are computed at every point to distinguish MISCALIBRATION
    # from genuine DISTORTION:
    #   - "real-scale calibration" = infidelity_at(), i.e. what every other
    #     panel and every regression test uses: tuneup_amplitude() picks a
    #     single REAL amplitude so the total X-quadrature area hits pi.
    #   - "complex-scale calibration" = complex_calibrated_infidelity_at():
    #     a strictly STRONGER calibration allowed to also pick the launch
    #     PHASE, so the total (X, Y) area lands exactly on (pi, 0) -- not
    #     just close, exact, verified directly (see that function's
    #     docstring). If the sub-carrier-period sensitivity were just a
    #     calibration gap, this curve would sit at the floor everywhere.
    #     It does not -- it tracks the real-scale curve within a factor of
    #     ~2 almost everywhere, proving this is genuine waveform distortion
    #     (the echo overlaps the drive's own tail at comparable amplitude,
    #     so the instantaneous drive axis wobbles between X and Y during the
    #     overlap -- no single global rescale, real or complex, of a FIXED
    #     pulse shape can undo a time-dependent axis wobble, only correct
    #     its net integrated total).
    print("Panel D: fine (sub-carrier-period) Tprop sweep at Gamma=0.71 "
          "-- real-scale vs complex-scale calibration...")
    tprop_d_base_ns = 100.0
    offsets_ns = np.linspace(0.0, 0.2, 11)   # exactly one carrier period (1/5GHz)
    tprops_d_ns = tprop_d_base_ns + offsets_ns
    infid_d_real = np.array([infidelity_at(300.0, t * 1e-9, qmodel) for t in tprops_d_ns])
    infid_d_complex = np.array([complex_calibrated_infidelity_at(300.0, t * 1e-9, qmodel) for t in tprops_d_ns])

    print()
    print("=== Summary ===")
    print(f"Panel A: infidelity spans {infid_a.min():.2e} to {infid_a.max():.2e} "
          f"as round-trip-delay/pulse-duration goes from {ratio_a.min():.3f} to {ratio_a.max():.3f}")
    print(f"Panel B (short delay, {tprop_short_s*1e9:.0f}ns): "
          f"infidelity stays within [{infid_b_short.min():.2e}, {infid_b_short.max():.2e}] "
          f"across Gamma=0 to Gamma={gammas.max():.2f}")
    print(f"Panel B (long delay, {tprop_long_s*1e9:.0f}ns): "
          f"infidelity spans [{infid_b_long.min():.2e}, {infid_b_long.max():.2e}] over the same Gamma range")
    print(f"Panel D: over ONE carrier period ({tprop_d_base_ns:.1f}ns to {tprops_d_ns[-1]:.1f}ns, Gamma=0.71), "
          f"real-scale-calibrated infidelity spans {infid_d_real.min():.2e} to {infid_d_real.max():.2e}; "
          f"complex-scale (exact total-area) calibration spans {infid_d_complex.min():.2e} to "
          f"{infid_d_complex.max():.2e} -- fixing the total launch amplitude+phase does NOT fix this, "
          f"confirming genuine waveform distortion, not miscalibration.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))
    floor = 1e-8

    ax1.loglog(ratio_a, np.maximum(infid_a, floor), "o-", color="C0")
    ax1.axvline(1.0, color="gray", ls="--", lw=1, label="round-trip delay = pulse duration")
    ax1.set_xlabel("Round-trip delay / pulse duration  (2*Tprop / T_pulse)")
    ax1.set_ylabel("Infidelity to X (1 - F_avg)")
    ax1.set_title("A: Fixed severe mismatch (Gamma=0.5)\nvs. delay ratio")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    ax2.semilogy(gammas, np.maximum(infid_b_short, floor), "o-", color="C2",
                 label=f"short delay ({tprop_short_s*1e9:.0f}ns, round-trip << pulse)")
    ax2.semilogy(gammas, np.maximum(infid_b_long, floor), "o-", color="C3",
                 label=f"long delay ({tprop_long_s*1e9:.0f}ns, round-trip > pulse)")
    ax2.set_xlabel("Reflection coefficient |Gamma| = |Z-50|/|Z+50|")
    ax2.set_ylabel("Infidelity to X (1 - F_avg)")
    ax2.set_title("B: Mismatch alone is not enough --\nneeds delay too")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    # z_grid/tprop_grid_ns are coarse and non-uniformly (log-ish) spaced --
    # pcolormesh + a genuine log y-axis mishandles that (matplotlib infers
    # the last row's cell edge from a LINEAR delta, which explodes once
    # log-scaled, leaving most of the axis blank). Simplest robust fix:
    # treat this as a categorical index grid (imshow) with the actual
    # sampled values as tick labels, rather than pretending a 7x7 grid
    # supports continuous log-axis interpolation anyway.
    im = ax3.imshow(
        np.log10(np.maximum(infid_c, floor)),
        origin="lower", aspect="auto", cmap="viridis",
    )
    ax3.set_xticks(range(len(z_grid)))
    ax3.set_xticklabels([f"{z:.0f}" for z in z_grid])
    ax3.set_yticks(range(len(tprop_grid_ns)))
    ax3.set_yticklabels([f"{t:.0f}" for t in tprop_grid_ns])
    ax3.set_xlabel("Zmismatch (Ohm)")
    ax3.set_ylabel("Tprop (ns)")
    ax3.set_title("C: Combined picture\nlog10(infidelity)")
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label("log10(1 - F_avg)")

    ax4.semilogy(offsets_ns, np.maximum(infid_d_real, floor), "o-", color="C1",
                 label="real-scale calibration (what A-C/tests use)")
    ax4.semilogy(offsets_ns, np.maximum(infid_d_complex, floor), "s--", color="C4",
                 label="complex-scale calibration (exact total area)")
    ax4.axvline(0.0, color="gray", ls=":", lw=1)
    ax4.axvline(0.1, color="gray", ls=":", lw=1, label="half carrier period")
    ax4.axvline(0.2, color="gray", ls=":", lw=1)
    ax4.set_xlabel(f"Tprop - {tprop_d_base_ns:.0f}ns  (offset, ns -- one full 5GHz carrier period)")
    ax4.set_ylabel("Infidelity to X (1 - F_avg)")
    ax4.set_title("D: Panels A-C's whole-ns Tprop grid\nalways lands on the safe points --\nthis is what's between them (Gamma=0.71)")
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out_path = Path(__file__).parent / "impedance_mismatch_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
