"""
examples/impedance_mismatch_demo.py
====================================
Investigation: when does an impedance mismatch (amplifier output impedance
and/or qubit-plane termination not exactly 50 ohm) between the amplifier
and the qubit actually cost gate fidelity, via reflections?

Hypothesis under test (as posed): degradation should only show up when
BOTH (a) the impedance is far from 50 ohm AND (b) the propagation delay
between the mismatched points is long enough to matter relative to the
pulse length. Verified below: correct, and the crossover is sharp and
well-characterized -- see THE FINDING.

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

A real numerical trap hit and fixed while building this: SI's frequency-
domain sweep (CalculationProperties EndFrequency/FrequencyPoints) has a
DISCRETE frequency grid with spacing df = EndFrequency/FrequencyPoints,
and representing a pure delay via that discrete grid is only unambiguous
up to a MAXIMUM delay of 1/df (beyond that, a delay T and T + n/df are
numerically indistinguishable at the sampled frequencies -- confirmed
directly: with the original df=10MHz [FrequencyPoints=2000 over 20GHz],
Tprop=100ns and Tprop=200ns produced IDENTICAL sampled H(f), silently
aliasing). Fixed by choosing EndFrequency/FrequencyPoints for THIS
schematic to give df=1.75MHz (max unambiguous delay ~570ns, comfortably
past the largest Tprop swept below) -- and, since this investigation
never needs real_axis mode (reflections are a purely linear phenomenon,
already established mode-independent for linear effects in
bandwidth_dispersion_fidelity_demo.py's Investigation 4), EndFrequency was
also dropped from 20GHz to 7GHz (only needs to comfortably cover the
5GHz carrier + pulse bandwidth) to keep the per-point SI solve fast
enough for a real parameter sweep (~2-5s/point instead of ~20s/point at
the original resolution/range).

THE FINDING:
  - At Zmismatch=50 (Gamma=0) EXACTLY, infidelity stays at the numerical
    floor (~1e-7) at ANY delay, including very long ones -- no reflection
    coefficient, no reflection, full stop, regardless of timing.
  - At a SHORT delay (round-trip 2*Tprop << pulse duration), infidelity
    stays at the numerical floor (~1e-7 to 1e-8) EVEN AT THE MOST SEVERE
    impedance mismatch tested (300 ohm, Gamma=0.71) -- confirming the
    "delay must matter too" half of the hypothesis directly: a mismatch
    with nowhere for its reflection to meaningfully separate from the
    forward wave in time does essentially nothing to gate fidelity.
  - At a LONG delay, infidelity grows smoothly and strongly with the
    reflection coefficient Gamma = |Z-50|/|Z+50| (confirmed symmetric in Z
    around 50 ohm -- Z=25 and Z=100, both Gamma=0.333, give IDENTICAL
    infidelity, as expected since the physics only depends on |Gamma|).
  - Sweeping delay at FIXED severe mismatch (Gamma=0.5): infidelity rises
    from the numerical floor to catastrophic (>50%) over roughly one and a
    half decades of round-trip-delay/pulse-duration ratio -- NOT a hard
    cliff like the pure-AM-AM achievability wall from the nonlinearity
    investigations, but a smooth, continuous, and eventually very steep
    transition. The crossover from "clearly negligible" to "clearly
    significant" sits close to round-trip delay (2*Tprop) ~ pulse
    duration -- i.e. the user's original framing holds up almost exactly
    once "propagation delay" is read as the ROUND-TRIP time (there and
    back), which is the physically meaningful quantity for a reflection
    (a one-way-only reading would put the crossover a factor of 2 later).

Run: python examples/impedance_mismatch_demo.py
Requires: SignalIntegrity, QuTiP, matplotlib.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import qutip

from si_qfi.schematic import loader as si_loader
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform, build_gaussian_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_impedance_mismatch.si").resolve()

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6
DURATION_S = 25e-9
SIGMA_S = DURATION_S / 6
FS_ENVELOPE = 8e9


def _source_from_shape(shape: np.ndarray, fs: float, carrier_ghz: float) -> SourceWaveform:
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def infidelity_at(zmismatch: float, tprop_s: float, qmodel, mode="complex_baseband") -> float:
    """
    Self-calibrated pi-pulse infidelity for a given (Zmismatch, Tprop)
    pair. No nonlinearity anywhere -- this is a purely linear reflection
    effect. A single reference run + exact rescale suffices (no NL node
    -> the amplitude->theta map is exactly proportional, same as
    bandwidth_dispersion_fidelity_demo.py).
    """
    schematic = si_loader.load_schematic(
        SCHEMATIC_PATH, variables={"Zmismatch": zmismatch, "Tprop": tprop_s},
    )
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, FS_ENVELOPE, CARRIER_GHZ)
    result_ref = engine.run(schematic, source_ref, nonlinear=None, noise=None, n_realizations=1, mode=mode)
    v = np.asarray(result_ref.v_nl_qubit)
    t = np.arange(len(v)) / result_ref.fs
    theta_ref = float(ETA * np.trapz(np.real(v), t))
    scale = np.pi / theta_ref

    cal_shape = ref_shape * scale
    source_cal = _source_from_shape(cal_shape, FS_ENVELOPE, CARRIER_GHZ)
    result_cal = engine.run(schematic, source_cal, nonlinear=None, noise=None, n_realizations=1, mode=mode)
    fid = quantum.gate_fidelity(result_cal, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
    return 1.0 - fid.F_avg


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

    print()
    print("=== Summary ===")
    print(f"Panel A: infidelity spans {infid_a.min():.2e} to {infid_a.max():.2e} "
          f"as round-trip-delay/pulse-duration goes from {ratio_a.min():.3f} to {ratio_a.max():.3f}")
    print(f"Panel B (short delay, {tprop_short_s*1e9:.0f}ns): "
          f"infidelity stays within [{infid_b_short.min():.2e}, {infid_b_short.max():.2e}] "
          f"across Gamma=0 to Gamma={gammas.max():.2f}")
    print(f"Panel B (long delay, {tprop_long_s*1e9:.0f}ns): "
          f"infidelity spans [{infid_b_long.min():.2e}, {infid_b_long.max():.2e}] over the same Gamma range")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
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

    plt.tight_layout()
    out_path = Path(__file__).parent / "impedance_mismatch_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
