"""
examples/noise_density_sweep_demo.py
=====================================
Investigation: how does a drive-line noise source's spectral density cost
gate fidelity, and does it scale the way a small-perturbation argument
predicts?

Uses tests/test_schematic_noise.si -- test_schematic_basic.si (the plain
lossless amplifier chain every other baseline investigation in this series
starts from) with one addition: a real SI DeviceVoltageStatisticalNoiseSource
device (VN1), spliced in series on the line between the amplifier's output
and the qubit-plane termination, configured as Johnson noise (50 ohm,
290 K -- see schematic.noise's module docstring for why this needs a
resolvable placeholder waveform on VSource, and why that doesn't affect the
physics here: si_qfi always injects its own drive waveform at run time).

This demo sweeps the noise density using the OVERRIDE mechanism
(`noise={"VN1": {"single_sided_psd_v2_per_hz": ...}}`) rather than SI's own
Johnson computation, specifically so the density can be swept over many
decades independent of any physical resistor/temperature value -- see
tests/test_engine_noise.py for the (already-passing) confirmation that both
the override and SI-native paths are physically equivalent (same code path
downstream, only the PSD's source differs).

Method: calibrate a pi-pulse ONCE (noise-free, via tuneup_amplitude()) at
the tuned amplitude, then re-run engine.run() at that fixed amplitude with
noise enabled at each swept density, computing gate_fidelity()'s noise
ensemble statistics (F_avg, F_std, F_sem) at each point.

Run: python examples/noise_density_sweep_demo.py
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

SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_noise.si").resolve()

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6
DURATION_S = 20e-9
SIGMA_S = DURATION_S / 6
FS_ENVELOPE = 4e9
N_REALIZATIONS = 200


def main():
    warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")
    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    qmodel = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

    # ------------------------------------------------------------------
    # Calibrate once, noise-free (tuneup_amplitude optimizes the noise-free
    # fidelity by design -- see quantum/fidelity.py's own docstring).
    # ------------------------------------------------------------------
    print("Calibrating pi-pulse amplitude (noise-free)...")
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=ETA, ideal_gate="X",
    )
    print(f"  tuned scale={tuned.scale:.6f}, achieved={tuned.achieved}, "
          f"noise-free infidelity={1.0 - tuned.fidelity.noise_free.F_avg:.3e}")
    tuned_shape = tuned.scale * ref_shape

    # ------------------------------------------------------------------
    # Sweep noise density at the fixed, noise-free-calibrated amplitude.
    # ------------------------------------------------------------------
    print("Sweeping noise density...")
    psd_values = np.geomspace(1e-20, 1e-10, 15)   # V^2/Hz
    infid_avg = np.zeros_like(psd_values)
    infid_sem = np.zeros_like(psd_values)
    snr_values = np.zeros_like(psd_values)

    for i, psd in enumerate(psd_values):
        src = source_from_envelope_array(tuned_shape, FS_ENVELOPE, CARRIER_GHZ)
        result = _engine.run(
            schematic, src, nonlinear=None, noise={"VN1": {"single_sided_psd_v2_per_hz": float(psd)}},
            n_realizations=N_REALIZATIONS, mode="complex_baseband", seed=123,
        )
        fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
        infid_avg[i] = 1.0 - fid.noise.F_avg
        infid_sem[i] = fid.noise.F_sem
        # complex_baseband's v_nl_qubit/v_qubit_ensemble are already in a
        # consistent complex-envelope representation, so no lpf_cutoff_hz
        # is needed here (see quantum.pulse_snr()'s own docstring).
        snr_result = quantum.pulse_snr(result)
        snr_values[i] = snr_result.snr
        print(f"  psd={psd:.2e} V^2/Hz  SNR={snr_result.snr:.3e} ({snr_result.snr_db:+.1f} dB)  "
              f"infidelity={infid_avg[i]:.4e} +/- {infid_sem[i]:.1e}")

    floor = tuned.fidelity.noise_free.F_avg
    print()
    print("=== Summary ===")
    print(f"Noise-free infidelity floor: {1.0 - floor:.3e}")
    print(f"Infidelity spans {infid_avg.min():.3e} to {infid_avg.max():.3e} "
          f"over psd={psd_values.min():.1e} to {psd_values.max():.1e} V^2/Hz")

    # Small-perturbation scaling check: fit infidelity ~ A * psd^p over the
    # region well above the noise-free floor (where noise-driven infidelity
    # actually dominates), report the fitted power p. Uses an ABSOLUTE floor
    # threshold, not `3 * (1.0 - floor)` directly -- the noise-free F_avg can
    # itself land slightly *above* 1.0 (a known QuTiP-solver-precision
    # artifact at near-perfect fidelity, documented elsewhere in this
    # codebase's investigations), making `1.0 - floor` negative and flipping
    # the comparison to admit near-floor NEGATIVE infidelities into the log
    # fit (log of a negative number -> nan). Also require infid_avg > 0
    # explicitly, since only that guarantees np.log() is well-defined,
    # regardless of the threshold's own sign.
    _NUMERICAL_FLOOR = 1e-7
    significant = (infid_avg > 0) & (infid_avg > 3 * max(abs(1.0 - floor), _NUMERICAL_FLOOR))
    if np.sum(significant) >= 3:
        p, log_a = np.polyfit(np.log(psd_values[significant]), np.log(infid_avg[significant]), 1)
        print(f"Fitted scaling (region above floor): infidelity ~ psd^{p:.2f} "
              f"(1.0 = linear, matches a small-perturbation/first-order argument)")
    if np.any(~significant):
        print(
            f"Note: {np.sum(~significant)} point(s) at the low-density/high-SNR end sit within "
            f"3x the noise-free floor -- there, noise-driven infidelity is comparable to or "
            f"smaller than the intrinsic finite-ensemble Monte Carlo scatter, so those points can "
            f"swing above OR below zero (even dip visibly below neighboring points, as seen in the "
            f"plot) without indicating any real non-monotonicity in the underlying physics -- only "
            f"the >3x-floor points (filled markers below) are used for the scaling fit above."
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Display-only floor clip -- infidelity (and the noise-free floor
    # itself) can land slightly negative near machine precision at
    # near-perfect fidelity (see the scaling-fit fix above for why); a log
    # y-axis silently drops negative points/lines entirely rather than
    # showing them, so clip for plotting only (printed/summary values above
    # are untouched). Matches the floor-clipping convention already used in
    # examples/impedance_mismatch_demo.py for the same reason.
    plot_floor = 1e-9
    fig, (ax_psd, ax_snr) = plt.subplots(1, 2, figsize=(13, 5))

    infid_plot = np.maximum(infid_avg, plot_floor)

    def _plot_significance_split(ax, x, color):
        # Solid line through everything (so the eye can still follow the
        # trend), but hollow/faded markers for points within 3x the floor
        # -- those are Monte Carlo scatter, not reliable individual data
        # points (see the printed note above and module docstring).
        ax.errorbar(x, infid_plot, yerr=infid_sem, fmt="-", color=color, alpha=0.5, zorder=1)
        ax.scatter(x[significant], infid_plot[significant], color=color, zorder=2, label="_nolegend_")
        ax.scatter(
            x[~significant], infid_plot[~significant],
            facecolors="none", edgecolors=color, zorder=2, label="_nolegend_",
        )

    _plot_significance_split(ax_psd, psd_values, "C0")
    ax_psd.axhline(max(1.0 - floor, plot_floor), color="gray", ls="--", lw=1, label="noise-free floor")
    ax_psd.scatter([], [], color="C0", label="> 3x floor (used in fit)")
    ax_psd.scatter([], [], facecolors="none", edgecolors="C0", label="within 3x floor (MC scatter)")
    ax_psd.set_xscale("log")
    ax_psd.set_yscale("log")
    ax_psd.set_xlabel("Noise source PSD (V²/Hz)")
    ax_psd.set_ylabel("Infidelity to X (1 - F_avg)")
    ax_psd.set_title("vs. drive-line noise density")
    ax_psd.legend(fontsize=8)
    ax_psd.grid(alpha=0.3, which="both")

    # SNR-vs-infidelity: SNR decreases as PSD increases, so this is the
    # same data re-parameterized -- useful because SNR (not raw PSD) is
    # the quantity that's actually physically meaningful for how much a
    # given noise source degrades the gate, and because a "hump" visible
    # in the PSD view can be checked here for whether it's a real feature
    # or just scatter near the noise-free floor (see module docstring).
    _plot_significance_split(ax_snr, snr_values, "C1")
    ax_snr.axhline(max(1.0 - floor, plot_floor), color="gray", ls="--", lw=1, label="noise-free floor")
    ax_snr.scatter([], [], color="C1", label="> 3x floor (used in fit)")
    ax_snr.scatter([], [], facecolors="none", edgecolors="C1", label="within 3x floor (MC scatter)")
    ax_snr.set_xscale("log")
    ax_snr.set_yscale("log")
    ax_snr.invert_xaxis()   # low SNR (bad) on the left, high SNR (good) on the right
    ax_snr.set_xlabel("Effective SNR (signal power / noise power)")
    ax_snr.set_ylabel("Infidelity to X (1 - F_avg)")
    ax_snr.set_title("vs. effective SNR")
    ax_snr.legend(fontsize=8)
    ax_snr.grid(alpha=0.3, which="both")

    fig.suptitle("Gate infidelity vs. drive-line noise\n(VN1, statistical noise source)")
    plt.tight_layout()
    out_path = Path(__file__).parent / "noise_density_sweep_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
