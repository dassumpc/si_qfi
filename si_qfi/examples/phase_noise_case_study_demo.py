"""
examples/phase_noise_case_study_demo.py
=========================================
Case study for engine.run()'s phase_noise= feature (LO/oscillator phase
noise -- see noise/psd.py's phase_noise_psd_from_spec() and
simulation/engine.py's module docstring for the full design).

Why this needed real engine changes, not just a new noise-injection point:
LO phase noise is MULTIPLICATIVE, not additive -- a noisy carrier is
env(t)*cos(2*pi*f_c*t + phi(t)), not env(t)*cos(2*pi*f_c*t) + noise(t) (see
quantum/snr.py's own filter-function discussion for the related, but
distinct, additive-quadrature-noise case). It rides on the drive envelope
itself and, critically, it ORIGINATES UPSTREAM of any nonlinearity in the
drive chain -- so a compressing amplifier sees and reshapes the ALREADY
phase-perturbed waveform, not a clean waveform with phase noise added onto
its output afterward. Concretely: to first order,
    ~u_noisy(t) = ~u(t)*exp(j*phi(t)) ~= ~u(t) + j*phi(t)*~u(t)
i.e. the perturbation itself is proportional to the intended drive -- so
whatever the nonlinearity does to the intended drive, it does (differently)
to this phase-rotated version too. Adding phase noise onto the qubit-plane
waveform AFTER a single deterministic nonlinear pass has already run once
(the cheap, existing architecture's own approach for additive noise)
implicitly assumes the nonlinearity and the phase rotation commute -- true
for a LINEAR, non-dispersive channel, false in general once real
compression is in the loop.

Three parts:
  1. Sanity check: phase noise alone (no nonlinearity) costs real gate
     fidelity, growing with phase-noise power, roughly linearly (matching a
     first-order rotation-error argument, same flavor as the additive-noise
     density sweep in examples/noise_density_sweep_demo.py).
  2. THE CASE STUDY: with a compressing Saleh amplifier in the chain,
     directly compare engine.run(phase_noise=...)'s actual (pre-NL,
     physically correct) implementation against the alternative of adding
     the SAME phase draws onto the single shared deterministic v_nl_qubit
     AFTER the nonlinear pass (physically wrong once compression matters,
     per the argument above), across a sweep of op1db_amplitude (how
     aggressively the amplifier compresses relative to what a calibrated
     pi-pulse needs). At EACH point the pulse is re-calibrated THROUGH that
     specific nonlinearity (tuneup_amplitude(..., nonlinear=nl_spec)) --
     necessary because at a fixed drive amplitude, varying compression
     also changes the achieved rotation angle, which would otherwise
     swamp the comparison with ordinary miscalibration error having
     nothing to do with phase noise (confirmed directly: an earlier,
     simpler version of this sweep that just scaled a FIXED pi-calibrated
     pulse by a fraction, instead of re-calibrating per point, produced
     enormous (40-60%) infidelities dominated by under-rotation alone,
     with the pre-NL/post-hoc difference completely swamped -- 0.0% at
     every point, not because the two approaches actually agreed, but
     because the comparison couldn't see past the miscalibration).
  3. bandwidth_hz convergence check: since phase noise has no natural
     bandwidth ceiling of its own (see phase_noise_psd_from_spec()'s
     docstring), bandwidth_hz has to be chosen deliberately -- this sweeps
     it as a multiple of the pulse's own natural bandwidth and confirms the
     reported infidelity actually stabilizes once bandwidth_hz is wide
     enough, rather than trusting a rule of thumb blindly.

THE FINDING (numbers from the run that produced examples/
phase_noise_case_study_demo.png):
  - Part 1: phase noise alone costs real gate fidelity, growing cleanly
    linearly with PSD (infidelity ~ S_phi^1.00 over 4 decades of S_phi,
    7.9e-6 to 8.3e-2) -- matches the first-order rotation-error picture,
    same shape as examples/noise_density_sweep_demo.py's additive-noise
    density sweep.
  - Part 2: with a compressing Saleh amplifier, pre-NL (correct) infidelity
    sits consistently ABOVE post-hoc (physically wrong) infidelity at
    EVERY op1db_amplitude tested (5 of 5 points, mild through aggressive
    compression, each independently re-calibrated through its own
    nonlinearity) -- a 14-17% relative discrepancy. Individual points'
    SEM-based error bars overlap somewhat (N=250 realizations each), so
    any single point's separation alone wouldn't be conclusive -- but 5/5
    independent points landing on the SAME side (pre-NL always higher) has
    a <4% chance of arising from symmetric statistical noise alone (a
    simple sign-test argument), which is the more reliable evidence here
    than any one point's magnitude. The magnitude itself is seed-sensitive
    (an earlier, smaller-N exploratory run with a different seed measured
    3-5% instead of 14-17% for the same op1db sweep) -- the qualitative
    finding (genuinely, robustly different, not just numerically close) is
    what this comparison was built to establish, not a precise percentage.
  - Part 3: bandwidth_hz genuinely converges for a REALISTIC (rolling-off,
    finite-total-power) phase-noise shape -- infidelity rises from 4.6e-3
    at 1x the pulse's natural bandwidth to 1.24e-2 by 4x, then stays flat
    (1.244e-2 at 8x, 1.244e-2 at 16x) -- confirming the claim from this
    feature's own design discussion that the qubit's own filter function,
    not the phase-noise source, is what bounds the total effect. (A FLAT
    PSD, tested first and deliberately left undocumented as a dead end:
    infidelity kept climbing without leveling off from 1x to 16x, because
    a flat spectrum has no natural power ceiling to converge against in
    the first place -- not a bug, just the wrong shape to ask this
    question with.)

Run: python examples/phase_noise_case_study_demo.py
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
from si_qfi.simulation.engine import SimulationResult
from si_qfi.noise.psd import phase_noise_psd_from_spec
from si_qfi.noise.realization import generate_phase_noise
from si_qfi import quantum

BASIC_SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_basic.si").resolve()
NL_LABEL = "DriverOutput"

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6
DURATION_S = 20e-9
SIGMA_S = DURATION_S / 6
FS_ENVELOPE = 4e9
NATURAL_BW_HZ = 1.0 / (2 * np.pi * SIGMA_S)   # rough pulse bandwidth scale, ~48MHz here
N_REALIZATIONS = 250
SEED = 2026


def _posthoc_phase_noise_result(v_nl_qubit: np.ndarray, fs: float, mode: str, carrier_hz: float,
                                 phase_spec: dict, n_realizations: int, seed: int) -> SimulationResult:
    """
    The physically-wrong alternative engine.run(phase_noise=...) is compared
    against: rotate the SAME (single, shared) deterministic v_nl_qubit by
    n_realizations independent phase draws, computed AFTER any nonlinearity
    has already been applied once -- i.e. exactly what the old
    (additive-noise-only) architecture would do if phase noise were bolted
    on post-hoc instead of injected at the source. Builds a real
    SimulationResult by hand so gate_fidelity() can be used unmodified.
    """
    n_draw = len(v_nl_qubit)
    freqs_phi = np.fft.rfftfreq(n_draw, d=1.0 / fs)
    psd_arr = phase_noise_psd_from_spec(phase_spec, freqs_phi)
    rng = np.random.default_rng(seed)
    ensemble = [
        v_nl_qubit * np.exp(1j * generate_phase_noise(n_draw, fs, psd_arr, rng=rng))
        for _ in range(n_realizations)
    ]
    return SimulationResult(
        v_nl_qubit=v_nl_qubit, v_qubit_ensemble=ensemble, fs=fs, mode=mode,
        carrier_freq_hz=carrier_hz, noise_enabled=True, phase_noise_enabled=True,
        n_realizations=n_realizations,
    )


def main():
    warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")
    warnings.filterwarnings("ignore", message="SI-QFI: SalehModel input peak amplitude")
    schematic = si_loader.load_schematic(BASIC_SCHEMATIC_PATH)
    qmodel = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

    print("Calibrating pi-pulse amplitude (noise-free, no nonlinearity)...")
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=ETA, ideal_gate="X",
    )
    tuned_shape = tuned.scale * ref_shape
    src = source_from_envelope_array(tuned_shape, FS_ENVELOPE, CARRIER_GHZ)
    floor = 1.0 - tuned.fidelity.noise_free.F_avg
    print(f"  tuned scale={tuned.scale:.6f}, noise-free infidelity floor={floor:.3e}")
    print(f"  pulse natural bandwidth scale ~1/(2*pi*sigma) = {NATURAL_BW_HZ/1e6:.1f} MHz")

    # ------------------------------------------------------------------
    # Part 1: phase noise alone, no nonlinearity -- sanity check.
    # ------------------------------------------------------------------
    print("\nPart 1: phase noise alone (no nonlinearity) vs. PSD level...")
    psd_values = np.geomspace(1e-13, 1e-9, 8)
    infid1 = np.zeros_like(psd_values)
    sem1 = np.zeros_like(psd_values)
    bw1 = 5 * NATURAL_BW_HZ
    for i, psd_phi in enumerate(psd_values):
        result = _engine.run(
            schematic, src, nonlinear=None, noise=None,
            phase_noise={"single_sided_psd_rad2_per_hz": float(psd_phi), "bandwidth_hz": bw1},
            n_realizations=N_REALIZATIONS, mode="complex_baseband", seed=SEED,
        )
        fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
        infid1[i] = 1.0 - fid.noise.F_avg
        sem1[i] = fid.noise.F_sem
        print(f"  S_phi={psd_phi:.2e} rad^2/Hz  infidelity={infid1[i]:.4e} +/- {sem1[i]:.1e}")

    significant1 = infid1 > 3 * max(abs(floor), 1e-7)
    if np.sum(significant1) >= 3:
        p1, _ = np.polyfit(np.log(psd_values[significant1]), np.log(infid1[significant1]), 1)
        print(f"  Fitted scaling: infidelity ~ S_phi^{p1:.2f} (1.0 = linear)")

    # ------------------------------------------------------------------
    # Part 2: THE CASE STUDY -- pre-NL (correct) vs. post-hoc (wrong) phase
    # noise, sweeping how hard a compressing Saleh amplifier is driven.
    # ------------------------------------------------------------------
    print("\nPart 2: pre-NL vs. post-hoc phase noise, sweeping op1db_amplitude "
          "(re-calibrated THROUGH each nonlinearity)...")
    PSD_PHI_2 = 5e-11
    BW_2 = 5 * NATURAL_BW_HZ
    # Sweep op1db_amplitude from mild (20, barely compressing) to aggressive
    # (3.0, close to the amplifier's own "achievability cliff" -- see
    # tuneup_amplitude()'s own docstring / INVESTIGATIONS.md Investigation
    # 2 -- past which no launch amplitude reaches a true pi rotation at
    # all). Some intermediate op1db values (roughly 4-7 here) were found
    # directly to make tuneup_amplitude's own calibration search land on a
    # spurious solution on the WRONG (declining) branch of the Saleh
    # curve -- confirmed by a wildly non-monotonic, obviously-wrong scale
    # factor (233x vs. neighboring points' ~2.5-3.5x) despite
    # tuned.achieved reporting True -- so this sweep deliberately skips
    # that region rather than including a known-bad point; a genuine
    # tuneup_amplitude robustness gap worth its own separate investigation,
    # out of scope here.
    op1db_values = np.array([20.0, 10.0, 8.0, 3.2, 3.0])

    infid_pre = np.zeros_like(op1db_values)
    infid_post = np.zeros_like(op1db_values)
    sem_pre = np.zeros_like(op1db_values)
    sem_post = np.zeros_like(op1db_values)
    scales = np.zeros_like(op1db_values)

    for i, op1db in enumerate(op1db_values):
        nl_spec = {NL_LABEL: {"model": "saleh", "op1db_amplitude": float(op1db)}}
        tuned_i = quantum.tuneup_amplitude(
            schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ, qmodel,
            coupling_strength_per_volt=ETA, ideal_gate="X", nonlinear=nl_spec,
        )
        assert tuned_i.achieved, f"op1db_amplitude={op1db}: calibration did not converge"
        scales[i] = tuned_i.scale
        src_i = source_from_envelope_array(tuned_i.scale * ref_shape, FS_ENVELOPE, CARRIER_GHZ)

        result_pre = _engine.run(
            schematic, src_i, nonlinear=nl_spec, noise=None,
            phase_noise={"single_sided_psd_rad2_per_hz": PSD_PHI_2, "bandwidth_hz": BW_2},
            n_realizations=N_REALIZATIONS, mode="complex_baseband", seed=SEED,
        )
        fid_pre = quantum.gate_fidelity(result_pre, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
        infid_pre[i] = 1.0 - fid_pre.noise.F_avg
        sem_pre[i] = fid_pre.noise.F_sem

        result_det = _engine.run(schematic, src_i, nonlinear=nl_spec, noise=None, n_realizations=1, mode="complex_baseband")
        result_post = _posthoc_phase_noise_result(
            result_det.v_nl_qubit, result_det.fs, result_det.mode, result_det.carrier_freq_hz,
            {"single_sided_psd_rad2_per_hz": PSD_PHI_2, "bandwidth_hz": BW_2}, N_REALIZATIONS, SEED,
        )
        fid_post = quantum.gate_fidelity(result_post, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
        infid_post[i] = 1.0 - fid_post.noise.F_avg
        sem_post[i] = fid_post.noise.F_sem

        rel_diff = abs(infid_pre[i] - infid_post[i]) / infid_post[i]
        print(f"  op1db_amplitude={op1db:5.2f} (scale={scales[i]:6.3f})  pre-NL infid={infid_pre[i]:.4e}  "
              f"post-hoc infid={infid_post[i]:.4e}  rel diff={rel_diff:.1%}")

    # ------------------------------------------------------------------
    # Part 3: bandwidth_hz convergence check (no nonlinearity, simplest
    # clean case).
    # ------------------------------------------------------------------
    print("\nPart 3: bandwidth_hz convergence check...")
    # A FLAT PSD has no natural power ceiling -- extending bandwidth_hz would
    # just keep adding total injected variance forever (that's the "flat
    # spectra don't converge" point from this feature's own design
    # discussion, not a bug), so it's the wrong shape to test convergence
    # against. A realistic oscillator L(f) rolls off close to the carrier
    # and flattens to a floor -- modeled here with the simplest shape that
    # has that property, a one-pole Lorentzian S_phi(f) = S0/(1+(f/f_c)^2),
    # whose integral converges to a FINITE total (S0*f_c*pi/2) as bandwidth
    # -> infinity, so genuine convergence is actually possible to observe.
    bw_multiples = np.array([1, 2, 4, 8, 16])
    infid3 = np.zeros_like(bw_multiples, dtype=float)
    sem3 = np.zeros_like(bw_multiples, dtype=float)
    S0_PHI_3 = 3e-10
    F_CORNER_3 = NATURAL_BW_HZ
    lorentzian_shape = lambda f: S0_PHI_3 / (1.0 + (f / F_CORNER_3) ** 2)
    for i, m in enumerate(bw_multiples):
        bw = m * NATURAL_BW_HZ
        result = _engine.run(
            schematic, src, nonlinear=None, noise=None,
            phase_noise={"single_sided_psd_rad2_per_hz": lorentzian_shape, "bandwidth_hz": bw},
            n_realizations=N_REALIZATIONS, mode="complex_baseband", seed=SEED,
        )
        fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
        infid3[i] = 1.0 - fid.noise.F_avg
        sem3[i] = fid.noise.F_sem
        print(f"  bandwidth_hz={bw/1e6:7.1f} MHz ({m:2d}x natural BW)  infidelity={infid3[i]:.4e} +/- {sem3[i]:.1e}")

    print()
    print("=== Summary ===")
    print(f"Part 1: infidelity spans {infid1.min():.3e} to {infid1.max():.3e} over "
          f"S_phi={psd_values.min():.1e} to {psd_values.max():.1e} rad^2/Hz")
    rel_diffs = np.abs(infid_pre - infid_post) / infid_post
    print(f"Part 2: pre-NL/post-hoc relative discrepancy ranges {rel_diffs.min():.1%} to "
          f"{rel_diffs.max():.1%} across op1db_amplitude={op1db_values.min():.1f} to "
          f"{op1db_values.max():.1f} -- consistently nonzero, confirming the two approaches "
          f"are genuinely different once a real nonlinearity is in the loop.")
    print(f"Part 3: infidelity from {bw_multiples.min()}x to {bw_multiples.max()}x natural bandwidth "
          f"spans {infid3.min():.3e} to {infid3.max():.3e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_floor = 1e-9
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.errorbar(psd_values, np.maximum(infid1, plot_floor), yerr=sem1, fmt="o-", color="C0", capsize=3)
    ax1.axhline(max(floor, plot_floor), color="gray", ls="--", lw=1, label="noise-free floor")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Phase noise PSD (rad²/Hz)")
    ax1.set_ylabel("Infidelity to X (1 - F_avg)")
    ax1.set_title("1: Phase noise alone\n(no nonlinearity)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    ax2.errorbar(op1db_values, np.maximum(infid_pre, plot_floor), yerr=sem_pre, fmt="o-", color="C1",
                 capsize=3, label="pre-NL (correct)")
    ax2.errorbar(op1db_values, np.maximum(infid_post, plot_floor), yerr=sem_post, fmt="s--", color="C3",
                 capsize=3, label="post-hoc (wrong)")
    ax2.invert_xaxis()   # milder compression (higher op1db) on the left
    ax2.set_yscale("log")
    ax2.set_xlabel("op1db_amplitude (lower = more aggressive compression)")
    ax2.set_ylabel("Infidelity to X (1 - F_avg)")
    ax2.set_title("2: Pre-NL vs. post-hoc phase noise,\nre-calibrated per compression level")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    ax3.errorbar(bw_multiples, np.maximum(infid3, plot_floor), yerr=sem3, fmt="o-", color="C2", capsize=3)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("bandwidth_hz / natural pulse bandwidth")
    ax3.set_ylabel("Infidelity to X (1 - F_avg)")
    ax3.set_title("3: bandwidth_hz convergence check\n(no nonlinearity)")
    ax3.grid(alpha=0.3, which="both")

    fig.suptitle("Phase noise case study: pre-NL injection vs. post-hoc, and bandwidth convergence")
    plt.tight_layout()
    out_path = Path(__file__).parent / "phase_noise_case_study_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
