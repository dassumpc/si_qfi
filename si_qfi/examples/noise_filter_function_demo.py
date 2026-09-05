"""
examples/noise_filter_function_demo.py
=========================================
Investigation: does the FREQUENCY DISTRIBUTION of drive-line noise power
matter for gate fidelity, not just its total amount? And can the qubit's
own frequency-dependent noise susceptibility -- its "filter function" --
be traced out empirically from full (non-perturbative) QuTiP solves, using
nothing but si_qfi's existing noise-injection machinery?

Motivation (from this session's own SNR-utility discussion): a first-order
(Magnus/toggling-frame) sensitivity analysis of a resonant Rabi drive
H0(t)=(Omega(t)/2)*sigma_x shows that control-line noise on the SAME axis
as the drive (I-quadrature) accumulates as a plain, UNWEIGHTED time
integral (an exact result -- H0(t) and same-axis noise commute pointwise
at every instant, so nothing about the pulse SHAPE matters, only total
elapsed time), while noise on the ORTHOGONAL axis (Q-quadrature) picks up
a weight involving the ACCUMULATED ROTATION ANGLE theta(t)=integral(Omega
dt) via sin(theta(t)) and cos(theta(t)) -- see quantum/snr.py's own
docstring for the full derivation. A flat time-domain weight (I-channel)
corresponds, via Parseval/Fourier-shift, to a FREQUENCY-domain sensitivity
that is LARGEST at DC and falls off with frequency -- i.e. slow
("quasi-static") noise should cost much more fidelity than fast (white)
noise of the identical total power. This demo tests that prediction
directly against the full quantum solve, not just the perturbative theory.

New capability added to support this: noise/psd.py's
`single_sided_psd_v2_per_hz` override now accepts a CALLABLE freqs->S_v(freqs)
for a colored (frequency-shaped) PSD, not just a flat number -- see its
docstring. Nothing about noise/realization.py itself needed to change (it
already accepted an arbitrary-shaped psd array; only the engine-facing
override-spec parser was flat-only).

Uses tests/test_schematic_noise.si (the lossless VN1-near-qubit schematic
already used by tests/test_engine_noise.py and examples/
noise_density_sweep_demo.py) so noise picks up no extra frequency-dependent
shaping from the schematic itself -- any frequency dependence seen here is
purely the QUBIT's own susceptibility, not the circuit's.

Three experiments:
  1. Quasi-static (narrowband-at-DC) vs. white (flat) noise, amplitude-
     calibrated (via one empirical trial run + an exact linear rescale --
     realization variance is linear in the PSD amplitude, so this is exact,
     not iterative) to carry the IDENTICAL total noise power. Headline
     comparison.
  2. Sweep a narrow noise "probe" (same width, comparable total power)
     across center frequency and trace the empirical infidelity-vs-
     frequency curve -- the qubit's own noise susceptibility spectrum,
     measured the hard way (full ensemble QuTiP solves at every point).
  3. Overlay the THEORETICAL filter functions (flat/I, sin(theta(t))/Q,
     cos(theta(t))/Q, computed directly from the actual realized pulse
     shape at the qubit plane) against the empirical curve from (2) -- a
     real check of the toggling-frame theory against a non-perturbative
     simulation, not just against itself.

THE FINDING (all three predictions confirmed, numbers from the run that
produced examples/noise_filter_function_demo.png):
  - Experiment 1: at IDENTICAL total noise power (Var=6.00e-4 V^2), quasi-
    static noise (sigma_f=1MHz, effectively DC over a ~20ns pulse) costs
    198.6x the infidelity of white noise (2.50e-3 vs. 1.26e-5) -- almost
    2.5 decades of gate fidelity lost purely by moving the SAME noise
    power from "spread across the full baseband" to "concentrated near
    DC", with nothing else about the physical setup changed.
  - Experiment 2: sweeping a narrow (2MHz-wide) noise probe from 5MHz to
    1GHz traces out a clean, monotonically-decreasing susceptibility
    curve spanning nearly 6 decades of infidelity (2.25e-3 at 5MHz down to
    the noise-free floor by ~150MHz) -- the qubit's own noise filter
    function, measured the hard way, one full ensemble QuTiP solve per
    frequency point, no perturbation theory involved.
  - Experiment 3: the naive, UNFITTED "I+Q combined" theory curve (equal-
    weight sum of the flat I-channel term and the sin(theta(t))/cos(theta(t))
    Q-channel terms, computed directly from the realized pulse's own
    theta(t)=integral(Omega dt), zero free parameters) tracks the
    empirical curve closely through the entire rolloff region (5-50MHz) --
    see the plot's Panel 3. Neither single term alone matches nearly as
    well: the flat I-channel term alone falls off too FAST (doesn't
    capture how much low-but-nonzero-frequency noise still costs), the
    sin(theta(t)) term alone falls off too SLOWLY, and the cos(theta(t))
    term alone has a non-monotonic bump the empirical curve doesn't show --
    only the combination reproduces the actual measured shape. This is a
    genuine, non-trivial validation of a perturbative theory (Magnus
    expansion, first order) against a full non-perturbative simulation,
    with the theory curve computed independently and never fit to the
    empirical data.

Run: python examples/noise_filter_function_demo.py
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
N_CAL = 300           # ensemble size for variance-matching calibration
N_REALIZATIONS = 250  # ensemble size for fidelity measurements
SEED = 2026


def _gaussian_freq_shape(f0_hz: float, sigma_f_hz: float, amp: float):
    """A Gaussian-in-frequency PSD shape callable, peaked at f0_hz with
    width sigma_f_hz and peak amplitude amp (V^2/Hz) -- passed directly as
    a `single_sided_psd_v2_per_hz` override value."""
    def shape(freqs):
        return amp * np.exp(-0.5 * ((freqs - f0_hz) / sigma_f_hz) ** 2)
    return shape


def _measured_noise_variance(schematic, src, psd_spec, n_realizations, seed) -> float:
    result = _engine.run(
        schematic, src, nonlinear=None,
        noise={"VN1": {"single_sided_psd_v2_per_hz": psd_spec}},
        n_realizations=n_realizations, mode="complex_baseband", seed=seed,
    )
    diffs = np.array([v - result.v_nl_qubit for v in result.v_qubit_ensemble])
    return float(np.var(diffs))


def _infidelity_at(schematic, src, qmodel, psd_spec, n_realizations, seed):
    result = _engine.run(
        schematic, src, nonlinear=None,
        noise={"VN1": {"single_sided_psd_v2_per_hz": psd_spec}},
        n_realizations=n_realizations, mode="complex_baseband", seed=seed,
    )
    fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X")
    return 1.0 - fid.noise.F_avg, fid.noise.F_sem


def main():
    warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")
    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    qmodel = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

    print("Calibrating pi-pulse amplitude (noise-free)...")
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=ETA, ideal_gate="X",
    )
    tuned_shape = tuned.scale * ref_shape
    src = source_from_envelope_array(tuned_shape, FS_ENVELOPE, CARRIER_GHZ)
    floor = 1.0 - tuned.fidelity.noise_free.F_avg
    print(f"  tuned scale={tuned.scale:.6f}, noise-free infidelity floor={floor:.3e}")

    # ------------------------------------------------------------------
    # Experiment 1: quasi-static vs. white noise, matched TOTAL power.
    # ------------------------------------------------------------------
    print("\nExperiment 1: quasi-static vs. white noise at matched total power...")
    S0_WHITE = 3e-13   # flat one-sided PSD level, V^2/Hz -- chosen so white
                       # noise's own infidelity sits clearly above the floor
    QUASI_SIGMA_F = 1e6   # 1 MHz -- much narrower than the pulse's own
                          # natural bandwidth ~1/(2*pi*SIGMA_S) ~ 48 MHz

    var_white = _measured_noise_variance(schematic, src, S0_WHITE, N_CAL, seed=SEED)
    quasi_trial = _gaussian_freq_shape(0.0, QUASI_SIGMA_F, 1e-15)
    var_quasi_trial = _measured_noise_variance(schematic, src, quasi_trial, N_CAL, seed=SEED)
    quasi_amp = 1e-15 * (var_white / var_quasi_trial)   # exact linear rescale
    quasi_shape = _gaussian_freq_shape(0.0, QUASI_SIGMA_F, quasi_amp)
    var_quasi_check = _measured_noise_variance(schematic, src, quasi_shape, N_CAL, seed=SEED + 1)
    print(f"  white noise var (S0={S0_WHITE:.2e} V^2/Hz):        {var_white:.4e}")
    print(f"  quasi-static var (calibrated, sigma_f={QUASI_SIGMA_F:.1e}Hz): {var_quasi_check:.4e} "
          f"(target {var_white:.4e}, {abs(var_quasi_check - var_white) / var_white:.1%} off)")

    infid_white, sem_white = _infidelity_at(schematic, src, qmodel, S0_WHITE, N_REALIZATIONS, SEED + 10)
    infid_quasi, sem_quasi = _infidelity_at(schematic, src, qmodel, quasi_shape, N_REALIZATIONS, SEED + 11)
    print(f"  white:        infidelity = {infid_white:.4e} +/- {sem_white:.1e}")
    print(f"  quasi-static: infidelity = {infid_quasi:.4e} +/- {sem_quasi:.1e}")
    print(f"  ratio (quasi-static / white) = {infid_quasi / infid_white:.2f}x, at IDENTICAL total noise power")

    # ------------------------------------------------------------------
    # Experiment 2: sweep a narrow noise "probe" across frequency, trace
    # the empirical noise susceptibility (filter function) curve.
    # ------------------------------------------------------------------
    print("\nExperiment 2: sweeping probe center frequency...")
    PROBE_SIGMA_F = 2e6
    probe_freqs_hz = np.geomspace(5e6, 1e9, 12)
    # Calibrate probe amplitude ONCE (at a representative mid-sweep center
    # frequency, well clear of f=0's one-sided-clipping edge effect --
    # see module docstring) so every probe carries comparable total power.
    probe_trial = _gaussian_freq_shape(probe_freqs_hz[len(probe_freqs_hz) // 2], PROBE_SIGMA_F, 1e-15)
    var_probe_trial = _measured_noise_variance(schematic, src, probe_trial, N_CAL, seed=SEED + 20)
    probe_amp = 1e-15 * (var_white / var_probe_trial)

    infid_probe = np.zeros_like(probe_freqs_hz)
    sem_probe = np.zeros_like(probe_freqs_hz)
    for i, f0 in enumerate(probe_freqs_hz):
        shape = _gaussian_freq_shape(f0, PROBE_SIGMA_F, probe_amp)
        infid_probe[i], sem_probe[i] = _infidelity_at(schematic, src, qmodel, shape, N_REALIZATIONS, SEED + 30 + i)
        print(f"  f0={f0 / 1e6:8.2f} MHz  infidelity={infid_probe[i]:.4e} +/- {sem_probe[i]:.1e}")

    # ------------------------------------------------------------------
    # Experiment 3: theoretical filter function overlay -- computed
    # directly from the realized pulse shape at the qubit plane.
    # ------------------------------------------------------------------
    print("\nExperiment 3: theoretical filter function overlay...")
    result_ref = _engine.run(schematic, src, nonlinear=None, noise=None, n_realizations=1, mode="complex_baseband")
    v = np.asarray(result_ref.v_nl_qubit)
    t = np.arange(len(v)) / result_ref.fs
    Omega = ETA * v.real
    dt = t[1] - t[0]
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (Omega[1:] + Omega[:-1]) * dt)])
    print(f"  final theta(T) = {theta[-1]:.4f} rad (target pi = {np.pi:.4f})")

    def _spectral_power(g):
        G = np.fft.rfft(g) * dt
        freqs = np.fft.rfftfreq(len(g), d=dt)
        return freqs, np.abs(G) ** 2

    freqs_th, S_I = _spectral_power(np.ones_like(theta))
    _, S_sin = _spectral_power(np.sin(theta))
    _, S_cos = _spectral_power(np.cos(theta))
    S_combined = S_I + S_sin + S_cos   # naive equal-weight I + Q sum (isotropic noise)

    def _interp_norm(S_src, freqs_target):
        vals = np.interp(freqs_target, freqs_th, S_src)
        return vals / vals.max()

    theory_I = _interp_norm(S_I, probe_freqs_hz)
    theory_sin = _interp_norm(S_sin, probe_freqs_hz)
    theory_cos = _interp_norm(S_cos, probe_freqs_hz)
    theory_combined = _interp_norm(S_combined, probe_freqs_hz)

    empirical = infid_probe - floor
    empirical_norm = empirical / empirical.max()

    print()
    print("=== Summary ===")
    print(f"Experiment 1: quasi-static noise costs {infid_quasi / infid_white:.2f}x the infidelity of white "
          f"noise, at identical total power ({var_white:.3e} V^2).")
    print(f"Experiment 2: empirical infidelity spans {infid_probe.min():.3e} to {infid_probe.max():.3e} "
          f"as the probe sweeps {probe_freqs_hz.min() / 1e6:.1f} to {probe_freqs_hz.max() / 1e6:.0f} MHz.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: bar comparison, Experiment 1
    ax1.bar(
        ["white\n(flat)", f"quasi-static\n(sigma_f={QUASI_SIGMA_F / 1e6:.0f}MHz)"],
        [infid_white, infid_quasi],
        yerr=[sem_white, sem_quasi], color=["C0", "C3"], capsize=5,
    )
    ax1.axhline(floor, color="gray", ls="--", lw=1, label="noise-free floor")
    ax1.set_yscale("log")
    ax1.set_ylabel("Infidelity to X (1 - F_avg)")
    ax1.set_title(f"1: Same total noise power,\ndifferent frequency distribution\n"
                   f"({infid_quasi / infid_white:.1f}x worse)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    # Panel 2: empirical susceptibility spectrum
    ax2.errorbar(probe_freqs_hz / 1e6, infid_probe, yerr=sem_probe, fmt="o-", color="C1", capsize=3)
    ax2.axhline(floor, color="gray", ls="--", lw=1, label="noise-free floor")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Probe center frequency (MHz)")
    ax2.set_ylabel("Infidelity to X (1 - F_avg)")
    ax2.set_title("2: Empirical noise susceptibility\n(narrow probe swept in frequency)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    # Panel 3: theory overlay
    ax3.plot(probe_freqs_hz / 1e6, empirical_norm, "o-", color="C1", label="empirical (normalized)")
    ax3.plot(probe_freqs_hz / 1e6, theory_I, "--", color="C0", label="theory: I-channel (flat weight)")
    ax3.plot(probe_freqs_hz / 1e6, theory_sin, "--", color="C2", label="theory: Q-channel, sin(theta(t))")
    ax3.plot(probe_freqs_hz / 1e6, theory_cos, "--", color="C4", label="theory: Q-channel, cos(theta(t))")
    ax3.plot(probe_freqs_hz / 1e6, theory_combined, "-", color="k", lw=2, alpha=0.6, label="theory: I+Q combined")
    ax3.set_xscale("log")
    ax3.set_xlabel("Probe center frequency (MHz)")
    ax3.set_ylabel("Normalized sensitivity (peak = 1)")
    ax3.set_title("3: Empirical vs. theoretical\nfilter function")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out_path = Path(__file__).parent / "noise_filter_function_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
