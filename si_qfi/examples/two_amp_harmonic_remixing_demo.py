"""
examples/two_amp_harmonic_remixing_demo.py
============================================
Follow-up investigation to examples/nonlinearity_fidelity_demo.py's finding
that pure AM-AM never limits a single amplifier stage's gate fidelity
(recalibration recovers F~1 exactly, right up to a hard achievability
cliff). Question: does that conclusion still hold with TWO cascaded AM-AM
stages in real_axis mode?

The physical concern: a real, memoryless, odd nonlinearity (Saleh AM-AM,
no phase term) driven by a narrowband single-tone-like signal generates a
3rd-harmonic image near 3*f_carrier alongside the in-band (near
f_carrier) distortion -- this is exactly the mechanism behind the 4/3
OIP3 factor documented in nonlinear/saleh.py. For ONE amplifier stage,
that 3rd-harmonic image is far out of band and irrelevant to the qubit
(matches examples/nonlinearity_fidelity_demo.py Panel C: single-stage
real_axis AM-AM tracks the "recalibrate -> F~1 or a hard cliff" pattern
exactly). But with a SECOND stage: expanding a cubic nonlinearity's
response to (near-f_carrier signal + near-3*f_carrier image), the
cross-term 3*x1^2*x3 contains a component near 3*f_carrier - 2*f_carrier
= f_carrier -- i.e. a genuine third-order intermodulation product that
mixes the first stage's 3rd-harmonic energy back down into the qubit's
own operating band at the SECOND stage. This is not a violation of the
"cascaded odd nonlinearities only ever produce odd harmonics of a single
CW tone" proof used elsewhere in this codebase (see
tests/test_engine.py's two-amp cascade harmonic tests) -- f_carrier
itself is the (odd) fundamental; this argument is about what NEW physics
contributes to that same in-band component, which the proof doesn't
constrain.

THE FINDING (verified below, both by hand against the two-stage schematic
directly and through the full engine.run() + quantum.gate_fidelity()
pipeline): the user's hypothesis is correct, but the way it shows up is
more specific and more interesting than a simple "fidelity gets worse":

  Splitting a given total compression across TWO cascaded stages extends
  the ACHIEVABLE operating range (the maximum total pulse area reachable
  via recalibration) well beyond what either stage could reach alone --
  true in BOTH modes, not real_axis-specific, since even complex_baseband
  mode's two independent local AM-AM compressions benefit somewhat from
  distributing the load. But real_axis mode's two-stage case extends
  FURTHER than complex_baseband's two-stage case does (harmonic energy
  genuinely helps carry more total rotation through than the baseband
  picture, which cannot represent that energy at all, predicts is
  possible) -- AND, uniquely to the real_axis two-stage case, there is a
  genuine GRAY ZONE of partial, non-zero, GROWING infidelity between
  "where a single stage's own cliff would have been" and "where the
  two-stage chain's own cliff actually is". Neither complex_baseband
  (either stage count) nor real_axis SINGLE-stage ever show this gray
  zone -- for those, recalibration is either exact (F~1 to numerical
  floor) or flatly impossible, no in-between. This gray zone IS the
  harmonic-remixing signature the user was asking about: it only exists
  where (a) real RF harmonics exist to remix (real_axis mode) AND (b)
  there's a second nonlinear stage downstream to do the remixing.

IMPORTANT CORRECTION (caught by the user questioning an earlier version of
this plot, 2026-07-11): the first version of this demo used a 100 ns pulse
and reported a "floor" around 1e-5 for op1db values well away from either
cliff -- e.g. op1db=1.0 gave ~9e-6 in BOTH single- and two-stage, in BOTH
real_axis and complex_baseband. That floor is real, but it is NOT
numerical, and it is NOT the harmonic-remixing effect either: it's the
schematic's own LOSSY, frequency-dependent transmission lines slightly
distorting the pulse even with NO nonlinearity active at all (confirmed
directly: engine.run() with nonlinear=None on this same schematic gives
the identical ~1e-5 infidelity). This is ordinary linear-channel
dispersion -- the line's phase response isn't perfectly flat across the
pulse's own bandwidth -- and it scales with bandwidth as expected
(halving the pulse bandwidth roughly quarters the infidelity, confirmed
by sweeping pulse duration from 100ns to 1600ns). At the original 100ns
pulse, this baseline floor (~1e-5) was uncomfortably close to the
smallest genuine gray-zone values, muddying the comparison. Fixed by
widening the pulse to 400 ns (bandwidth ~1/4 as large, baseline floor
~13x lower, ~6e-7) so the baseline sits well below where the harmonic-
remixing effect actually begins -- and the baseline itself is now plotted
explicitly (dotted reference line) rather than left for the reader to
infer, exactly BECAUSE conflating a linear dispersion floor with a
nonlinear effect is an easy, real mistake to make (this script made it
once already).

Run: python examples/two_amp_harmonic_remixing_demo.py
Requires: SignalIntegrity, QuTiP, matplotlib.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import qutip
from scipy.signal import fftconvolve

from si_qfi.schematic import loader as si_loader
from si_qfi.schematic import transfer_function as si_tf
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform, build_gaussian_envelope
from si_qfi.nonlinear.saleh import SalehModel, SalehRealAxisModel
from si_qfi import quantum

SCHEMATIC_PATH = (
    Path(__file__).parent.parent / "tests" / "test_schematic_lossy_T_line_2_amplifier.si"
).resolve()
# Topology (see tests/test_engine.py's two-amp cascade section for the full
# characterization): VSource -> R1 -> T3 -> D1(amp) -> DriverOutput ->
# T1(lossy) -> D2(amp) -> DriverOutput2 -> T2(lossy) -> R2 -> VQubit.
# NL nodes: DriverOutput (after amp 1) and DriverOutput2 (after amp 2).
NODE_1, NODE_2 = "DriverOutput", "DriverOutput2"

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6
# 400 ns, not 100 -- narrower pulse bandwidth means less linear-dispersion
# distortion from this schematic's own lossy lines, pushing the baseline
# (no-NL) infidelity floor from ~1e-5 down to ~6e-7, well below where the
# genuine harmonic-remixing effect begins. See the "IMPORTANT CORRECTION"
# note in the module docstring above.
DURATION_S = 400e-9
SIGMA_S = DURATION_S / 6
FS_ENVELOPE = 8e9        # see nonlinearity_fidelity_demo.py -- 2 GSa/s left a
                          # spurious ~1e-6 numerical (not physical) infidelity
                          # from under-resolving compression-reshaped pulses
LPF_CUTOFF_HZ = 500e6


# ---------------------------------------------------------------------------
# Two-stage calibration (generalizes nonlinearity_fidelity_demo.py's
# single-node calibrate_and_run to a chain of up to two NL nodes; either
# may be None for a "single-stage" comparison run through the SAME
# schematic/channel).
# ---------------------------------------------------------------------------

def _source_from_shape(shape: np.ndarray, fs: float, carrier_ghz: float) -> SourceWaveform:
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def _pre_nl_waveform(schematic, source, mode, mid_label=NODE_1):
    raw = si_tf._extract_single_tf(schematic.si_app, schematic.source_label, mid_label, schematic.source_label)
    h = si_tf.compute_impulse_response(raw, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz).h
    if mode == "real_axis":
        fs_native = si_tf.native_sample_rate(raw)
        _, v_initial = source.rf_waveform_at(fs_native)
        fs_out = fs_native
    else:
        v_initial = source.envelope_complex
        fs_out = source.fs
    return np.convolve(v_initial, h, mode="full"), fs_out


def _segment_h(schematic, source, mode, label_in, label_out):
    raw = si_tf._extract_single_tf(schematic.si_app, label_in, label_out, schematic.source_label)
    return si_tf.compute_impulse_response(raw, mode, fs=source.fs, carrier_hz=source.carrier_freq_hz).h


def calibrate_and_run(
    schematic, mode, nl1_fn, nl2_fn, target_theta=np.pi,
    lpf_cutoff_hz=None, scale_lo=1e-4, scale_hi=1000.0, n_scan=80, n_bisect=35,
):
    """
    Self-calibrate a Gaussian I-only pulse to hit target_theta radians of
    rotation through UP TO TWO cascaded nonlinear stages (nl1_fn at
    NODE_1, nl2_fn at NODE_2 -- either may be None to model only one
    stage being nonlinear, e.g. for a "single-stage" comparison run
    through this SAME two-amplifier schematic/channel), then runs the
    full engine.run() pipeline once at the calibrated amplitude.

    Returns (result_or_None, achieved: bool, theta_hit_or_max_achievable).
    Same non-monotonic-theta caveat as nonlinearity_fidelity_demo.py's
    calibrate_and_run() -- coarse scan brackets the rising branch first.
    """
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, FS_ENVELOPE, CARRIER_GHZ)
    v_pre1, fs_pre = _pre_nl_waveform(schematic, source_ref, mode, NODE_1)
    h_mid = _segment_h(schematic, source_ref, mode, NODE_1, NODE_2)
    h_post = _segment_h(schematic, source_ref, mode, NODE_2, schematic.qubit_probe_label)

    def theta_for_scale(scale):
        nl1 = nl1_fn() if nl1_fn else None
        nl2 = nl2_fn() if nl2_fn else None
        if mode == "complex_baseband":
            v_nl1 = nl1.apply_baseband(v_pre1 * scale) if nl1 else v_pre1 * scale
            v_pre2 = fftconvolve(v_nl1, h_mid, mode="full")
            v_nl2 = nl2.apply_baseband(v_pre2) if nl2 else v_pre2
        else:
            v_nl1 = nl1.apply_real_axis(v_pre1 * scale) if nl1 else v_pre1 * scale
            v_pre2 = fftconvolve(v_nl1, h_mid, mode="full")
            v_nl2 = nl2.apply_real_axis(v_pre2) if nl2 else v_pre2
        v_post = fftconvolve(v_nl2, h_post, mode="full")
        t = np.arange(len(v_post)) / fs_pre
        if mode == "complex_baseband":
            env_i = np.real(v_post)
        else:
            env_i, _ = quantum.demodulate(v_post, t, CARRIER_GHZ * 1e9, lpf_cutoff_hz)
        return float(ETA * np.trapz(env_i, t)), v_post

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scales = np.geomspace(scale_lo, scale_hi, n_scan)
        thetas = np.array([theta_for_scale(s)[0] for s in scales])
        if thetas[int(np.argmax(thetas >= target_theta))] < target_theta:
            return None, False, float(thetas.max())

        idx = int(np.argmax(thetas >= target_theta))
        lo, hi = scales[max(idx - 1, 0)], scales[idx]
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            th, _ = theta_for_scale(mid)
            if th < target_theta:
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
        theta_hit, v_post_hit = theta_for_scale(scale)

    cal_shape = ref_shape * scale
    source_cal = _source_from_shape(cal_shape, FS_ENVELOPE, CARRIER_GHZ)
    nl_spec = {}
    if nl1_fn:
        nl_spec[NODE_1] = nl1_fn.spec()
    if nl2_fn:
        nl_spec[NODE_2] = nl2_fn.spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = engine.run(
            schematic, source_cal, nonlinear=nl_spec if nl_spec else None,
            noise=None, n_realizations=1, mode=mode,
        )
    return result, True, (theta_hit, v_post_hit)


class SalehFn:
    """nl_model_fn for either baseband SalehModel or real-axis SalehRealAxisModel."""
    def __init__(self, op1db, cls=SalehModel):
        self.op1db, self.cls = op1db, cls

    def __call__(self):
        return self.cls.from_op1db_oip3(op1db_amplitude=self.op1db)

    def spec(self):
        return {"model": "saleh", "op1db_amplitude": self.op1db}


def fidelity_at(schematic, mode, nl1_fn, nl2_fn, qmodel, lpf_cutoff_hz=None):
    result, achieved, extra = calibrate_and_run(schematic, mode, nl1_fn, nl2_fn, lpf_cutoff_hz=lpf_cutoff_hz)
    if not achieved:
        return None, False
    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X", lpf_cutoff_hz=lpf_cutoff_hz,
    )
    return 1.0 - fid.F_avg, True


def baseline_infidelity(schematic, mode, qmodel, lpf_cutoff_hz=None):
    """
    Infidelity with NO nonlinearity at all -- purely from this schematic's
    own linear (lossy, dispersive) transmission lines. Computed once and
    plotted as an explicit reference so a reader can't mistake ordinary
    channel dispersion for a nonlinear/harmonic-remixing effect (see the
    module docstring's "IMPORTANT CORRECTION").
    """
    infid, achieved = fidelity_at(schematic, mode, None, None, qmodel, lpf_cutoff_hz)
    assert achieved, "the no-NL baseline should always be trivially achievable"
    return infid


def main():
    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    qmodel = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

    print("Measuring no-NL baseline (linear channel dispersion only)...")
    baseline_ra = baseline_infidelity(schematic, "real_axis", qmodel, LPF_CUTOFF_HZ)
    baseline_bb = baseline_infidelity(schematic, "complex_baseband", qmodel)
    print(f"  real_axis baseline:        {baseline_ra:.3e}")
    print(f"  complex_baseband baseline: {baseline_bb:.3e}")

    # Range chosen from direct exploration (see module docstring): at 400ns,
    # real_axis single-stage cliff ~0.18-0.20, two-stage cliff ~0.14-0.15;
    # baseband single-stage cliff ~0.22-0.25, two-stage cliff ~0.16-0.18.
    op1db_sweep = np.array([0.50, 0.40, 0.32, 0.28, 0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.15, 0.14, 0.12, 0.1])

    print("Sweeping real_axis: single-stage (node 2 only) vs two-stage (both nodes)...")
    infid_ra_single, ok_ra_single = [], []
    infid_ra_two, ok_ra_two = [], []
    for op1db in op1db_sweep:
        i1, ok1 = fidelity_at(schematic, "real_axis", None, SalehFn(op1db, SalehRealAxisModel), qmodel, LPF_CUTOFF_HZ)
        infid_ra_single.append(i1); ok_ra_single.append(ok1)
        i2, ok2 = fidelity_at(schematic, "real_axis", SalehFn(op1db, SalehRealAxisModel),
                               SalehFn(op1db, SalehRealAxisModel), qmodel, LPF_CUTOFF_HZ)
        infid_ra_two.append(i2); ok_ra_two.append(ok2)

    print("Sweeping complex_baseband: single-stage vs two-stage (reference/comparison)...")
    infid_bb_single, ok_bb_single = [], []
    infid_bb_two, ok_bb_two = [], []
    for op1db in op1db_sweep:
        i1, ok1 = fidelity_at(schematic, "complex_baseband", None, SalehFn(op1db), qmodel)
        infid_bb_single.append(i1); ok_bb_single.append(ok1)
        i2, ok2 = fidelity_at(schematic, "complex_baseband", SalehFn(op1db), SalehFn(op1db), qmodel)
        infid_bb_two.append(i2); ok_bb_two.append(ok2)

    arrs = {}
    for name, vals, oks in [
        ("ra_single", infid_ra_single, ok_ra_single), ("ra_two", infid_ra_two, ok_ra_two),
        ("bb_single", infid_bb_single, ok_bb_single), ("bb_two", infid_bb_two, ok_bb_two),
    ]:
        arrs[name] = np.array([v if v is not None else np.nan for v in vals])
        arrs[name + "_ok"] = np.array(oks)

    def cliff(ok_mask):
        return op1db_sweep[ok_mask].min() if ok_mask.any() else float("nan")

    print()
    print("=== Achievability cliffs (smallest op1db still reaching a full pi pulse) ===")
    print(f"  real_axis,  single-stage: op1db >= {cliff(arrs['ra_single_ok']):.2f}")
    print(f"  real_axis,  two-stage:    op1db >= {cliff(arrs['ra_two_ok']):.2f}")
    print(f"  baseband,   single-stage: op1db >= {cliff(arrs['bb_single_ok']):.2f}")
    print(f"  baseband,   two-stage:    op1db >= {cliff(arrs['bb_two_ok']):.2f}")

    # FFT snapshot at a representative "two-stage-only-achievable" point,
    # to visualize the 3rd-harmonic content physically responsible.
    illustrative_op1db = op1db_sweep[arrs["ra_two_ok"] & ~arrs["ra_single_ok"]].max() \
        if (arrs["ra_two_ok"] & ~arrs["ra_single_ok"]).any() else op1db_sweep[arrs["ra_two_ok"]].min()
    result_illustrative, ok_ill, _ = calibrate_and_run(
        schematic, "real_axis", SalehFn(illustrative_op1db, SalehRealAxisModel),
        SalehFn(illustrative_op1db, SalehRealAxisModel), lpf_cutoff_hz=LPF_CUTOFF_HZ,
    )
    if ok_ill:
        v_final = result_illustrative.v_nl_qubit
        fs_final = result_illustrative.fs
        freqs = np.fft.rfftfreq(len(v_final), 1.0 / fs_final)
        spectrum = np.abs(np.fft.rfft(v_final))

    import matplotlib
    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    floor = 1e-11

    ax1.semilogx(op1db_sweep[arrs["ra_two_ok"]], np.maximum(arrs["ra_two"][arrs["ra_two_ok"]], floor),
                 "o-", color="C0", label="two-stage")
    ax1.semilogx(op1db_sweep[arrs["bb_two_ok"]], np.maximum(arrs["bb_two"][arrs["bb_two_ok"]], floor),
                 "s--", color="C1", label="two-stage (baseband, no harmonics)")
    ax1.axhline(baseline_ra, color="C0", ls=":", lw=1, label="real_axis baseline (no NL)")
    ax1.axhline(baseline_bb, color="C1", ls=":", lw=1, label="baseband baseline (no NL)")
    ax1.set_yscale("log")
    ax1.set_xlabel("op1db_amplitude (same at both nodes)")
    ax1.set_ylabel("Infidelity to X (1 - F_avg)")
    ax1.set_title("A: Two-stage cascade\nreal_axis vs baseband")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2.semilogx(op1db_sweep[arrs["ra_single_ok"]], np.maximum(arrs["ra_single"][arrs["ra_single_ok"]], floor),
                 "o-", color="C2", label="single-stage (node 2 only)")
    ax2.semilogx(op1db_sweep[arrs["ra_two_ok"]], np.maximum(arrs["ra_two"][arrs["ra_two_ok"]], floor),
                 "o-", color="C0", label="two-stage (both nodes)")
    ax2.axhline(baseline_ra, color="gray", ls=":", lw=1, label="no-NL baseline\n(linear dispersion only)")
    ax2.axvspan(op1db_sweep.min(), cliff(arrs["ra_single_ok"]), color="red", alpha=0.08,
                label="gray zone: two-stage-only\nachievable, real infidelity")
    ax2.set_yscale("log")
    ax2.set_xlabel("op1db_amplitude")
    ax2.set_ylabel("Infidelity to X (1 - F_avg)")
    ax2.set_title("B: real_axis ONLY\nsingle- vs two-stage")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    if ok_ill:
        fc = CARRIER_GHZ * 1e9
        ax3.semilogy(freqs / 1e9, spectrum + 1e-300)
        for k, label in [(1, "f_c"), (3, "3 f_c")]:
            ax3.axvline(k * fc / 1e9, color="red", ls=":", lw=1)
            ax3.text(k * fc / 1e9, ax3.get_ylim()[1], label, ha="center", va="bottom", fontsize=8, color="red")
        ax3.set_xlim(0, 4 * CARRIER_GHZ)
        ax3.set_xlabel("Frequency (GHz)")
        ax3.set_ylabel("|FFT| (a.u., log)")
        ax3.set_title(f"C: Qubit-plane spectrum\n(two-stage, op1db={illustrative_op1db}, gray zone)")
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "two_amp_harmonic_remixing_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
