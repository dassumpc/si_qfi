"""
examples/nonlinearity_fidelity_demo.py
=======================================
Investigation tool: in what regime does a real amplifier nonlinearity
actually limit two-level qubit gate fidelity? Is it AM-AM (real gain
compression) alone, only AM-PM (phase-vs-amplitude distortion), or only
once you're driving near/beyond the 1dB compression point?

Uses the same SI-QFI -> QuTiP pipeline as examples/rabi_oscillation_demo.py
(tests/test_schematic_basic.si, a plain resonant 2-level qubit), but now
with a REAL nonlinear node (Saleh AM-AM/AM-PM in complex_baseband mode;
Saleh real-axis and Volterra -- both AM-AM only, see nonlinear/saleh.py and
nonlinear/volterra.py module docstrings -- in real_axis mode) at
"DriverOutput", swept across compression depth (op1db_amplitude) and, for
the baseband Saleh case, AM-PM peak phase.

At every sweep point, the drive amplitude is SELF-CALIBRATED through the
actual nonlinearity (see calibrate_and_run() below) to hit exactly a pi
pulse area on the I axis -- i.e. "what fidelity do you get if you correctly
recalibrate your pulse for the amplifier you actually have", not "what
fidelity do you get if you assumed a linear amplifier and didn't
recalibrate". Both are legitimate questions; this script answers the first
one, which isolates whatever DISTORTION (not just miscalibration) the
nonlinearity introduces.

Run: python examples/nonlinearity_fidelity_demo.py
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
from si_qfi.nonlinear.volterra import VolterraModel
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_basic.si").resolve()

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6
DURATION_S = 100e-9
SIGMA_S = DURATION_S / 6
# 8 GSa/s, not 2 -- deep AM-AM compression reshapes the pulse into a
# sharper, more flat-topped waveform (compresses the peak much more than
# the low-amplitude edges) that a coarser envelope grid under-resolves for
# QuTiP's cubic-spline/ODE solve. Confirmed directly: at 2 GSa/s, deep
# compression alone (no AM-PM at all) produced a spurious ~1e-6 infidelity
# that is NUMERICAL, not physical -- it shrinks to ~1e-11 (float64 floor)
# at 8 GSa/s and stays there, with no further physics to find. Without
# this, Panel A's "pure AM-AM never limits fidelity" story would show a
# fake bump that looks like real physics but isn't.
FS_ENVELOPE = 8e9
LPF_CUTOFF_HZ = 500e6
NL_LABEL = "DriverOutput"


# ---------------------------------------------------------------------------
# Calibration machinery (see tests/test_quantum_nonlinear.py for the full
# rationale -- theta(scale) is NOT monotonic for a Saleh/Volterra AM-AM
# curve pushed far enough, so this brackets the RISING branch via a coarse
# scan before bisecting, and reports "not achievable" rather than
# converging to a wrong point if the target is out of reach entirely).
# ---------------------------------------------------------------------------

def _source_from_shape(shape: np.ndarray, fs: float, carrier_ghz: float) -> SourceWaveform:
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def _pre_nl_waveform(schematic, source, mode, mid_label=NL_LABEL):
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
    schematic, mode, nl_model_fn, target_theta=np.pi,
    lpf_cutoff_hz=None, scale_lo=1e-4, scale_hi=500.0, n_scan=60, n_bisect=30,
):
    """
    Self-calibrate a Gaussian I-only pulse to hit target_theta radians of
    rotation through nl_model_fn()'s actual nonlinearity, then run the full
    engine.run() pipeline once at that amplitude.

    Returns (result_or_None, achieved: bool, theta_hit_or_max_achievable).
    """
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, FS_ENVELOPE, CARRIER_GHZ)
    v_pre, fs_pre = _pre_nl_waveform(schematic, source_ref, mode)
    h_post = _segment_h(schematic, source_ref, mode, NL_LABEL, schematic.qubit_probe_label)

    def theta_for_scale(scale):
        nl_model = nl_model_fn()
        if mode == "complex_baseband":
            v_nl = nl_model.apply_baseband(v_pre * scale)
        else:
            v_nl = nl_model.apply_real_axis(v_pre * scale)
        v_post = fftconvolve(v_nl, h_post, mode="full")
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
        achievable_max = float(np.max(thetas))
        idx = int(np.argmax(thetas >= target_theta))
        if thetas[idx] < target_theta:
            return None, False, achievable_max

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

    peak_at_nl = float(np.max(np.abs(v_pre * scale)))

    cal_shape = ref_shape * scale
    source_cal = _source_from_shape(cal_shape, FS_ENVELOPE, CARRIER_GHZ)
    nl_spec = {NL_LABEL: nl_model_fn.spec()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = engine.run(
            schematic, source_cal, nonlinear=nl_spec, noise=None, n_realizations=1, mode=mode,
        )
    return result, True, (theta_hit, peak_at_nl)


class SalehSpec:
    def __init__(self, op1db, enable_am_pm=False, am_pm_peak_deg=0.0):
        self.op1db, self.enable_am_pm, self.am_pm_peak_deg = op1db, enable_am_pm, am_pm_peak_deg

    def __call__(self):
        return SalehModel.from_op1db_oip3(
            op1db_amplitude=self.op1db,
            enable_am_pm=self.enable_am_pm, am_pm_peak_deg=self.am_pm_peak_deg,
        )

    def spec(self):
        d = {"model": "saleh", "op1db_amplitude": self.op1db}
        if self.enable_am_pm:
            d["enable_am_pm"] = True
            d["am_pm_peak_deg"] = self.am_pm_peak_deg
        return d


class SalehRealAxisSpec:
    def __init__(self, op1db):
        self.op1db = op1db

    def __call__(self):
        return SalehRealAxisModel.from_op1db_oip3(op1db_amplitude=self.op1db)

    def spec(self):
        return {"model": "saleh", "op1db_amplitude": self.op1db}


class VolterraSpec:
    def __init__(self, op1db):
        self.op1db = op1db

    def __call__(self):
        return VolterraModel(option="describing", op1db_amplitude=self.op1db, memory_depth=0)

    def spec(self):
        return {"model": "volterra", "option": "describing",
                "op1db_amplitude": self.op1db, "memory_depth": 0}


def fidelity_at(schematic, mode, nl_model_fn, qmodel, lpf_cutoff_hz=None):
    """Returns (infidelity, achieved, peak_dB_above_op1db) for one sweep point."""
    result, achieved, extra = calibrate_and_run(schematic, mode, nl_model_fn, lpf_cutoff_hz=lpf_cutoff_hz)
    if not achieved:
        return None, False, None
    theta_hit, peak_at_nl = extra
    fid = quantum.gate_fidelity(
        result, qmodel, coupling_strength_per_volt=ETA, ideal_gate="X",
        lpf_cutoff_hz=lpf_cutoff_hz,
    )
    depth_db = 20 * np.log10(peak_at_nl / nl_model_fn.op1db)
    return 1.0 - fid.F_avg, True, depth_db


def main():
    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    qmodel = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

    # ------------------------------------------------------------------
    # Panel A: baseband, pure AM-AM (no AM-PM), sweep compression severity
    # (op1db_amplitude) -- prediction: infidelity ~0 wherever achievable,
    # then a hard cliff to "not achievable at all" below some critical
    # op1db (see tests/test_quantum_nonlinear.py's module docstring for
    # the physics reasoning).
    # ------------------------------------------------------------------
    print("Panel A: baseband AM-AM only vs. compression severity...")
    op1db_sweep = np.geomspace(0.2, 15.0, 24)
    infid_A, achieved_A, depth_A = [], [], []
    for op1db in op1db_sweep:
        fn = SalehSpec(op1db, enable_am_pm=False)
        infid, ok, depth = fidelity_at(schematic, "complex_baseband", fn, qmodel)
        infid_A.append(infid); achieved_A.append(ok); depth_A.append(depth)
    infid_A = np.array([v if v is not None else np.nan for v in infid_A])
    achieved_A = np.array(achieved_A)

    # ------------------------------------------------------------------
    # Panel B: baseband, AM-AM + AM-PM, sweep AM-PM peak phase at three
    # fixed (achievable) compression depths -- prediction: infidelity
    # grows smoothly/monotonically with AM-PM depth, worse for deeper
    # compression.
    # ------------------------------------------------------------------
    print("Panel B: baseband AM-PM sweep at three compression depths...")
    am_pm_sweep = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    op1db_levels = [10.0, 1.5, 1.0]
    infid_B = {}
    for op1db in op1db_levels:
        vals = []
        for peak_deg in am_pm_sweep:
            fn = SalehSpec(op1db, enable_am_pm=(peak_deg > 0), am_pm_peak_deg=float(peak_deg))
            infid, ok, _ = fidelity_at(schematic, "complex_baseband", fn, qmodel)
            vals.append(infid if ok else np.nan)
        infid_B[op1db] = np.array(vals)

    # ------------------------------------------------------------------
    # Panel C: real_axis, AM-AM only (Saleh real-axis vs. Volterra cubic)
    # -- same sweep/prediction as Panel A, different model shapes, no
    # AM-PM mechanism available in real_axis mode at all in this codebase.
    # ------------------------------------------------------------------
    print("Panel C: real_axis AM-AM only (Saleh real-axis vs. Volterra)...")
    infid_C_saleh, achieved_C_saleh = [], []
    infid_C_volt, achieved_C_volt = [], []
    for op1db in op1db_sweep:
        infid_s, ok_s, _ = fidelity_at(schematic, "real_axis", SalehRealAxisSpec(op1db), qmodel, LPF_CUTOFF_HZ)
        infid_C_saleh.append(infid_s); achieved_C_saleh.append(ok_s)
        infid_v, ok_v, _ = fidelity_at(schematic, "real_axis", VolterraSpec(op1db), qmodel, LPF_CUTOFF_HZ)
        infid_C_volt.append(infid_v); achieved_C_volt.append(ok_v)
    infid_C_saleh = np.array([v if v is not None else np.nan for v in infid_C_saleh])
    infid_C_volt = np.array([v if v is not None else np.nan for v in infid_C_volt])
    achieved_C_saleh = np.array(achieved_C_saleh)
    achieved_C_volt = np.array(achieved_C_volt)

    # ------------------------------------------------------------------
    # Report + plot
    # ------------------------------------------------------------------
    floor = 1e-11
    print()
    print("=== Summary ===")
    print(f"Panel A (baseband AM-AM only): achievable for op1db >= "
          f"{op1db_sweep[achieved_A][0]:.3f} "
          f"(critical op1db between {op1db_sweep[~achieved_A][-1] if (~achieved_A).any() else float('nan'):.3f} "
          f"and {op1db_sweep[achieved_A][0]:.3f}); "
          f"max infidelity where achievable: {np.nanmax(infid_A):.2e}")
    print(f"Panel B (AM-PM, op1db=1.5): infidelity {infid_B[1.5][0]:.2e} at 0deg -> "
          f"{infid_B[1.5][-1]:.2e} at {am_pm_sweep[-1]:.0f}deg")
    print(f"Panel C (real_axis AM-AM only): Saleh achievable from op1db>="
          f"{op1db_sweep[achieved_C_saleh][0]:.3f}, Volterra achievable from op1db>="
          f"{op1db_sweep[achieved_C_volt][0]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 5))

    axA.semilogx(op1db_sweep[achieved_A], np.maximum(infid_A[achieved_A], floor), "o-", color="C0", label="achievable")
    if (~achieved_A).any():
        axA.axvspan(op1db_sweep.min(), op1db_sweep[~achieved_A].max(), color="red", alpha=0.08,
                    label="NOT achievable\n(no amplitude reaches a full pi pulse)")
    axA.set_yscale("log")
    axA.set_xlabel("op1db_amplitude (severity of compression)")
    axA.set_ylabel("Infidelity to X (1 - F_avg)")
    axA.set_title("A: Baseband, AM-AM ONLY\n(recalibrated each point)")
    axA.legend(fontsize=8)
    axA.grid(alpha=0.3)

    for op1db in op1db_levels:
        axB.semilogy(am_pm_sweep, np.maximum(infid_B[op1db], floor), "o-", label=f"op1db={op1db}")
    axB.set_xlabel("AM-PM peak phase (deg)")
    axB.set_ylabel("Infidelity to X (1 - F_avg)")
    axB.set_title("B: Baseband, AM-AM + AM-PM\n(I-axis recalibrated each point)")
    axB.legend(fontsize=8)
    axB.grid(alpha=0.3)

    axC.semilogx(op1db_sweep[achieved_C_saleh], np.maximum(infid_C_saleh[achieved_C_saleh], floor),
                 "o-", label="SalehRealAxisModel")
    axC.semilogx(op1db_sweep[achieved_C_volt], np.maximum(infid_C_volt[achieved_C_volt], floor),
                 "x-", label="VolterraModel (cubic)")
    if (~achieved_C_saleh).any():
        axC.axvspan(op1db_sweep.min(), op1db_sweep[~achieved_C_saleh].max(), color="C0", alpha=0.10,
                    label="NOT achievable (Saleh RA)")
    if (~achieved_C_volt).any():
        axC.axvspan(op1db_sweep.min(), op1db_sweep[~achieved_C_volt].max(), color="C1", alpha=0.10,
                    label="NOT achievable (Volterra)")
    axC.set_yscale("log")
    axC.set_xlabel("op1db_amplitude (severity of compression)")
    axC.set_ylabel("Infidelity to X (1 - F_avg)")
    axC.set_title("C: Real-axis, AM-AM ONLY\n(no AM-PM mechanism exists here)")
    axC.legend(fontsize=8)
    axC.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "nonlinearity_fidelity_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
