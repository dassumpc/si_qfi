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

from pathlib import Path

import numpy as np
import qutip

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope
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
    """Returns (infidelity, achieved) for one sweep point. tuneup_amplitude()
    reuses this file's existing SalehSpec/SalehRealAxisSpec/VolterraSpec
    classes' .spec() dict output (the same nonlinear= annotation engine.run()
    always took), and reproduces the coarse-scan + bisection-on-the-rising-
    branch strategy needed here (Saleh/Volterra AM-AM makes theta(scale)
    non-monotonic once driven hard enough) internally."""
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    nl_spec = {NL_LABEL: nl_model_fn.spec()}
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=ETA, ideal_gate="X",
        nonlinear=nl_spec, mode=mode, lpf_cutoff_hz=lpf_cutoff_hz,
    )
    if not tuned.achieved:
        return None, False
    return 1.0 - tuned.fidelity.noise_free.F_avg, True


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
    infid_A, achieved_A = [], []
    for op1db in op1db_sweep:
        fn = SalehSpec(op1db, enable_am_pm=False)
        infid, ok = fidelity_at(schematic, "complex_baseband", fn, qmodel)
        infid_A.append(infid); achieved_A.append(ok)
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
            infid, ok = fidelity_at(schematic, "complex_baseband", fn, qmodel)
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
        infid_s, ok_s = fidelity_at(schematic, "real_axis", SalehRealAxisSpec(op1db), qmodel, LPF_CUTOFF_HZ)
        infid_C_saleh.append(infid_s); achieved_C_saleh.append(ok_s)
        infid_v, ok_v = fidelity_at(schematic, "real_axis", VolterraSpec(op1db), qmodel, LPF_CUTOFF_HZ)
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
