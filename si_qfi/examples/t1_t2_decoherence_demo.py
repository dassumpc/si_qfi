"""
examples/t1_t2_decoherence_demo.py
===================================
Every prior demo in this codebase (and transmon_leakage_demo.py) simulates
a perfectly ISOLATED qubit -- no intrinsic decoherence, only drive-chain
imperfections (nonlinearity, dispersion, reflections) or qubit-model
effects (leakage). `quantum.gate_fidelity()` has supported intrinsic T1/T2
via Lindblad collapse operators (T1_us, T2_us kwargs, dispatching to
mesolve internally) since the QuTiP backend was first wired up, but it had
only ever been exercised by a single unit test
(test_gate_fidelity_with_T1_reduces_fidelity_sensibly in tests/
test_quantum.py) -- never as its own investigation. This demo asks the
obvious next question: how much does a qubit's OWN T1/T2 cost gate
fidelity, as a function of gate time, independent of any drive-chain
impairment?

Stays on the idealized 2-level qubit throughout (H0=0, exactly resonant),
NOT the 3-level transmon from transmon_leakage_demo.py -- gate_fidelity()'s
T2 collapse operator uses qt.sigmaz(), which is only meaningful for a
2-level qubit (see its own docstring); combining leakage physics with T2
dephasing would need a generalized dephasing operator this codebase
doesn't have yet (see INVESTIGATIONS.md's "Open questions").

Convention note: this demo's T2_us follows this codebase's existing
convention (see gate_fidelity()'s docstring) of deriving a pure-dephasing
rate as 1/T_phi = 1/T2 - 1/(2*T1) -- i.e. T2_us here means the same thing
as a measured "T2*" (free-induction-decay) time in an experiment, not a
Hahn-echo T2. T2_us = 2*T1_us reproduces the T1-limited case (zero extra
pure dephasing); T2_us < 2*T1_us adds pure dephasing on top of relaxation.

A real trap, found and fixed while building this: the first version of this
demo swept "gate duration" (the nominal pulse duration passed to
build_gaussian_envelope) and plotted infidelity vs. gate_duration/T1 --
expecting the curves for different T1 to collapse onto one line. They
didn't; the effective per-ns cost dropped by ~100x between very short and
very long nominal durations, for EVERY T1 tested. The cause: `result.
v_qubit_ensemble`'s array is longer than the nominal pulse -- convolving
the drive envelope with this schematic's own impulse response adds a
roughly FIXED ~99ns tail (confirmed directly: n_out - n_in corresponds to
~99ns at this demo's sample rate, essentially independent of the input
pulse's length), because that tail's length is set by the schematic's own
frequency-domain resolution, not by the drive pulse. For a CLOSED-system
run this tail is harmless (no drive there, H0=0, so the propagator segment
over it is ~identity). But `gate_fidelity()` uses the FULL array length as
T_gate for the Lindblad solve -- so with T1_us/T2_us given, decoherence
keeps acting for that extra ~99ns of essentially-zero-drive tail, inflating
the reported infidelity by an amount that has nothing to do with the
nominal gate and everything to do with how much convolution padding this
particular schematic happens to add. The fix: measure and plot against the
ACTUAL simulated gate time (`len(result.v_qubit_ensemble[0]) / result.fs`),
not the nominal requested duration. Once corrected, infidelity/[T_gate_true
/ T1] collapses onto a single, nearly-constant value (~0.33) across three
different T1 (10, 40, 100 us) and a 400x range of gate times -- confirming
clean, expected linear-in-time perturbative decoherence physics once the
padding artifact is accounted for. See gate_fidelity()'s docstring for a
permanent note about this for any future T1_us/T2_us user of this codebase.

THE FINDING (see INVESTIGATIONS.md Investigation 7 for full numbers):
  - Panel A (T1-limited, T2=2*T1): infidelity / (T_gate_true/T1) is close
    to a single constant (~0.33) across T1 in [10, 40, 100] us and
    T_gate_true spanning ~100ns to ~4us -- i.e. decoherence-driven
    infidelity really is governed by T_gate_true/T1 alone, once T_gate_true
    (not the nominal pulse duration) is used. Curvature appears only at the
    largest ratio tested (T_gate/T1 ~ 0.4), consistent with entering the
    non-perturbative regime.
  - Panel B (fixed gate time and T1, T2 swept from the T1-limited value down
    to heavy extra dephasing): confirms pure dephasing (T2 << 2*T1) can
    dominate the total decoherence-driven infidelity even when T1 itself is
    long and "good" -- consistent with real experience that T2* is often
    the practically-limiting number, not T1.

Run: python examples/t1_t2_decoherence_demo.py
Requires: SignalIntegrity, QuTiP, matplotlib.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import qutip

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_basic.si").resolve()

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6      # rad/(s.V) drive coupling
FS_ENVELOPE = 2e9           # 2 GSa/s baseband envelope grid

# Panel A: T1-limited (T2 = 2*T1), nominal gate duration sweep at three T1
# values -- x-axis in plots/reporting uses the TRUE simulated T_gate (see
# module docstring), not these nominal values directly.
DURATIONS_NS_A = np.array([10.0, 50.0, 200.0, 1000.0, 4000.0])
T1_US_VALUES_A = [10.0, 40.0, 100.0]

# Panel B: fixed gate duration and T1, T2 swept from T1-limited down to
# heavy extra pure dephasing
DURATION_NS_B = 200.0
T1_US_B = 30.0
T2_US_VALUES_B = np.array([60.0, 30.0, 15.0, 7.5, 3.75, 1.875, 0.9375])   # 60us = 2*T1 (T1-limited)


def infidelity_and_true_gate_time(schematic, duration_s: float, qmodel, T1_us=None, T2_us=None):
    """Returns (infidelity, T_gate_true) -- T_gate_true is the ACTUAL
    simulated propagation time (array length / fs), which is longer than
    `duration_s` by this schematic's own convolution tail (see module
    docstring) and is the value that actually matters for T1_us/T2_us.

    Calibration (via tuneup_amplitude()) is deliberately done WITHOUT
    T1_us/T2_us -- tuneup_amplitude() only ever optimizes the noise-free
    (closed-system) fidelity, matching this demo's own original approach
    of calibrating on the clean/classical pulse and only applying
    decoherence in a separate final gate_fidelity() call at the tuned
    scale.
    """
    sigma_s = duration_s / 6
    ref_shape = build_gaussian_envelope(duration_s, sigma_s, FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=ETA, ideal_gate="X", mode="complex_baseband",
    )
    result = tuned.result
    T_gate_true = (len(result.v_qubit_ensemble[0]) - 1) / result.fs
    fid = quantum.gate_fidelity(
        result, qmodel, ideal_gate="X", coupling_strength_per_volt=ETA,
        T1_us=T1_us, T2_us=T2_us,
    )
    return 1.0 - fid.noise_free.F_avg, T_gate_true


def main():
    warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    qmodel = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)

    # --- Panel A: T1-limited, duration sweep at three T1 values ---
    print("Panel A: T1-limited decoherence vs. TRUE gate_time/T1...")
    infid_A = {}
    ratio_true_A = {}
    for T1_us in T1_US_VALUES_A:
        infids, ratios = [], []
        for d_ns in DURATIONS_NS_A:
            infid, T_gate_true = infidelity_and_true_gate_time(
                schematic, d_ns * 1e-9, qmodel, T1_us=T1_us, T2_us=2 * T1_us,
            )
            infids.append(infid)
            ratios.append(T_gate_true / (T1_us * 1e-6))
        infid_A[T1_us] = np.array(infids)
        ratio_true_A[T1_us] = np.array(ratios)
        print(f"  T1={T1_us:.0f}us:")
        for d_ns, ratio, infid in zip(DURATIONS_NS_A, ratio_true_A[T1_us], infid_A[T1_us]):
            print(f"    nominal_d={d_ns:7.1f}ns  true_gate_time/T1={ratio:.5f}  infidelity={infid:.4e}  infid/ratio={infid/ratio:.4f}")

    # --- Panel B: fixed gate time and T1, T2 sweep ---
    print("\nPanel B: dephasing sweep (nominal gate_time=200ns, T1=30us, T2 varied)...")
    infid_B = []
    T_gate_true_B = None
    for T2_us in T2_US_VALUES_B:
        infid, T_gate_true_B = infidelity_and_true_gate_time(
            schematic, DURATION_NS_B * 1e-9, qmodel, T1_us=T1_US_B, T2_us=T2_us,
        )
        infid_B.append(infid)
    infid_B = np.array(infid_B)
    print(f"  (true gate time = {T_gate_true_B*1e9:.1f}ns)")
    for T2_us, infid in zip(T2_US_VALUES_B, infid_B):
        print(f"  T2={T2_us:6.3f}us (T2/2T1={T2_us/(2*T1_US_B):.3f})  infidelity={infid:.4e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for T1_us in T1_US_VALUES_A:
        ax1.loglog(ratio_true_A[T1_us], np.maximum(infid_A[T1_us], 1e-16), "o-", label=f"T1={T1_us:.0f}us (T2=2*T1)")
    ax1.set_xlabel("TRUE gate time / T1")
    ax1.set_ylabel("Gate infidelity (1 - F_avg)")
    ax1.set_title("T1-limited decoherence: collapses vs. true gate_time/T1")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    x_axis_B = T2_US_VALUES_B / (2 * T1_US_B)
    ax2.semilogy(x_axis_B, np.maximum(infid_B, 1e-16), "s-", color="C1")
    ax2.axvline(1.0, color="gray", lw=0.8, ls=":", label="T2 = 2*T1 (T1-limited, no extra dephasing)")
    ax2.set_xlabel("T2 / (2*T1)")
    ax2.set_ylabel("Gate infidelity (1 - F_avg)")
    ax2.set_title(f"Dephasing sweep (true gate_time~{T_gate_true_B*1e9:.0f}ns, T1={T1_US_B:.0f}us)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out_path = Path(__file__).parent / "t1_t2_decoherence_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
