"""
examples/rabi_oscillation_demo.py
==================================
Demonstrates the SI-QFI -> QuTiP link end-to-end: sweep a resonant drive
pulse's amplitude through a real SI schematic (tests/test_schematic_basic.si
-- lossless, flat 2.5x gain, matched at every junction), propagate it via
both simulation modes (complex_baseband and real_axis), and solve a plain
2-level qubit's dynamics with QuTiP at each point.

No nonlinearity, no noise -- this is the "no impairment" baseline case, so:
  - |1>-population should trace the textbook Rabi flopping curve
    P1(theta) = sin^2(theta/2), theta = pulse area in radians.
  - Gate fidelity to the ideal X gate should reach unity -- limited only by
    numerical accuracy -- exactly at the calibrated pi-pulse (theta=pi), for
    BOTH simulation modes, confirming the SI-QFI -> QuTiP bridge is correct
    end-to-end (not just that each half works in isolation).

Run: python examples/rabi_oscillation_demo.py
Requires: SignalIntegrity, QuTiP, matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import qutip

from si_qfi.schematic import loader as si_loader
from si_qfi.simulation import engine
from si_qfi.source.waveform import SourceWaveform, build_gaussian_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_basic.si").resolve()

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6      # rad/(s.V) drive coupling
DURATION_S = 100e-9         # 100 ns gate window
SIGMA_S = DURATION_S / 6
FS_ENVELOPE = 2e9           # 2 GSa/s baseband envelope grid
LPF_CUTOFF_HZ = 500e6       # real-axis demod low-pass (>> pulse BW ~10MHz,
                             # << 2*carrier image at 10 GHz)
N_SWEEP = 41                # sweep points, theta/pi in [0, 2.2]


def _source_from_shape(shape: np.ndarray, fs: float, carrier_ghz: float) -> SourceWaveform:
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    n = len(shape)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(shape.astype(complex)))
    return SourceWaveform(carrier_freq_ghz=carrier_ghz, envelope=envelope)


def _realized_theta(result, eta: float, lpf_cutoff_hz=None) -> float:
    v = np.asarray(result.v_nl_qubit)
    t = np.arange(len(v)) / result.fs
    if result.mode == "complex_baseband":
        env_i = np.real(v)
    else:
        env_i, _ = quantum.demodulate(v, t, result.carrier_freq_hz, lpf_cutoff_hz)
    return float(eta * np.trapz(env_i, t))


def _population_1(result, qmodel, eta: float, lpf_cutoff_hz=None) -> float:
    """
    Probability of finding the qubit in |1> at the end of the gate, starting
    from |0> -- the quantity a real Rabi-flopping measurement (e.g. readout
    vs. pulse amplitude) would report. Mirrors quantum.gate_fidelity()'s own
    internals (including its max_step/nsteps fix -- see quantum/__init__.py)
    but solves the state directly via sesolve rather than the full
    propagator, since only one input state (|0>) is needed here.
    """
    v = np.asarray(result.v_qubit_ensemble[0])
    fs = result.fs
    t_array = np.arange(len(v)) / fs
    dt = 1.0 / fs
    if result.mode == "complex_baseband":
        env_i = np.ascontiguousarray(np.real(v))
        env_q = np.ascontiguousarray(np.imag(v))
    else:
        env_i, env_q = quantum.demodulate(v, t_array, result.carrier_freq_hz, lpf_cutoff_hz)
        env_i = np.ascontiguousarray(env_i)
        env_q = np.ascontiguousarray(env_q)

    H = quantum.build_hamiltonian(qmodel, env_i, env_q, t_array, eta)
    psi0 = qutip.basis(qmodel.n_levels, 0)
    solver_options = {"max_step": dt, "nsteps": max(10_000, 20 * len(v))}
    res = qutip.sesolve(H, psi0, t_array, options=solver_options)
    psi_final = res.states[-1]
    return float(np.abs(psi_final.full()[1, 0]) ** 2)


def _calibrate_amp_pi(schematic, mode: str, eta: float, lpf_cutoff_hz=None) -> np.ndarray:
    """One reference run -> exact (linear system) amplitude scale hitting
    theta=pi. Returns the reference SHAPE array already scaled to a pi-pulse."""
    ref_shape = build_gaussian_envelope(DURATION_S, SIGMA_S, FS_ENVELOPE, amp=1.0)
    source_ref = _source_from_shape(ref_shape, FS_ENVELOPE, CARRIER_GHZ)
    result_ref = engine.run(
        schematic, source_ref, nonlinear=None, noise=None, n_realizations=1, mode=mode,
    )
    theta_ref = _realized_theta(result_ref, eta, lpf_cutoff_hz)
    return ref_shape * (np.pi / theta_ref)


def sweep_mode(schematic, mode: str, qmodel, thetas_over_pi: np.ndarray, lpf_cutoff_hz=None):
    pi_shape = _calibrate_amp_pi(schematic, mode, ETA, lpf_cutoff_hz)
    populations = np.zeros_like(thetas_over_pi)
    fidelities = np.zeros_like(thetas_over_pi)
    for i, s in enumerate(thetas_over_pi):
        shape = pi_shape * s
        source = _source_from_shape(shape, FS_ENVELOPE, CARRIER_GHZ)
        result = engine.run(
            schematic, source, nonlinear=None, noise=None, n_realizations=1, mode=mode,
        )
        populations[i] = _population_1(result, qmodel, ETA, lpf_cutoff_hz)
        fid = quantum.gate_fidelity(
            result, qmodel, ideal_gate="X", coupling_strength_per_volt=ETA,
            lpf_cutoff_hz=lpf_cutoff_hz,
        )
        fidelities[i] = fid.F_avg
    return populations, fidelities


def main():
    import warnings
    warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    H0 = 0 * qutip.qeye(2)          # exactly resonant, rotating frame
    qmodel = quantum.QubitModel(H0=H0, n_levels=2)

    # Force theta/pi=1.0 (the calibrated X pulse) to be an EXACT grid point
    # -- a plain linspace(0, 2.2, N_SWEEP) does not land on 1.0 exactly
    # (nearest point was 0.99), which was previously misreported as ~2e-4
    # numerical infidelity at "theta=pi" when it was actually a ~1% pulse-area
    # miscalibration from an off-grid sample point, not a solver limit.
    thetas_over_pi = np.union1d(np.linspace(0.0, 2.2, N_SWEEP), [1.0])

    print("Sweeping complex_baseband mode...")
    pop_bb, fid_bb = sweep_mode(schematic, "complex_baseband", qmodel, thetas_over_pi)

    print("Sweeping real_axis mode...")
    pop_ra, fid_ra = sweep_mode(schematic, "real_axis", qmodel, thetas_over_pi, LPF_CUTOFF_HZ)

    pop_analytic = np.sin(np.pi * thetas_over_pi / 2.0) ** 2

    i_pi = int(np.argmin(np.abs(thetas_over_pi - 1.0)))
    print()
    print(f"At theta = pi (calibrated X pulse):")
    print(f"  complex_baseband: F_avg = {fid_bb[i_pi]:.7f}   P1 = {pop_bb[i_pi]:.7f}")
    print(f"  real_axis:        F_avg = {fid_ra[i_pi]:.7f}   P1 = {pop_ra[i_pi]:.7f}")
    print(f"  analytic P1 at theta=pi: {pop_analytic[i_pi]:.7f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(thetas_over_pi, pop_analytic, "k--", lw=1, label="analytic sin^2(theta/2)")
    ax1.plot(thetas_over_pi, pop_bb, "o", ms=4, label="complex_baseband (simulated)")
    ax1.plot(thetas_over_pi, pop_ra, "x", ms=5, label="real_axis (simulated)")
    ax1.axvline(1.0, color="gray", lw=0.8, ls=":")
    ax1.set_ylabel("P(|1>) after gate")
    ax1.set_title("Rabi oscillation: population vs. pulse area (no impairments)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(thetas_over_pi, fid_bb, "o", ms=4, label="complex_baseband gate fidelity to X")
    ax2.plot(thetas_over_pi, fid_ra, "x", ms=5, label="real_axis gate fidelity to X")
    ax2.axvline(1.0, color="gray", lw=0.8, ls=":", label="theta = pi (calibrated X gate)")
    ax2.set_xlabel("Pulse area, theta / pi")
    ax2.set_ylabel("Average gate fidelity to X")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "rabi_oscillation_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
