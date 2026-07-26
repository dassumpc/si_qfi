"""
examples/transmon_leakage_demo.py
==================================
Every prior demo in this codebase (rabi_oscillation_demo.py through
impedance_mismatch_demo.py) used an idealized 2-level qubit (H0=0, exactly
resonant, no third level to leak into). This demo swaps in a real
`quantum.Transmon` (finite anharmonicity, n_levels=3) and asks: does a more
realistic qubit model change the fidelity story, independent of any
drive-chain impairment? No NL, no noise, no channel impairment anywhere in
this demo -- the schematic (tests/test_schematic_basic.si) is lossless and
perfectly matched, so any infidelity seen here comes purely from the qubit
Hamiltonian itself (leakage to |2>), not from anything upstream.

A real modeling trap, found while building this (#1 of 2): `Transmon.
build_H0()` returns the LAB-frame Hamiltonian (omega_q*num + (alpha/2)*
a+a+aa), but `quantum.build_hamiltonian()` (used by every gate_fidelity()
call in this codebase) assumes H0 is already expressed in the frame
ROTATING at the drive carrier -- its own docstring says so explicitly
("H0=0 for an exactly-resonant drive"). Naively calling `Transmon(...).
as_qubit_model()` and feeding it straight into gate_fidelity() would add
the qubit's full ~2*pi*5GHz precession term on top of a drive built for a
rotating frame -- nonsense, not leakage. The fix (see
`_transmon_rotating_frame_qubit_model()` below) is the standard DRAG-paper
starting point (Motzoi et al., PRL 103, 110501 (2009)): move to the frame
rotating at the carrier, assumed exactly resonant with the 0->1 transition,
so the precession term cancels exactly and only the anharmonic term
survives: H0' = (alpha/2)*a+a+aa.

A second, more consequential trap (#2 of 2): the first version of this demo
used `gate_fidelity(ideal_gate="X")` -- average GATE fidelity of the full
3-level propagator against an X-embedded-in-3-levels target -- and got
nonsense: infidelity ~0.3-0.7 that barely improved even at pulse durations
100x longer than the leakage timescale, where a direct check of the final
state (starting from |0>) showed population transfer to |1> was already
>99.99% complete with ~0 population in |2>. The propagator itself (checked
directly) was essentially perfect: -i*sigma_x on {|0>,|1>} plus a near-unit-
magnitude phase on |2>. The bug: that phase on |2> is UNPOPULATED-level free
evolution under H0' (accumulates for the *entire* gate duration, not just
while the drive is on) -- physically unobservable for any qubit that starts
and ends in the {|0>,|1>} computational subspace, but qt.average_gate_
fidelity() penalizes it anyway, because it's a *relative* phase between the
{0,1} block and the |2> block, and average_gate_fidelity is only phase-
invariant when the SAME global phase applies to the WHOLE Hilbert space (as
it does for n_levels=2, but not once a 3rd level with different H0-driven
phase evolution is present). The fix used here: measure LEAKAGE with the
metric that's actually insensitive to this artifact -- `gate_fidelity(
target_state=...)`'s STATE fidelity from |0> to the target |1>, plus the
raw leakage population read directly off `FidelityResult.final_states()`
(both already-supported, previously under-exercised features of this
module) -- rather than average GATE fidelity against a naively-embedded
target. See gate_fidelity()'s docstring for a permanent note about this.

THE FINDING (see INVESTIGATIONS.md Investigation 6 for full numbers):
  - Panel A: at fixed anharmonicity (alpha = -200 MHz, a typical transmon),
    a real 3-level transmon shows a genuine leakage-driven infidelity that
    grows sharply as the pulse gets fast relative to 1/|alpha| -- while the
    idealized 2-level model (no 3rd level to leak into) stays at the
    numerical floor regardless of pulse duration. This is the first
    investigation in this series where the *qubit model itself*, not the
    drive chain, is the thing that limits fidelity.
  - Panel B: DRAG suppresses the actual leakage POPULATION into |2> by up
    to ~65x at the shortest duration tested -- build_drag_envelope()'s
    derivative correction genuinely does what its docstring claims. But the
    overall state-infidelity-to-|1> improves only modestly at short
    durations, because this codebase's DRAG implementation is the
    leading-order I/Q-quadrature correction only, with no companion
    frequency-detuning term -- so it leaves behind the AC-Stark-shift-
    driven rotation-angle error that full DRAG implementations correct
    separately. A real, useful nuance: "DRAG suppresses leakage" and "DRAG
    fixes the gate" are not quite the same claim with this codebase's
    current DRAG implementation.

Run: python examples/transmon_leakage_demo.py
Requires: SignalIntegrity, QuTiP, matplotlib.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import qutip

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope, build_drag_envelope
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent.parent / "tests" / "test_schematic_basic.si").resolve()

CARRIER_GHZ = 5.0
ETA = 2 * np.pi * 10e6          # rad/(s.V) drive coupling
FS_ENVELOPE = 20e9              # 20 GSa/s -- enough headroom to resolve the
                                 # shortest (5ns) pulse and its DRAG derivative
ANHARMONICITY_MHZ = -200.0      # typical transmon anharmonicity, alpha ~ -Ec
DURATIONS_NS = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
SHORT_DURATION_NS = 5.0         # fixed leakage-heavy point for the DRAG panel

# This demo intentionally runs complex_baseband mode only: leakage is a pure
# qubit-Hamiltonian effect (governed entirely by the I/Q envelope reaching
# the qubit plane), and this schematic is flat-gain/lossless/matched, so
# real_axis mode would just reproduce the same I(t)/Q(t) at extra simulation
# cost (Investigation 4 already established the two modes agree on any
# purely-linear channel).


def _transmon_rotating_frame_qubit_model(anharmonicity_hz: float, n_levels: int) -> quantum.QubitModel:
    """See module docstring's "modeling trap #1" for why this is needed
    instead of Transmon.as_qubit_model()."""
    a = qutip.destroy(n_levels)
    alpha = 2 * np.pi * anharmonicity_hz
    H0 = (alpha / 2.0) * a.dag() * a.dag() * a * a
    return quantum.QubitModel(H0=H0, n_levels=n_levels)


def run_leakage_case(schematic, duration_s: float, qmodel: quantum.QubitModel, use_drag: bool):
    """
    Returns (state_infidelity, leakage_population): state_infidelity is
    1 - state fidelity from |0> to the target |1> (the physically-meaningful
    "did the X gate work" metric -- see module docstring's trap #2 for why
    this is used instead of ideal_gate="X" average gate fidelity for
    n_levels > 2); leakage_population is the final population found outside
    the {|0>,|1>} computational subspace (0.0 for the 2-level model, which
    has no such subspace to leak into).

    Calibration is via tuneup_amplitude(): its target_state|0>-\>|1>
    pattern detection recognizes this exact target/initial_state pair and
    uses the same exact classical-pulse-area fast path as the old
    _calibrate_to_pi() (2 engine.run() calls, no NL here) -- scaling the
    whole complex (I+jQ) reference shape by one real scalar, which is what
    DRAG needs to keep its I/Q ratio intact after calibration.
    """
    sigma_s = duration_s / 6
    if use_drag:
        ref_shape = build_drag_envelope(duration_s, sigma_s, ANHARMONICITY_MHZ * 1e6, FS_ENVELOPE, pi_amp=1.0)
    else:
        ref_shape = build_gaussian_envelope(duration_s, sigma_s, FS_ENVELOPE, amp=1.0)

    n = qmodel.n_levels
    target = qutip.basis(n, 1)
    initial = qutip.basis(n, 0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, FS_ENVELOPE, CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=ETA,
        target_state=target, initial_state=initial, mode="complex_baseband",
    )
    fid = tuned.fidelity
    state_infidelity = 1.0 - fid.noise_free.state_F_avg

    rho_final = fid.noise_free.final_state(initial_state=initial)
    populations = np.real(rho_final.diag())
    leakage_population = float(np.sum(populations[2:])) if n > 2 else 0.0

    return state_infidelity, leakage_population


def main():
    warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    qmodel_2lvl = quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)
    qmodel_3lvl = _transmon_rotating_frame_qubit_model(ANHARMONICITY_MHZ * 1e6, n_levels=3)

    # --- Panel A: state infidelity vs. pulse duration, 2-level idealized vs. real transmon ---
    print("Panel A: sweeping pulse duration (2-level idealized vs. 3-level transmon)...")
    infid_2lvl = np.zeros_like(DURATIONS_NS)
    infid_3lvl_gauss = np.zeros_like(DURATIONS_NS)
    leak_3lvl_gauss = np.zeros_like(DURATIONS_NS)
    for i, d_ns in enumerate(DURATIONS_NS):
        infid_2lvl[i], _ = run_leakage_case(schematic, d_ns * 1e-9, qmodel_2lvl, use_drag=False)
        infid_3lvl_gauss[i], leak_3lvl_gauss[i] = run_leakage_case(schematic, d_ns * 1e-9, qmodel_3lvl, use_drag=False)

    # --- Panel B: Gaussian vs. DRAG at the fixed short, leakage-heavy duration ---
    print("Panel B: Gaussian vs. DRAG across the duration sweep (state infidelity AND raw leakage population)...")
    infid_3lvl_drag = np.zeros_like(DURATIONS_NS)
    leak_3lvl_drag = np.zeros_like(DURATIONS_NS)
    for i, d_ns in enumerate(DURATIONS_NS):
        infid_3lvl_drag[i], leak_3lvl_drag[i] = run_leakage_case(schematic, d_ns * 1e-9, qmodel_3lvl, use_drag=True)

    print()
    print("Panel A/B (duration sweep, 3-level transmon):")
    for i, d_ns in enumerate(DURATIONS_NS):
        print(
            f"  duration={d_ns:6.1f}ns   2lvl infid={infid_2lvl[i]:.3e}   "
            f"gauss: state_infid={infid_3lvl_gauss[i]:.3e} leak_pop={leak_3lvl_gauss[i]:.3e}   "
            f"drag: state_infid={infid_3lvl_drag[i]:.3e} leak_pop={leak_3lvl_drag[i]:.3e}   "
            f"leak_suppression={leak_3lvl_gauss[i] / max(leak_3lvl_drag[i], 1e-300):.1f}x"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(DURATIONS_NS, np.maximum(infid_2lvl, 1e-16), "o-", label="2-level idealized (no leakage channel)")
    ax1.loglog(DURATIONS_NS, np.maximum(infid_3lvl_gauss, 1e-16), "s-", label="3-level transmon (Gaussian)")
    ax1.set_xlabel("Pulse duration (ns)")
    ax1.set_ylabel("State infidelity, |0> -> target |1>")
    ax1.set_title(f"Leakage-driven infidelity vs. duration (alpha = {ANHARMONICITY_MHZ:.0f} MHz)")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    ax2b = ax2.twinx()
    l1, = ax2.loglog(DURATIONS_NS, np.maximum(infid_3lvl_gauss, 1e-16), "s-", color="C1", label="state infid (Gaussian)")
    l2, = ax2.loglog(DURATIONS_NS, np.maximum(infid_3lvl_drag, 1e-16), "s--", color="C2", label="state infid (DRAG)")
    l3, = ax2b.loglog(DURATIONS_NS, np.maximum(leak_3lvl_gauss, 1e-16), "^-", color="C3", label="leak population (Gaussian)")
    l4, = ax2b.loglog(DURATIONS_NS, np.maximum(leak_3lvl_drag, 1e-16), "^--", color="C4", label="leak population (DRAG)")
    ax2.set_xlabel("Pulse duration (ns)")
    ax2.set_ylabel("State infidelity")
    ax2b.set_ylabel("Leakage population in |2>")
    ax2.set_title("Gaussian vs. DRAG: state infidelity vs. raw leakage")
    ax2.legend(handles=[l1, l2, l3, l4], loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out_path = Path(__file__).parent / "transmon_leakage_demo.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
