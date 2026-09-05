"""
tests/test_quantum_noise_density_sweep.py
===========================================
Regression test for examples/noise_density_sweep_demo.py's finding: gate
infidelity grows with the drive-line noise source's spectral density,
staying near the noise-free floor at low density and becoming clearly,
substantially larger at high density -- exercising the full pipeline
(engine.run() with a real SI statistical-noise-source schematic +
gate_fidelity()'s noise-ensemble statistics), not just the underlying
noise/propagation.py or schematic/noise.py units already covered by
tests/test_noise.py and tests/test_engine_noise.py.

Uses the override PSD mechanism (same as the demo) so the density can be
swept over decades independent of any physical resistor/temperature value.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")
qutip = pytest.importorskip("qutip")

warnings.filterwarnings("ignore", message="SI-QFI: Narrowband ratio")

from si_qfi.schematic import loader as si_loader
from si_qfi.source.waveform import build_gaussian_envelope, source_from_envelope_array
from si_qfi.simulation import engine as _engine
from si_qfi import quantum

SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_noise.si").resolve()

_CARRIER_GHZ = 5.0
_ETA = 2 * np.pi * 10e6
_DURATION_S = 20e-9
_SIGMA_S = _DURATION_S / 6
_FS_ENVELOPE = 4e9
_N_REALIZATIONS = 200


@pytest.fixture(scope="module")
def qmodel():
    return quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)


@pytest.fixture(scope="module")
def tuned_shape(qmodel):
    """Calibrate once (noise-free) and reuse across the sweep -- matches
    the demo script's own approach (tuneup_amplitude optimizes noise-free
    fidelity by design, see quantum/fidelity.py)."""
    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    ref_shape = build_gaussian_envelope(_DURATION_S, _SIGMA_S, _FS_ENVELOPE, amp=1.0)
    tuned = quantum.tuneup_amplitude(
        schematic, ref_shape, _FS_ENVELOPE, _CARRIER_GHZ,
        qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X",
    )
    assert tuned.achieved
    return tuned.scale * ref_shape


def _infidelity_at_psd(psd_v2_per_hz, tuned_shape, qmodel, seed=123):
    schematic = si_loader.load_schematic(SCHEMATIC_PATH)
    src = source_from_envelope_array(tuned_shape, _FS_ENVELOPE, _CARRIER_GHZ)
    result = _engine.run(
        schematic, src, nonlinear=None,
        noise={"VN1": {"single_sided_psd_v2_per_hz": psd_v2_per_hz}},
        n_realizations=_N_REALIZATIONS, mode="complex_baseband", seed=seed,
    )
    fid = quantum.gate_fidelity(result, qmodel, coupling_strength_per_volt=_ETA, ideal_gate="X")
    return 1.0 - fid.noise.F_avg


class TestNoiseDensitySweep:
    def test_low_density_stays_near_floor(self, tuned_shape, qmodel):
        """At a negligible noise density, infidelity should be indistinguishable
        from the ordinary numerical floor already established elsewhere in
        this codebase's investigations (~1e-7) -- not measurably driven by
        the (vanishingly small) injected noise."""
        infid = _infidelity_at_psd(1e-20, tuned_shape, qmodel)
        assert abs(infid) < 1e-6

    def test_high_density_costs_real_fidelity(self, tuned_shape, qmodel):
        """At a large noise density, infidelity should be clearly,
        substantially above the numerical floor -- a real, measurable cost."""
        infid = _infidelity_at_psd(1e-12, tuned_shape, qmodel)
        assert infid > 1e-5

    def test_infidelity_grows_with_density(self, tuned_shape, qmodel):
        """Coarse monotonic trend across several decades of density --
        tolerates the same kind of small-scale statistical scatter near the
        floor documented elsewhere in this codebase (e.g.
        test_quantum_impedance_mismatch.py's _assert_monotonic_within_floor)
        by only checking widely-separated points, not every adjacent pair."""
        densities = [1e-19, 1e-16, 1e-13, 1e-10]
        infidelities = [_infidelity_at_psd(d, tuned_shape, qmodel) for d in densities]
        assert infidelities[-1] > infidelities[0]
        assert infidelities[-1] > 10 * max(abs(infidelities[0]), 1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
