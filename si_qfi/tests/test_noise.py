"""
tests/test_noise.py
====================
Unit tests for noise PSD computation and stochastic realization generation.
Verifiable without SignalIntegrity or QuTiP installed.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Noise tests
# ---------------------------------------------------------------------------

class TestNoise:

    def test_baseband_noise_variance(self):
        """Generated noise variance should match target PSD integral."""
        from si_qfi.noise.realization import generate_baseband_noise
        N = 8192
        fs = 1e9
        target_psd = 1e-12   # V²/Hz
        psd_array = np.full(N, target_psd)
        rng = np.random.default_rng(42)
        noise = generate_baseband_noise(N, fs, psd_array, rng=rng)
        # Expected variance: PSD_onesided * fs/2 (integrate over positive freqs)
        # For two-sided PSD: variance = PSD * fs
        expected_var = target_psd * fs
        actual_var = np.var(noise)
        # Allow 20% tolerance (statistical)
        assert abs(actual_var - expected_var) / expected_var < 0.2, (
            f"Noise variance {actual_var:.3e} should be near {expected_var:.3e}"
        )

    def test_rf_noise_is_real(self):
        """RF noise should be real-valued."""
        from si_qfi.noise.realization import generate_rf_noise
        N = 1024
        fs = 20e9
        psd = np.ones(N // 2 + 1) * 1e-12
        rng = np.random.default_rng(0)
        noise = generate_rf_noise(N, fs, psd, rng=rng)
        assert noise.dtype == np.float64
        assert np.all(np.isreal(noise))

    def test_baseband_noise_is_complex(self):
        """Baseband noise should be complex-valued."""
        from si_qfi.noise.realization import generate_baseband_noise
        N = 512
        fs = 500e6
        psd = np.ones(N) * 1e-12
        rng = np.random.default_rng(1)
        noise = generate_baseband_noise(N, fs, psd, rng=rng)
        assert np.iscomplexobj(noise)

    def test_psd_noise_figure(self):
        """Noise figure spec should produce correct noise temperature."""
        from si_qfi.noise.psd import psd_from_spec
        KB = 1.380649e-23
        R = 50.0
        NF_db = 3.0
        T_phys = 290.0
        T_eff = T_phys * (10**(NF_db/10) - 1)
        expected_psd = 4 * KB * T_eff * R

        spec = {"type": "noise_figure", "noise_figure_db": NF_db, "temperature_k": T_phys}
        freqs = np.array([1e9, 5e9, 10e9])
        psd = psd_from_spec(spec, freqs, R)
        np.testing.assert_allclose(psd, expected_psd, rtol=1e-6)
