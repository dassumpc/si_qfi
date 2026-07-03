"""
tests/test_nonlinear.py
=======================
Unit tests for the nonlinear models.
Verifiable without SignalIntegrity or QuTiP installed.
"""

import numpy as np
import pytest
from si_qfi.nonlinear.saleh import SalehModel
from si_qfi.nonlinear.amam_ampm import TabulatedAMAM
from si_qfi.nonlinear.memory_polynomial import MemoryPolynomial
from si_qfi.nonlinear.volterra import VolterraModel


# ---------------------------------------------------------------------------
# Saleh model tests
# ---------------------------------------------------------------------------

class TestSalehModel:

    def test_linear_regime_no_compression(self):
        """At very small amplitudes, Saleh output should be ≈ alpha_a * input."""
        model = SalehModel(alpha_a=2.0, beta_a=1.0, alpha_phi=0.0, beta_phi=0.0)
        u_small = np.array([0.01 + 0j, 0.01 + 0j])
        u_out = model.apply_baseband(u_small)
        expected_gain = 2.0
        actual_gain = np.abs(u_out[0]) / np.abs(u_small[0])
        assert abs(actual_gain - expected_gain) < 1e-3, (
            f"Linear-regime gain should be ≈ {expected_gain}, got {actual_gain:.4f}"
        )

    def test_gain_compression_at_large_amplitude(self):
        """At large amplitudes, gain should be lower than small-signal gain."""
        model = SalehModel(alpha_a=2.0, beta_a=1.0)
        amp_small = np.array([0.001])
        amp_large = np.array([2.0])
        g_small = model.gain(amp_small)[0]
        g_large = model.gain(amp_large)[0]
        assert g_large < g_small, "Gain should compress at large amplitudes."

    def test_zero_input_gives_zero_output(self):
        """Zero input should give zero output (no division by zero)."""
        model = SalehModel(alpha_a=2.0, beta_a=1.0, alpha_phi=1.0, beta_phi=0.5)
        u = np.array([0.0 + 0j, 1.0 + 0j, 0.0 + 0j])
        out = model.apply_baseband(u)
        assert out[0] == 0.0j, "Zero input must give zero output."
        assert out[2] == 0.0j

    def test_phase_preserved_no_ampm(self):
        """With alpha_phi=0, output phase should match input phase."""
        model = SalehModel(alpha_a=1.0, beta_a=0.5, alpha_phi=0.0, beta_phi=0.0)
        u = np.array([0.5 * np.exp(1j * np.pi / 4)])
        out = model.apply_baseband(u)
        assert abs(np.angle(out[0]) - np.pi / 4) < 1e-6

    def test_ampm_shifts_phase(self):
        """With non-zero alpha_phi, output phase should differ from input."""
        model = SalehModel(alpha_a=1.0, beta_a=0.1, alpha_phi=2.0, beta_phi=1.0)
        u = np.array([1.0 + 0j])
        out = model.apply_baseband(u)
        phi_out = np.angle(out[0])
        expected_phi = model.phase_shift(np.array([1.0]))[0]
        assert abs(phi_out - expected_phi) < 1e-6

    def test_from_p1db_ip3_compression_point(self):
        """Model built from P1dB should compress by ~1 dB at the given P1dB amplitude."""
        a1db = 0.5
        a_ip3 = a1db / 0.383   # 9.6 dB rule
        model = SalehModel.from_p1db_ip3(
            p1db_amplitude=a1db,
            ip3_amplitude=a_ip3,
            small_signal_gain=1.0,
        )
        # Check that gain at P1dB is 20*log10(G[A_1dB]/G[0]) ≈ -1 dB
        g_small = model.gain(np.array([1e-4]))[0]
        g_at_p1db = model.gain(np.array([a1db]))[0]
        ratio_db = 20 * np.log10(g_at_p1db / g_small)
        assert abs(ratio_db - (-1.0)) < 0.3, (
            f"Expected -1 dB at P1dB, got {ratio_db:.2f} dB"
        )

    def test_describing_function_coefficient(self):
        """
        Verify the 3/4 describing function coefficient (PRD §5.1).

        For f(x) = x + a·x³, the complex baseband AM-AM gives:
            A_out = A + (3a/4)·A³

        Fit a MemoryPolynomial from P1dB/IP3 and verify the k=3 coefficient
        matches the (3/4)·c formula.
        """
        a_ip3 = 1.0
        # Coefficient c from cubic model: A_IP3 = sqrt(-4/(3c)) → c = -4/(3*A_IP3^2)
        c = -4.0 / (3.0 * a_ip3**2)
        # Describing function coefficient for k=3 is 3/4
        expected_a30 = (3.0 / 4.0) * c   # = -1/A_IP3^2 = -1.0

        mp = MemoryPolynomial.from_p1db_ip3(
            p1db_amplitude=a_ip3 * 0.383,
            ip3_amplitude=a_ip3,
            small_signal_gain=1.0,
            memory_depth=0,
        )
        a30 = mp._coefficients[1, 0]   # row 1 = order 3, col 0 = m=0
        assert abs(np.real(a30) - expected_a30) < 1e-6, (
            f"k=3 coefficient should be {expected_a30:.4f}, got {np.real(a30):.4f}"
        )


# ---------------------------------------------------------------------------
# Tabulated AM-AM tests
# ---------------------------------------------------------------------------

class TestTabulatedAMAM:

    def _linear_model(self, gain=2.0):
        amp_in = np.linspace(0.01, 2.0, 50)
        amp_out = gain * amp_in
        return TabulatedAMAM(amp_in, amp_out)

    def test_linear_passthrough(self):
        """A linear AM-AM table should act as a gain."""
        gain = 1.5
        model = self._linear_model(gain)
        u = np.array([0.5 + 0.3j, 0.2 - 0.1j])
        out = model.apply_baseband(u)
        expected_amp = gain * np.abs(u)
        np.testing.assert_allclose(np.abs(out), expected_amp, rtol=1e-3)

    def test_phase_preserved(self):
        """Output phase must equal input phase (no AM-PM for this model)."""
        model = self._linear_model()
        u = np.array([0.5 * np.exp(1j * 1.2)])
        out = model.apply_baseband(u)
        assert abs(np.angle(out[0]) - 1.2) < 1e-4

    def test_monotone_input_required(self):
        """Non-monotone amp_in should raise ValueError."""
        with pytest.raises(ValueError, match="strictly monotonically"):
            TabulatedAMAM(
                amp_in=np.array([0.1, 0.5, 0.3, 1.0]),
                amp_out=np.array([0.1, 0.5, 0.3, 1.0]),
            )

    def test_clip_extrapolation(self):
        """Amplitudes beyond table range should be clipped to edge value."""
        amp_in = np.array([0.1, 0.5, 1.0])
        amp_out = np.array([0.1, 0.45, 0.8])
        model = TabulatedAMAM(amp_in, amp_out, extrapolate="clip")
        u_large = np.array([5.0 + 0j])
        out = model.apply_baseband(u_large)
        # Output amplitude should be clipped to amp_out[-1] = 0.8
        assert abs(np.abs(out[0]) - 0.8) < 1e-3


# ---------------------------------------------------------------------------
# Memory polynomial tests
# ---------------------------------------------------------------------------

class TestMemoryPolynomial:

    def test_memoryless_m0_matches_saleh_linear_regime(self):
        """At M=0 and small amplitude, MP should behave like a linear gain."""
        mp = MemoryPolynomial.from_p1db_ip3(
            p1db_amplitude=0.5,
            ip3_amplitude=1.3,
            small_signal_gain=2.0,
            memory_depth=0,
        )
        u = np.array([0.001 + 0j] * 10)
        out = mp.apply_baseband(u)
        gain = np.abs(out[5]) / np.abs(u[5])
        assert abs(gain - 2.0) < 0.01, f"Small-signal gain should be 2.0, got {gain:.4f}"

    def test_memory_introduces_delay_dependence(self):
        """With M > 0, a step input should produce transient at transition."""
        # Coefficients: k=1 only, equal weights across taps (moving average)
        M = 3
        coeff = np.zeros((1, M + 1), dtype=complex)
        coeff[0, :] = 1.0 / (M + 1)   # uniform averaging
        mp = MemoryPolynomial(coefficients=coeff, orders=[1])

        # Step input: 0 for first half, 1 for second
        N = 100
        u = np.zeros(N, dtype=complex)
        u[N // 2:] = 1.0
        out = mp.apply_baseband(u)

        # Output at step transition should be intermediate (not 0 or 1 immediately)
        step_idx = N // 2
        assert 0 < np.abs(out[step_idx]) < 1.0, (
            "Memory should cause gradual transition at step."
        )

    def test_zero_input_zero_output(self):
        mp = MemoryPolynomial.from_p1db_ip3(0.5, 1.3, memory_depth=2)
        u = np.zeros(20, dtype=complex)
        out = mp.apply_baseband(u)
        np.testing.assert_allclose(np.abs(out), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Volterra tests
# ---------------------------------------------------------------------------

class TestVolterraModel:

    def test_linear_passthrough_at_small_amplitude(self):
        """At very small amplitude, Volterra output should be ≈ linear."""
        v = VolterraModel(
            option="describing",
            p1db_amplitude=0.5,
            ip3_amplitude=1.3,
            small_signal_gain=1.0,
            memory_depth=0,
        )
        x_small = np.ones(100, dtype=float) * 0.001
        out = v.apply_real_axis(x_small)
        # Should be ≈ 1.0 * x_small (small nonlinear correction)
        gain = np.mean(out[10:]) / 0.001
        assert abs(gain - 1.0) < 0.01

    def test_compression_at_large_amplitude(self):
        """At large amplitude, output/input ratio should be less than small-signal gain."""
        v = VolterraModel(
            option="describing",
            p1db_amplitude=0.5,
            ip3_amplitude=1.3,
            small_signal_gain=1.0,
            memory_depth=0,
        )
        x_small = np.ones(50) * 0.001
        x_large = np.ones(50) * 0.8
        out_small = v.apply_real_axis(x_small)
        out_large = v.apply_real_axis(x_large)
        gain_small = np.mean(out_small[10:]) / 0.001
        gain_large = np.mean(out_large[10:]) / 0.8
        assert gain_large < gain_small, "Gain should compress at large amplitude."

    def test_supports_real_axis_not_baseband(self):
        v = VolterraModel(option="describing", p1db_amplitude=0.5, ip3_amplitude=1.3)
        assert v.supports_real_axis is True
        assert v.supports_baseband is False


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
