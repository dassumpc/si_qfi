"""
tests/test_nonlinear.py
=======================
Unit tests for the nonlinear models.
Verifiable without SignalIntegrity or QuTiP installed.

Only two nonlinear model families exist in this codebase: Saleh (SalehModel
for complex baseband, SalehRealAxisModel for real-axis) and VolterraModel
(real-axis, plain cubic only -- describing-function fits requiring BOTH
op1db_amplitude and oip3_amplitude simultaneously were removed; each model's
`from_op1db_oip3()`/`option='describing'` constructor now accepts EXACTLY
ONE of the two, never both, never neither).

Hand-verification note: the op1db-only cases below intentionally include a
"clean numbers" variant chosen so the input-referred amplitude at the 1dB
point (op1db_in) comes out to exactly 1.0 -- this removes a
division/squaring step from the mental arithmetic, leaving only the
well-known identity 20*log10(10**(-1/20)) == -1 (exact, by definition of a
-1dB ratio) to check by hand/calculator.

OIP3-only -> implied OP1dB checks: fitting from OIP3 alone does not leave
OP1dB free -- it's fully determined by the resulting model's shape. Each
oip3-only test below also checks this implied OP1dB lands at the expected
dB gap below OIP3 (~10.1dB for Saleh's rational G[A], ~10.6dB for a plain
cubic Volterra polynomial -- see the derivations in nonlinear/saleh.py's and
nonlinear/volterra.py's module docstrings). For the real-axis models
(SalehRealAxisModel, VolterraModel), "OP1dB" here means the actual physical
single-tone CW FUNDAMENTAL compression point (found via _fundamental_gain/
_find_compression_point below, which simulate an actual sinusoid through
apply_real_axis() and FFT-extract the fundamental), NOT the raw polynomial
evaluated at a constant input -- those are different questions for a
real-axis model that produces harmonics (see module docstrings).

Noise generation/PSD tests live in tests/test_noise.py.
"""

import warnings

import numpy as np
import pytest
from si_qfi.nonlinear.saleh import SalehModel, SalehRealAxisModel
from si_qfi.nonlinear.volterra import VolterraModel
from si_qfi.nonlinear.tabulated import TabulatedModel
from si_qfi.nonlinear.registry import build_nonlinear_nodes

_ONE_DB_RATIO = 10 ** (-1.0 / 20.0)   # ~0.891250938; 20*log10(this) == -1 exactly


# ---------------------------------------------------------------------------
# Shared helpers for physical (single-tone CW) compression-point checks on
# real-axis models -- see module docstring.
# ---------------------------------------------------------------------------

def _fundamental_gain(apply_fn, amplitude, fs=2000.0, f0=10.0, n_cycles=200):
    """
    Drive a real memoryless nonlinearity (apply_fn: array -> array, e.g.
    model.apply_real_axis) with a single CW tone of the given amplitude, and
    return the FFT-extracted FUNDAMENTAL gain (fundamental output amplitude
    / input amplitude) -- the physically meaningful quantity a spectrum
    analyzer would report after harmonic filtering, as opposed to a raw
    quasi-static (constant-input) evaluation. fs/f0/n_cycles are chosen so
    f0 lands exactly on an FFT bin (T=n_cycles/f0 -> df=f0/n_cycles), no
    spectral leakage.
    """
    t = np.arange(0, n_cycles / f0, 1.0 / fs)
    x = amplitude * np.cos(2 * np.pi * f0 * t)
    y = apply_fn(x)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(t), 1.0 / fs)
    i0 = np.argmin(np.abs(freqs - f0))
    amp0 = 2 * np.abs(Y[i0]) / len(t)
    return amp0 / amplitude


def _find_compression_point(apply_fn, small_signal_gain, lo=1e-4, hi=5.0, n_iter=40):
    """
    Bisection search for the input amplitude at which the single-tone
    FUNDAMENTAL gain (_fundamental_gain) has compressed by exactly 1dB
    relative to small_signal_gain. apply_fn must be monotonically
    compressing over [lo, hi]. Intermediate bisection probes routinely
    exceed the model's own max_monotonic_amplitude before converging (that's
    expected -- the bracket has to start wide) -- their overdrive warnings
    are suppressed here rather than polluting test output.
    """
    target = small_signal_gain * _ONE_DB_RATIO
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            g = _fundamental_gain(apply_fn, mid)
            if g > target:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Saleh model tests (complex baseband)
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

    def test_from_op1db_oip3_requires_exactly_one_neither_given(self):
        with pytest.raises(ValueError, match="requires exactly one of"):
            SalehModel.from_op1db_oip3()

    def test_from_op1db_oip3_requires_exactly_one_both_given(self):
        """Both op1db_amplitude and oip3_amplitude -> ValueError. This
        codebase no longer supports fitting both exactly (previously via a
        gamma_a extension, removed for simplicity) -- see module docstring."""
        with pytest.raises(ValueError, match="not both"):
            SalehModel.from_op1db_oip3(op1db_amplitude=0.5, oip3_amplitude=1.3)

    def test_from_op1db_oip3_takes_no_gain_argument(self):
        """from_op1db_oip3() has no small_signal_gain parameter at all --
        it always builds alpha_a=1.0 (purely output-referred nonlinearity,
        PRD §3.6). Passing it is a TypeError, not silently ignored."""
        with pytest.raises(TypeError):
            SalehModel.from_op1db_oip3(oip3_amplitude=1.0, small_signal_gain=2.0)

    # -- op1db_amplitude only -----------------------------------------------

    def test_op1db_only_compression_point(self):
        """
        Model built from op1db_amplitude alone should compress by EXACTLY
        1 dB at the input-referred P1dB amplitude -- an exact
        rational-equation solve, not the linearized ("9.6 dB rule")
        approximation.
        """
        op1db = 0.5
        model = SalehModel.from_op1db_oip3(op1db_amplitude=op1db)
        assert model.alpha_a == 1.0   # no gain argument -> always unity
        assert model.beta_a == pytest.approx((10**(1/20) - 1) / (op1db / _ONE_DB_RATIO)**2)
        op1db_in = op1db / _ONE_DB_RATIO
        g_small = model.gain(np.array([1e-6]))[0]
        g_at_p1db = model.gain(np.array([op1db_in]))[0]
        ratio_db = 20 * np.log10(g_at_p1db / g_small)
        assert abs(ratio_db - (-1.0)) < 1e-6, (
            f"Expected exactly -1 dB at (input-referred) P1dB, got {ratio_db:.4f} dB"
        )

    def test_op1db_only_hand_checkable_clean_numbers(self):
        """
        op1db_amplitude = 10**(-1/20) -> op1db_in == 1.0 exactly -- this
        makes beta_a and the output both directly hand-checkable with a
        single exponentiation, no division/squaring needed:

            beta_a  = (10**(1/20) - 1) / 1**2 = 10**(1/20) - 1   (~0.122018)
            gain(1) = 1 / (1 + beta_a) = 1 / 10**(1/20) = 10**(-1/20)
            output  = 1 * gain(1) = 10**(-1/20) == op1db_amplitude  (exact)
        """
        op1db = _ONE_DB_RATIO   # ~0.8912509
        model = SalehModel.from_op1db_oip3(op1db_amplitude=op1db)
        expected_beta_a = 10 ** (1.0 / 20.0) - 1.0
        assert model.beta_a == pytest.approx(expected_beta_a)

        op1db_in = 1.0   # op1db / _ONE_DB_RATIO == 1.0 by construction
        output = model.gain(np.array([op1db_in]))[0] * op1db_in
        assert output == pytest.approx(op1db)

    # -- oip3_amplitude only --------------------------------------------------

    def test_oip3_only_fits_classic_2param(self):
        """
        beta_a fixed by the asymptotic two-tone IP3 definition. Baseband/
        envelope factor is 1.0, NOT the real-axis 4/3 (see
        SalehRealAxisModel below and nonlinear/saleh.py's module docstring).
        Since alpha_a is always 1.0, oip3_in == oip3_amplitude exactly (no
        conversion needed).
        """
        oip3 = 6.0
        m = SalehModel.from_op1db_oip3(oip3_amplitude=oip3)
        assert m.alpha_a == 1.0
        assert m.beta_a == pytest.approx(1.0 / oip3**2)

    def test_oip3_only_implies_op1db_10p1db_below(self):
        """
        OIP3-only fitting does not leave OP1dB free -- solving this model's
        OWN compression_point_amplitude() and converting to output-referred
        gives, in closed form (see nonlinear/saleh.py module docstring's
        "OIP3-only implies a specific OP1dB"):

            OP1dB/OIP3 = 10**(-1/20) * sqrt((10**(1/20)-1) / _OIP3_BETA_FACTOR)

        which for SalehModel's _OIP3_BETA_FACTOR=1.0 works out to ~10.14dB
        below OIP3 -- a DIFFERENT number from a plain cubic Volterra
        polynomial's ~10.6dB (see TestVolterraModel below): same asymptotic
        (leading-order IP3) behavior, different nonlinearity SHAPE, so a
        different actual compression point.
        """
        oip3 = 5.0
        m = SalehModel.from_op1db_oip3(oip3_amplitude=oip3)
        a1db_in = m.compression_point_amplitude()
        op1db = a1db_in * m.gain(np.array([a1db_in]))[0]
        ratio_db = 20 * np.log10(op1db / oip3)

        expected_ratio = _ONE_DB_RATIO * np.sqrt(
            (10 ** (1.0 / 20.0) - 1.0) / SalehModel._OIP3_BETA_FACTOR
        )
        expected_db = 20 * np.log10(expected_ratio)
        assert ratio_db == pytest.approx(expected_db, abs=1e-9)
        assert ratio_db == pytest.approx(-10.1357, abs=0.001)

    def test_describing_function_coefficient(self):
        """
        Verify the 3/4 describing function coefficient (PRD §5.1) via
        SalehModel's own baseband beta_a-from-OIP3 formula.

        For f(x) = x + a·x³ (real-axis cubic), the complex baseband AM-AM
        gives A_out = A + (3a/4)·A³, i.e. baseband a_{3,0} = (3/4)·c where
        c = -4/(3·A_IP3²) is the real-axis coefficient -- the (3/4) and
        (4/3) factors cancel exactly, so the baseband coefficient (and thus
        alpha_a*beta_a) is simply -1/A_IP3² -- NOT -(4/3)/A_IP3² (that
        factor belongs to the real-axis case -- see SalehRealAxisModel and
        nonlinear/saleh.py's module docstring).
        """
        a_oip3 = 1.0
        c = -4.0 / (3.0 * a_oip3**2)
        expected_a30 = (3.0 / 4.0) * c   # = -1/A_IP3^2 = -1.0

        m = SalehModel.from_op1db_oip3(oip3_amplitude=a_oip3)
        a30 = -m.alpha_a * m.beta_a   # small-A expansion: G[A]*A ≈ alpha_a*A - alpha_a*beta_a*A^3
        assert abs(a30 - expected_a30) < 1e-6, (
            f"k=3 coefficient should be {expected_a30:.4f}, got {a30:.4f}"
        )

    def test_max_monotonic_amplitude_and_overdrive_warning(self):
        """
        Even this classic 2-parameter form has a genuine breakdown
        amplitude: raw output y(A)=alpha_a*A/(1+beta_a*A^2) peaks at
        A=1/sqrt(beta_a) and DECLINES beyond that -- see module docstring's
        "max_monotonic_amplitude" section (this fixes a real bug in an
        earlier gamma_a-extended version of this class, which incorrectly
        returned inf here for every single-point fit).
        """
        m = SalehModel.from_op1db_oip3(oip3_amplitude=1.0)
        expected = 1.0 / np.sqrt(m.beta_a)
        assert m.max_monotonic_amplitude == pytest.approx(expected)
        with pytest.warns(UserWarning, match="exceeds the amplitude"):
            m.apply_baseband(np.array([2.0 * expected + 0j]))


# ---------------------------------------------------------------------------
# Saleh real-axis model tests
# ---------------------------------------------------------------------------

class TestSalehRealAxisModel:

    def test_supports_real_axis_not_baseband(self):
        m = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=1.3)
        assert m.supports_real_axis is True
        assert m.supports_baseband is False

    def test_apply_baseband_raises(self):
        m = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=1.3)
        with pytest.raises(NotImplementedError):
            m.apply_baseband(np.array([0.1 + 0j]))

    def test_enable_am_pm_not_supported(self):
        """No separate envelope/phase representation on the real axis."""
        with pytest.raises(ValueError, match="no AM-PM mechanism"):
            SalehRealAxisModel.from_op1db_oip3(
                oip3_amplitude=1.3, enable_am_pm=True, am_pm_peak_deg=10.0,
            )

    def test_from_op1db_oip3_takes_no_gain_argument(self):
        with pytest.raises(TypeError):
            SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=1.0, small_signal_gain=2.0)

    def test_from_op1db_oip3_requires_exactly_one_both_given(self):
        with pytest.raises(ValueError, match="not both"):
            SalehRealAxisModel.from_op1db_oip3(op1db_amplitude=0.5, oip3_amplitude=1.3)

    # -- op1db_amplitude only -----------------------------------------------

    def test_op1db_only_compression_point(self):
        """
        Same exact-1dB-at-input-referred-P1dB check as
        TestSalehModel.test_op1db_only_compression_point, applied via
        apply_real_axis() instead of gain() -- the op1db-only beta_a
        formula is domain-independent (see nonlinear/saleh.py module
        docstring), so this should behave identically to the baseband case.
        """
        op1db = 0.5
        model = SalehRealAxisModel.from_op1db_oip3(op1db_amplitude=op1db)
        assert model.alpha_a == 1.0
        op1db_in = op1db / _ONE_DB_RATIO
        output_at_op1db = model.apply_real_axis(np.array([op1db_in]))[0]
        assert output_at_op1db == pytest.approx(op1db)

    def test_op1db_only_hand_checkable_clean_numbers(self):
        """
        op1db_amplitude = 10**(-1/20) -> op1db_in == 1.0 exactly -- same
        clean-number check as TestSalehModel's, applied on the real axis.
        """
        op1db = _ONE_DB_RATIO
        model = SalehRealAxisModel.from_op1db_oip3(op1db_amplitude=op1db)
        expected_beta_a = 10 ** (1.0 / 20.0) - 1.0
        assert model.beta_a == pytest.approx(expected_beta_a)

        op1db_in = 1.0
        output = model.apply_real_axis(np.array([op1db_in]))[0]
        assert output == pytest.approx(op1db)

    # -- oip3_amplitude only --------------------------------------------------

    def test_oip3_only_uses_four_thirds_factor(self):
        """
        Real-axis beta_a-from-OIP3 uses the SAME 4/3 factor as
        VolterraModel's real-axis cubic term -- unlike the baseband
        SalehModel, which uses 1.0 (see nonlinear/saleh.py module docstring
        and TestSalehModel.test_oip3_only_fits_classic_2param). Since
        alpha_a is always 1.0, oip3_in == oip3_amplitude exactly.
        """
        oip3 = 6.0
        m = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=oip3)
        assert m.beta_a == pytest.approx((4.0 / 3.0) / oip3**2)

    def test_oip3_only_implies_op1db_close_to_baseband(self):
        """
        The real-axis model's TRUE (single-tone CW fundamental, found via
        bisection + FFT -- see module docstring) compression point should
        land close to the SAME ~10.1dB-below-OIP3 figure as the baseband
        model (this is the physical content of the "4/3 factor" claim --
        see TestSalehBasebandRealAxisEquivalence for the fuller sweep). The
        two rational curves (baseband beta_a=1/oip3², real-axis
        beta_a=(4/3)/oip3²) are not algebraically identical, so exact
        agreement isn't expected -- only close agreement.
        """
        oip3 = 1.0
        ra = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=oip3)
        a1db_in = _find_compression_point(ra.apply_real_axis, ra.alpha_a)
        op1db = _fundamental_gain(ra.apply_real_axis, a1db_in) * a1db_in
        ratio_db = 20 * np.log10(op1db / oip3)

        baseband = SalehModel.from_op1db_oip3(oip3_amplitude=oip3)
        a1db_in_bb = baseband.compression_point_amplitude()
        op1db_bb = a1db_in_bb * baseband.gain(np.array([a1db_in_bb]))[0]
        expected_db = 20 * np.log10(op1db_bb / oip3)

        assert expected_db == pytest.approx(-10.1357, abs=0.001)
        assert ratio_db == pytest.approx(expected_db, abs=0.15)

    def test_linear_passthrough_at_small_amplitude(self):
        """At very small amplitude, output should be ≈ input (alpha_a=1.0)."""
        m = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=1.3)
        x = np.array([0.001])
        out = m.apply_real_axis(x)
        assert abs(out[0] / x[0] - 1.0) < 1e-3

    def test_compression_at_large_amplitude(self):
        """Gain should compress at large (but not overdriven) amplitude."""
        m = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=1.3)
        assert m.gain(np.array([0.3]))[0] < m.gain(np.array([0.001]))[0]

    def test_warns_when_overdriven(self):
        m = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=1.3)
        expected = 1.0 / np.sqrt(m.beta_a)
        assert m.max_monotonic_amplitude == pytest.approx(expected)
        with pytest.warns(UserWarning, match="exceeds the amplitude"):
            m.apply_real_axis(np.array([2.0 * expected]))

    def test_generates_third_harmonic(self):
        """
        The whole point of the real-axis variant: applying G[x(t)]*x(t)
        directly to a real sinusoid must generate harmonic content (unlike
        the baseband/envelope SalehModel, which only ever produces in-band
        output) -- verified here by checking a 3rd-harmonic FFT component
        appears at the leading-order-predicted amplitude for a small-signal
        single tone. y = a1*x + a3*x^3 + ... with a3 = -alpha_a*beta_a (small
        A regime); x(t)=A*cos(w0 t) gives a 3rd-harmonic component of
        amplitude (a3/4)*A^3 (from cos^3 = (3/4)cos+(1/4)cos3).
        """
        m = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=5.0)
        a3 = -m.alpha_a * m.beta_a

        A = 0.05   # small enough that a3*x^3 dominates
        fs, f0, n_cycles = 1000.0, 10.0, 50
        t = np.arange(0, n_cycles / f0, 1 / fs)
        x = A * np.cos(2 * np.pi * f0 * t)
        y = m.apply_real_axis(x)

        Y = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(t), 1 / fs)
        i3 = np.argmin(np.abs(freqs - 3 * f0))
        amp_3rd = 2 * np.abs(Y[i3]) / len(t)

        expected_3rd = abs(a3) * A**3 / 4.0
        assert amp_3rd == pytest.approx(expected_3rd, rel=0.05)


# ---------------------------------------------------------------------------
# Baseband vs. real-axis Saleh equivalence (same OIP3/OP1dB spec should
# describe the same physical amplifier)
# ---------------------------------------------------------------------------

class TestSalehBasebandRealAxisEquivalence:
    """
    Verifies that SalehModel (baseband) and SalehRealAxisModel (real-axis),
    built from the SAME OIP3 or OP1dB point, represent the SAME physical
    amplifier: the real-axis model's TRUE single-tone CW fundamental
    response (FFT-extracted from an actual simulated sinusoid through
    apply_real_axis()) should track the baseband model's gain(A) curve
    closely across a range of amplitudes -- confirming the 4/3 real-axis
    OIP3-beta factor (nonlinear/saleh.py module docstring) does what it's
    supposed to do. Not EXACT agreement is expected (these are two
    different rational functions, only guaranteed to match at the leading
    asymptotic/IP3 order) -- but close (~1%) agreement well into
    compression.
    """

    def test_oip3_spec_fundamental_gain_tracks_baseband_across_sweep(self):
        oip3 = 1.0
        baseband = SalehModel.from_op1db_oip3(oip3_amplitude=oip3)
        realaxis = SalehRealAxisModel.from_op1db_oip3(oip3_amplitude=oip3)

        for A in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
            g_bb = baseband.gain(np.array([A]))[0]
            g_ra = _fundamental_gain(realaxis.apply_real_axis, A)
            assert g_ra == pytest.approx(g_bb, rel=0.01), (
                f"At A={A}: real-axis fundamental gain {g_ra:.6f} should "
                f"track baseband gain {g_bb:.6f} closely (same OIP3={oip3})"
            )

    def test_op1db_spec_fundamental_gain_tracks_baseband_across_sweep(self):
        """Same check, calibrated from OP1dB instead of OIP3 -- confirms
        this isn't an OIP3-specific coincidence. (The op1db-only beta_a
        formula happens to be domain-independent, so both models share the
        exact same beta_a here -- but the real-axis model's true CW-
        fundamental response still isn't algebraically gain(A)*A once
        you're off the calibration point itself, so still worth checking.)"""
        op1db = 0.3
        baseband = SalehModel.from_op1db_oip3(op1db_amplitude=op1db)
        realaxis = SalehRealAxisModel.from_op1db_oip3(op1db_amplitude=op1db)
        assert baseband.beta_a == pytest.approx(realaxis.beta_a)

        for A in [0.01, 0.05, 0.1, 0.15]:
            g_bb = baseband.gain(np.array([A]))[0]
            g_ra = _fundamental_gain(realaxis.apply_real_axis, A)
            assert g_ra == pytest.approx(g_bb, rel=0.015), (
                f"At A={A}: real-axis fundamental gain {g_ra:.6f} should "
                f"track baseband gain {g_bb:.6f} closely (same op1db={op1db})"
            )


# ---------------------------------------------------------------------------
# Volterra tests
# ---------------------------------------------------------------------------

class TestVolterraModel:

    def test_linear_passthrough_at_small_amplitude(self):
        """At very small amplitude, Volterra output should be ≈ linear."""
        v = VolterraModel(
            option="describing",
            oip3_amplitude=1.3,
            memory_depth=0,
        )
        x_small = np.ones(100, dtype=float) * 0.001
        out = v.apply_real_axis(x_small)
        gain = np.mean(out[10:]) / 0.001
        assert abs(gain - 1.0) < 0.01

    def test_compression_at_large_amplitude(self):
        """
        At large (but not overdriven) amplitude, output/input ratio should
        be less than small-signal gain. x_large is kept below
        v.max_monotonic_amplitude (~0.65 for oip3_amplitude=1.3, gain=1.0)
        deliberately -- see test_run_volterra_describing_warns_when_overdriven
        in tests/test_engine.py for the separate, deliberately-overdriven case.
        """
        v = VolterraModel(
            option="describing",
            oip3_amplitude=1.3,
            memory_depth=0,
        )
        x_small = np.ones(50) * 0.001
        x_large = np.ones(50) * 0.45
        out_small = v.apply_real_axis(x_small)
        out_large = v.apply_real_axis(x_large)
        gain_small = np.mean(out_small[10:]) / 0.001
        gain_large = np.mean(out_large[10:]) / 0.45
        assert gain_large < gain_small, "Gain should compress at large amplitude."

    def test_supports_real_axis_not_baseband(self):
        v = VolterraModel(option="describing", oip3_amplitude=1.3)
        assert v.supports_real_axis is True
        assert v.supports_baseband is False

    def test_describing_requires_exactly_one_neither_given(self):
        with pytest.raises(ValueError, match="requires exactly one of"):
            VolterraModel(option="describing")

    def test_describing_requires_exactly_one_both_given(self):
        """Both op1db_amplitude and oip3_amplitude -> ValueError. This
        codebase no longer supports a 5th-order fit hitting both exactly
        (removed for simplicity, sticking with the plain 3rd-order cubic)."""
        with pytest.raises(ValueError, match="not both"):
            VolterraModel(option="describing", op1db_amplitude=0.5, oip3_amplitude=1.3)

    # -- oip3_amplitude only --------------------------------------------------

    def test_describing_oip3_only_fits_plain_cubic(self):
        """
        checked via the public apply_real_axis() API against the same
        formula _build_describing_coeff() uses. a1 is always fixed at 1.0
        (no small_signal_gain parameter -- see nonlinear/volterra.py module
        docstring), so output-referred oip3_amplitude coincides with the
        input-referred value the underlying cubic fit uses directly.
        """
        oip3 = 6.0
        v = VolterraModel(option="describing", oip3_amplitude=oip3, memory_depth=0)
        a1 = 1.0
        a3 = -(4.0 / 3.0) * a1 / oip3**2
        A = 0.5
        expected = a1 * A + a3 * A**3
        actual = v.apply_real_axis(np.array([A]))[0]
        assert actual == pytest.approx(expected)

    def test_oip3_only_implies_op1db_10p6db_below(self):
        """
        OIP3-only fitting does not leave OP1dB free either -- the PHYSICAL
        (single-tone CW fundamental) compression point is fully determined
        by a3 alone. See nonlinear/volterra.py module docstring's
        "OIP3-only implies a specific OP1dB" for the full derivation
        (closed form below), including why this is a DIFFERENT question
        from where the raw polynomial itself crosses -1dB (what
        op1db_amplitude fits/verifies elsewhere in this file).
        """
        oip3 = 5.0
        v = VolterraModel(option="describing", oip3_amplitude=oip3, memory_depth=0)

        # closed form: OP1dB/OIP3 = sqrt(1 - 10**(-1/20)) * 10**(-1/20)
        ratio = np.sqrt(1.0 - _ONE_DB_RATIO) * _ONE_DB_RATIO
        expected_db = 20 * np.log10(ratio)
        assert expected_db == pytest.approx(-10.6357, abs=0.001)

        # cross-check via actual single-tone simulation + FFT fundamental
        # extraction (independent of the closed form above); the model's
        # fixed small-signal gain (a1, always 1.0 -- see module docstring)
        # is the target _find_compression_point bisects against.
        a1db_in = _find_compression_point(v.apply_real_axis, small_signal_gain=1.0)
        op1db = _fundamental_gain(v.apply_real_axis, a1db_in) * a1db_in
        ratio_db_sim = 20 * np.log10(op1db / oip3)
        assert ratio_db_sim == pytest.approx(expected_db, abs=0.02)

    # -- op1db_amplitude only -----------------------------------------------

    def test_describing_op1db_only_hits_output_referred_p1db(self):
        """
        op1db_amplitude is OUTPUT-referred: the actual compressed output
        amplitude at the 1dB point, not the input amplitude that produced
        it -- confirmed here by checking that applying the model at the
        (internally-converted) input-referred amplitude yields an output
        that equals op1db_amplitude itself, exactly 10**(-1/20) below
        where linear (small-signal) scaling would have put it. a1 is always
        fixed at 1.0 (no small_signal_gain parameter -- see
        nonlinear/volterra.py module docstring).
        """
        op1db = 3.0
        v = VolterraModel(option="describing", op1db_amplitude=op1db, memory_depth=0)
        op1db_in = op1db / _ONE_DB_RATIO
        output_at_op1db = v.apply_real_axis(np.array([op1db_in]))[0]
        assert output_at_op1db == pytest.approx(op1db)

    def test_describing_op1db_only_hand_checkable_clean_numbers(self):
        """
        a1=1.0 (fixed), op1db_amplitude=10**(-1/20) -> op1db_in == 1.0
        exactly, so a3 = (one_db_ratio - 1) directly (no division/squaring):

            a3     = 1*(10**(-1/20) - 1) / 1**2 = 10**(-1/20) - 1  (~-0.108749)
            output = a1*1 + a3*1**3 = 1 + (10**(-1/20) - 1) = 10**(-1/20)
                   == op1db_amplitude   (exact)
        """
        op1db = _ONE_DB_RATIO
        v = VolterraModel(option="describing", op1db_amplitude=op1db, memory_depth=0)
        op1db_in = 1.0   # op1db / _ONE_DB_RATIO == 1.0 by construction
        output_at_op1db = v.apply_real_axis(np.array([op1db_in]))[0]
        assert output_at_op1db == pytest.approx(op1db)


class TestTabulatedModel:

    # -- construction validation ---------------------------------------

    def test_requires_table_start_at_origin_amplitude(self):
        with pytest.raises(ValueError, match="start at exactly"):
            TabulatedModel(amplitude=[0.1, 1.0], output_amplitude=[0.0, 0.9])

    def test_requires_table_start_at_origin_output(self):
        with pytest.raises(ValueError, match="start at exactly"):
            TabulatedModel(amplitude=[0.0, 1.0], output_amplitude=[0.05, 0.9])

    def test_requires_matching_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            TabulatedModel(amplitude=[0.0, 0.5, 1.0], output_amplitude=[0.0, 0.9])

    def test_requires_at_least_two_points(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            TabulatedModel(amplitude=[0.0], output_amplitude=[0.0])

    def test_requires_strictly_ascending_amplitude(self):
        with pytest.raises(ValueError, match="strictly ascending"):
            TabulatedModel(amplitude=[0.0, 1.0, 1.0], output_amplitude=[0.0, 0.9, 0.95])

    def test_phase_rad_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length as amplitude"):
            TabulatedModel(
                amplitude=[0.0, 1.0], output_amplitude=[0.0, 0.9], phase_rad=[0.0]
            )

    # -- baseband ---------------------------------------------------------

    def test_baseband_exact_at_table_points(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0, 2.0], output_amplitude=[0.0, 0.9, 1.5]
        )
        u = np.array([1.0 + 0j, 2.0 + 0j])
        out = model.apply_baseband(u)
        assert np.abs(out[0]) == pytest.approx(0.9)
        assert np.abs(out[1]) == pytest.approx(1.5)

    def test_baseband_linear_interpolation_between_points(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0, 2.0], output_amplitude=[0.0, 0.9, 1.5]
        )
        u = np.array([1.5 + 0j])
        out = model.apply_baseband(u)
        assert np.abs(out[0]) == pytest.approx((0.9 + 1.5) / 2)

    def test_baseband_preserves_phasor_direction_when_no_am_pm(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0, 2.0], output_amplitude=[0.0, 0.9, 1.5]
        )
        u = np.array([1j * 1.0])   # pure +j phasor, amplitude 1.0
        out = model.apply_baseband(u)
        assert out[0] == pytest.approx(0.9j)

    def test_baseband_am_pm_applied(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0, 2.0],
            output_amplitude=[0.0, 0.9, 1.5],
            phase_rad=[0.0, 0.2, 0.5],
        )
        u = np.array([1.0 + 0j])
        out = model.apply_baseband(u)
        expected = 0.9 * np.exp(1j * 0.2)
        assert out[0] == pytest.approx(expected)

    def test_baseband_zero_amplitude_safe(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0], output_amplitude=[0.0, 0.9]
        )
        out = model.apply_baseband(np.array([0.0 + 0j]))
        assert out[0] == pytest.approx(0.0)

    # -- real-axis ----------------------------------------------------------

    def test_real_axis_matches_baseband_magnitude_at_table_points(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0, 2.0], output_amplitude=[0.0, 0.9, 1.5]
        )
        out = model.apply_real_axis(np.array([1.0, 2.0]))
        assert out[0] == pytest.approx(0.9)
        assert out[1] == pytest.approx(1.5)

    def test_real_axis_odd_symmetry(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0, 2.0], output_amplitude=[0.0, 0.9, 1.5]
        )
        pos = model.apply_real_axis(np.array([0.5, 1.5]))
        neg = model.apply_real_axis(np.array([-0.5, -1.5]))
        assert neg == pytest.approx(-pos)

    def test_real_axis_ignores_am_pm_table_without_error(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0], output_amplitude=[0.0, 0.9], phase_rad=[0.0, 0.5]
        )
        out = model.apply_real_axis(np.array([1.0]))
        assert out[0] == pytest.approx(0.9)

    # -- extrapolation warning -----------------------------------------------

    def test_warns_when_peak_exceeds_table_range(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0], output_amplitude=[0.0, 0.9]
        )
        with pytest.warns(UserWarning, match="calibrated range"):
            model.apply_baseband(np.array([2.0 + 0j]))

    def test_no_warning_within_table_range(self):
        model = TabulatedModel(
            amplitude=[0.0, 1.0], output_amplitude=[0.0, 0.9]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.apply_baseband(np.array([0.9 + 0j]))   # should not raise

    # -- non-monotonic table (the motivating AOM use case) -------------------

    def test_non_monotonic_table_not_rejected(self):
        """
        An AOM's RF-power-to-diffracted-amplitude response is sinusoidal
        (A_out = sin(kappa*A_in)) and genuinely turns over past its first
        diffraction maximum -- unlike Saleh/Volterra's max_monotonic_amplitude
        checks, TabulatedModel must not flag or misreport this: it's a
        legitimate, intentional table shape, not an overdrive bug signal.
        """
        kappa = np.pi / 2.0   # turnover (peak) at amplitude = 1.0
        amp = np.linspace(0.0, 2.0, 41)
        out_amp = np.sin(kappa * amp)
        model = TabulatedModel(amplitude=amp, output_amplitude=out_amp)

        peak_output = model.apply_real_axis(np.array([1.0]))[0]
        past_turnover_output = model.apply_real_axis(np.array([1.8]))[0]
        assert past_turnover_output < peak_output, (
            "Table must faithfully reproduce the turnover, not just the "
            "monotonic rising part."
        )
        # No warning should fire -- 1.8 is within the table's own range.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.apply_real_axis(np.array([1.8]))

    # -- small_signal_gain / gain-convention warning -------------------------

    def test_small_signal_gain_matches_second_point_ratio(self):
        model = TabulatedModel(
            amplitude=[0.0, 0.5, 2.0], output_amplitude=[0.0, 0.45, 1.5]
        )
        assert model.small_signal_gain == pytest.approx(0.9)

    def test_registry_warns_on_off_unity_small_signal_gain(self):
        annotation = {
            "NL1": {
                "model": "table",
                "amplitude": [0.0, 0.5, 2.0],
                "output_amplitude": [0.0, 0.6, 1.5],   # gain ~1.2 at small signal
            }
        }
        warnings_list = []
        build_nonlinear_nodes(annotation, mode="complex_baseband", warnings_list=warnings_list)
        assert any("small-signal gain" in w for w in warnings_list)

    # -- registry integration ------------------------------------------------

    def test_registry_builds_table_model_baseband(self):
        annotation = {
            "NL1": {
                "model": "table",
                "amplitude": [0.0, 1.0, 2.0],
                "output_amplitude": [0.0, 0.9, 1.5],
            }
        }
        nodes = build_nonlinear_nodes(annotation, mode="complex_baseband")
        assert isinstance(nodes["NL1"], TabulatedModel)
        out = nodes["NL1"].apply_baseband(np.array([1.0 + 0j]))
        assert np.abs(out[0]) == pytest.approx(0.9)

    def test_registry_builds_table_model_real_axis(self):
        annotation = {
            "NL1": {
                "model": "table",
                "amplitude": [0.0, 1.0, 2.0],
                "output_amplitude": [0.0, 0.9, 1.5],
            }
        }
        nodes = build_nonlinear_nodes(annotation, mode="real_axis")
        assert isinstance(nodes["NL1"], TabulatedModel)
        out = nodes["NL1"].apply_real_axis(np.array([1.0]))
        assert out[0] == pytest.approx(0.9)

    def test_registry_builds_table_model_real_axis_out_of_order(self):
        annotation = {
            "NL1": {
                "model": "table",
                "amplitude": [0.0, 2.0, 1.0],
                "output_amplitude": [0.0, 1.5, 0.9],
            }
        }
        with pytest.raises(ValueError):
            build_nonlinear_nodes(annotation, mode="real_axis")


    def test_registry_rejects_unknown_model_mentions_table(self):
        annotation = {"NL1": {"model": "not_a_model"}}
        with pytest.raises(ValueError, match="'table'"):
            build_nonlinear_nodes(annotation, mode="complex_baseband")


if __name__ == "__main__":
    pytest.main([__file__])
