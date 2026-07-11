"""
si_qfi.nonlinear.volterra
=========================
Volterra series nonlinear model (real-axis mode only).

Implements a third-order Volterra series truncation operating on the full
real RF waveform. Three ways to build one (the `option` constructor
argument) -- these are NOT three variations on the same math, they're three
genuinely different ways of specifying the model, for three different
situations:

Option 'diagonal' — you already have fitted/measured polynomial
coefficients (memory polynomial on the real axis):
    y[n] = Σ_{p in orders} Σ_{m=0}^{M} a_{pm} · x[n-m]^p
    You supply `coefficients` (shape (len(orders), M+1)) and `orders`
    directly -- e.g. from your own curve-fitting to measured data. Any
    orders (odd or even) and any memory depth M are allowed. This is the
    most general/least convenient option: you do the fitting yourself.

Option 'full_kernel' — you have (or want to model) a genuine 3-D
third-order kernel, not just a diagonal (memoryless-per-tap) one:
    y[n] = Σ_τ h₁[τ]·x[n-τ]
           + Σ_{τ₁,τ₂,τ₃} h₃[τ₁,τ₂,τ₃]·x[n-τ₁]·x[n-τ₂]·x[n-τ₃]
    h₃ is the full symmetric kernel, shape (M+1, M+1, M+1) -- lets off-
    diagonal terms (cross-products of different delays) contribute, which
    'diagonal' cannot represent. O(M³) storage/compute — keep M small (≤20).
    NOTE: h1/set_h1() only has an effect here; 'diagonal' and 'describing'
    both ignore h1 entirely (they dispatch to _apply_diagonal(), which only
    reads self._coeff).

Option 'describing' (default) — you only have the numbers most real
amplifier datasheets actually give you: OP1dB or OIP3 (OUTPUT-referred
P1dB / third-order intercept -- the actual output amplitude at that point,
not the input amplitude that produced it; ALL nonlinearity specs in this
codebase are output-referred by convention, not input-referred). The
linear (k=1) coefficient a1 is always fixed at 1.0 -- not a constructor
parameter -- so output-referred and input-referred amplitudes coincide for
that term; see _build_describing_coeff()). Builds a `'diagonal'` model for
you (memory taps at m=0 only), a plain cubic (orders [1,3]). EXACTLY ONE of
op1db_amplitude/oip3_amplitude is required (raises if given neither or
both) -- a single free coefficient (a3, since a1 is fixed at 1.0) can only
be calibrated from ONE point; there is no second free coefficient here to
also hit a second point (see "OIP3-only implies a specific OP1dB" below).

CAVEAT: fitting a polynomial to hit one specific point exactly does not make
it well-behaved beyond it -- a polynomial has no saturation mechanism, so
pushing the input well past the calibration point can still yield
non-physical (declining, or gain-expanding/sign-flipped) output. See
max_monotonic_amplitude below; apply_real_axis() warns if the actual
signal's peak amplitude exceeds it. (Contrast with nonlinear/saleh.py's
rational G[A] curve, which stays bounded for ALL amplitudes -- a
numerator-polynomial like this one doesn't have that property.)

OIP3-only implies a specific OP1dB (and vice versa) -- NOT a free choice
--------------------------------------------------------------------------
Since a3 has only one degree of freedom, fitting from OIP3 alone
*determines* where this model's actual 1dB compression point falls. That
point must be measured the way OP1dB is physically defined -- via the
single-tone CW FUNDAMENTAL response, not via the raw polynomial evaluated
at a constant value (those are different things for a real-axis model that
produces harmonics; see below). For a single tone x(t)=A·cos(ω₀t), the
fundamental output is a1·A + (3/4)·a3·A³ (cos³θ = (3/4)cosθ + (1/4)cos3θ),
so the FUNDAMENTAL's own 1dB point solves:
    (3/4)·a3·A_1dB,in² = a1·(10^(-1/20) - 1)
Substituting a3 = -(4/3)·a1/A_IP3,in² (this module's existing OIP3-only
formula, unaffected) gives, in closed form:
    A_1dB,in = A_IP3,in · sqrt(1 - 10^(-1/20))
    OP1dB (fundamental, output-referred) = A_1dB,in · 10^(-1/20)
    OP1dB / OIP3 = sqrt(1 - 10^(-1/20)) · 10^(-1/20)  ≈  10**(-10.64/20)
i.e. OP1dB sits **~10.6 dB below OIP3** (output-referred, in dB) for this
plain cubic model -- verified both by this closed form and independently by
simulating an actual single tone through apply_real_axis() and FFT-
extracting the fundamental (tests/test_nonlinear.py). This is DIFFERENT
from SalehModel's ~10.1dB (see nonlinear/saleh.py module docstring) because
a truncated cubic and Saleh's bounded rational G[A] are different SHAPES of
nonlinearity that only agree asymptotically (same leading-order IP3
behavior), not at the compression point.

NOTE: this ~10.6dB figure is NOT the same quantity as what
`apply_real_axis()` returns when evaluated at a raw constant input (which is
what op1db_amplitude ITSELF is fit/verified against elsewhere in this
module and its tests) -- that raw evaluation is a quasi-static/DC-like
curve, not a physical single-tone CW measurement (it doesn't separate the
fundamental from the 3rd harmonic the way an actual sinusoidal drive does).
Both are legitimate, just different, questions: "where does the RAW
polynomial itself cross -1dB" (what op1db_amplitude fits/verifies) vs.
"where does an actual CW tone's FUNDAMENTAL, after harmonic filtering, cross
-1dB" (what the classic ~9.6dB/~10.6dB literature figures, and this
derivation, are about).

Gain convention (PRD §3.6)
---------------------------
A device's linear gain (and, for 'full_kernel', its linear filtering shape)
belongs in the SI schematic, not in this model — the engine already
convolves the signal with the schematic's segment transfer function before
and after this node. Consequently:
  - 'describing': the k=1, m=0 coefficient (a1) is always fixed at 1.0 --
    not a constructor parameter at all (removed; there is no non-unity case
    to default away from, same reasoning as SalehModel.from_op1db_oip3()
    taking no gain argument -- see nonlinear/saleh.py).
  - 'diagonal': you supply `coefficients` yourself, so the k=1, m=0 entry
    (if order 1 is included) should stay ≈1.0 by the same convention --
    not enforced at construction time here, but checked at runtime (below).
  - 'full_kernel': h1 should remain the default unit impulse [1.0] (identity)
    unless intentionally modeling *additional* dispersion beyond what the
    schematic's own transfer function already captures — setting h1 to a
    copy of that same segment transfer function would double-apply it.
`siq.run()` warns if a 'diagonal' model's small_signal_gain (see the
read-only property below) deviates from 1.0 by more than ~3 dB (not checked
for 'full_kernel', where gain is embedded in an array rather than a scalar;
not applicable to 'describing', which is always exactly 1.0 by
construction).
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional
from scipy.signal import fftconvolve

from .base import NonlinearNode


class VolterraModel(NonlinearNode):
    """
    Third-order Volterra series model for real RF waveform propagation.

    Parameters
    ----------
    h1 : np.ndarray, float64, shape (L1,)
        First-order (linear) kernel — used only by option='full_kernel'
        (see module docstring; ignored by 'diagonal' and 'describing').
        Defaults to delta[0] (identity): the linear channel response is
        already applied by the engine's own segment convolution, so h1
        should not duplicate it — see gain convention in the module
        docstring.
    option : str
        'diagonal'    — diagonal kernel; requires coefficients.
        'full_kernel' — full h3 tensor; requires h3.
        'describing'  — h1 from SI + h3 derived from op1db_amplitude, oip3_amplitude.
    coefficients : np.ndarray, shape (n_orders, M+1), optional
        For option='diagonal'. Row per order (1,2,3,...), column per memory tap.
    orders : list of int, optional
        Orders included in diagonal kernel. Default [1, 2, 3].
    h3 : np.ndarray, float64, shape (M+1, M+1, M+1), optional
        Full symmetric third-order kernel for option='full_kernel'.
    op1db_amplitude : float, optional
        For option='describing': OUTPUT-referred P1dB amplitude (same units
        as waveform) -- i.e. the actual (compressed) output amplitude at
        the 1dB compression point, not the input amplitude that produced
        it. The linear (a1) coefficient is always fixed at 1.0 (see module
        docstring), so this is also the input-referred value the underlying
        cubic fit uses directly. EXACTLY one of op1db_amplitude/
        oip3_amplitude is required (raises if given neither or both) --
        fits a plain cubic (orders [1,3]) from that one point.
    oip3_amplitude : float, optional
        For option='describing': OUTPUT-referred third-order intercept
        amplitude (OIP3) -- same note as op1db_amplitude (a1 fixed at 1.0,
        so no separate input-referred conversion). Mutually exclusive with
        op1db_amplitude.
    memory_depth : int
        Memory depth M (samples) for diagonal / describing options.
    """

    def __init__(
        self,
        h1: Optional[np.ndarray] = None,
        option: str = "describing",
        coefficients: Optional[np.ndarray] = None,
        orders: Optional[list] = None,
        h3: Optional[np.ndarray] = None,
        op1db_amplitude: Optional[float] = None,
        oip3_amplitude: Optional[float] = None,
        memory_depth: int = 5,
    ) -> None:
        self._h1 = np.asarray(h1, dtype=float) if h1 is not None else np.array([1.0])
        self._option = option
        self._M = int(memory_depth)

        if option == "diagonal":
            if coefficients is None:
                raise ValueError("option='diagonal' requires coefficients.")
            self._coeff = np.asarray(coefficients, dtype=float)
            self._orders = list(orders) if orders is not None else list(range(1, self._coeff.shape[0] + 1))
            self._max_monotonic_amplitude = self._compute_max_monotonic_amplitude()

        elif option == "full_kernel":
            if h3 is None:
                raise ValueError("option='full_kernel' requires h3.")
            self._h3 = np.asarray(h3, dtype=float)
            if self._h3.ndim != 3:
                raise ValueError("h3 must be a 3-D array (M+1, M+1, M+1).")
            # No simple scalar gain(amplitude) curve exists for a full 3-D
            # kernel (off-diagonal cross-terms mean output isn't a function
            # of |x[n]| alone) -- overdrive detection isn't attempted here.
            self._max_monotonic_amplitude = float("inf")

        elif option == "describing":
            if op1db_amplitude is None and oip3_amplitude is None:
                raise ValueError(
                    "option='describing' requires exactly one of "
                    "op1db_amplitude or oip3_amplitude -- neither was given."
                )
            if op1db_amplitude is not None and oip3_amplitude is not None:
                raise ValueError(
                    "option='describing' accepts exactly one of "
                    "op1db_amplitude / oip3_amplitude, not both -- a plain "
                    "cubic has only one free coefficient (a3), which can be "
                    "calibrated from ONE point only (see module docstring)."
                )
            self._op1db = float(op1db_amplitude) if op1db_amplitude is not None else None
            self._oip3 = float(oip3_amplitude) if oip3_amplitude is not None else None
            self._coeff, self._orders = self._build_describing_coeff()
            self._max_monotonic_amplitude = self._compute_max_monotonic_amplitude()

        else:
            raise ValueError(f"Unknown option '{option}'. Use 'diagonal', 'full_kernel', or 'describing'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_describing_coeff(self):
        """
        Build diagonal kernel coefficients (plain cubic, orders [1,3]) from
        whichever ONE of OP1dB / OIP3 was supplied (PRD §5.1 -- see module
        docstring for the full derivation).

            y = a1·x + a3·x³     (odd-symmetric amplifier)

        a1 = 1.0, always (fixed -- see module docstring's "Gain convention"
        section; not a constructor parameter).

        self._op1db / self._oip3 are OUTPUT-referred (the actual output
        amplitude at the compression point / the output-referred third-order
        intercept, OIP3) -- all nonlinearity specs in this codebase are
        output-referred by convention. Since a1=1.0, output-referred and
        input-referred amplitudes coincide for the linear term, so no
        separate conversion is needed here (contrast with SalehRealAxisModel/
        VolterraModel's OIP3-only-implies-OP1dB derivation elsewhere, which
        is about a different, physical single-tone-CW-fundamental quantity,
        not this a1 normalization):
            A_IP3,in  = A_OIP3                               (a1 = 1.0)
            A_1dB,in  = A_OP1dB / 10^(-1/20)                  (compressed output
                                                              amplitude at the
                                                              input-referred
                                                              1dB point)

        Exactly one of the two is set (enforced in __init__):
          - IP3: a3 fixed by the ASYMPTOTIC two-tone IP3 definition (the 4/3
            factor comes from the two-tone cubic expansion: IM3 amplitude is
            (3/4)·a3·A³, fundamental is a1·A; IP3 is where the extrapolated
            lines a1·A and (3/4)·|a3|·A³ cross):
                a3 = -(4/3) · a1 / A_IP3,in²
            (this alone DETERMINES, not leaves free, where the model's
            actual OP1dB falls -- see module docstring's "OIP3-only implies
            a specific OP1dB" section.)
          - P1dB: a3 fixed directly from the 1dB-compression-point
            definition G[A_1dB] = a1·10^(-1/20) (raw polynomial evaluated at
            a constant input -- see module docstring's NOTE on why this is a
            different question from the single-tone-CW-fundamental
            derivation used for the OIP3-only case's implied OP1dB):
                a3 = a1·(10^(-1/20) - 1) / A_1dB,in²

        All memory taps beyond m=0 initialised to zero.
        """
        a1 = 1.0
        one_db_ratio = 10 ** (-1.0 / 20.0)   # linear amplitude ratio at -1dB

        if self._oip3 is not None:
            oip3_in = self._oip3 / a1
            a3 = -(4.0 / 3.0) * a1 / (oip3_in ** 2)
        else:
            op1db_in = self._op1db / (a1 * one_db_ratio)
            a3 = a1 * (one_db_ratio - 1.0) / (op1db_in ** 2)

        orders, coeffs_1d = [1, 3], [a1, a3]
        M = self._M
        coeff = np.zeros((len(orders), M + 1), dtype=float)
        for row, a in enumerate(coeffs_1d):
            coeff[row, 0] = a
        return coeff, orders

    def _compute_max_monotonic_amplitude(self) -> float:
        """
        Largest input amplitude for which this diagonal/describing model
        still behaves like a plausible compressive amplifier -- the
        smaller of two distinct breakdown points, both of which a
        finite polynomial (no saturation mechanism) will eventually hit as
        amplitude grows far enough past wherever it was calibrated:

        1. Raw output turns down (dy/dA < 0): the amplifier's output
           starts *decreasing* as input increases -- never true for a real
           amplifier short of hard clipping/protection circuitry.
        2. Gain expands (d(gain)/dA > 0, gain = y/A): compression starts
           *improving* as input increases -- also never true for a real
           compressive amplifier, which should compress monotonically
           worse (or at least not better) as you keep driving it harder.

        These are genuinely different failure modes depending on the sign
        of the highest-order coefficient (found the hard way: a naive
        version of this check that only looked at (1) reported "no problem"
        for an op1db/oip3 pair whose gain visibly starts *increasing* well
        before its raw output ever turns down -- see nonlinear/volterra.py
        history). Both are checked here; the smaller (more conservative)
        root wins.

        Only meaningful for the memoryless (m=0-only) part of the
        coefficients -- if any m>0 (memory) tap is non-zero, output isn't a
        function of instantaneous amplitude alone, so this check is skipped
        (returns inf) rather than giving a misleading answer.

        Returns np.inf if neither breakdown occurs for any A >= 0 (e.g. a
        pure linear gain, or coefficients that happen to stay well-behaved
        everywhere).
        """
        if self._M > 0 and np.any(self._coeff[:, 1:] != 0):
            return float("inf")

        def smallest_positive_root(deriv_by_power: dict) -> float:
            if not deriv_by_power:
                return float("inf")
            max_power = max(deriv_by_power)
            poly_coeffs = np.zeros(max_power + 1)   # index = power of A
            for power, coeff in deriv_by_power.items():
                poly_coeffs[power] = coeff
            roots = np.roots(poly_coeffs[::-1])   # np.roots wants highest-degree first
            positive_real_roots = [
                r.real for r in roots
                if abs(r.imag) < 1e-9 * max(abs(r.real), 1.0) and r.real > 0
            ]
            return min(positive_real_roots) if positive_real_roots else float("inf")

        # y(A) = sum_p a_p*A^p  =>  dy/dA = sum_p p*a_p*A^(p-1)
        dy_dA: dict[int, float] = {}
        # gain(A) = y(A)/A = sum_p a_p*A^(p-1)  =>  d(gain)/dA = sum_p (p-1)*a_p*A^(p-2)
        dgain_dA: dict[int, float] = {}
        for row, p in enumerate(self._orders):
            a = self._coeff[row, 0]
            if a == 0.0 or p == 0:
                continue
            dy_dA[p - 1] = dy_dA.get(p - 1, 0.0) + p * a
            if p != 1:
                dgain_dA[p - 2] = dgain_dA.get(p - 2, 0.0) + (p - 1) * a

        return min(smallest_positive_root(dy_dA), smallest_positive_root(dgain_dA))

    def _apply_diagonal(self, x: np.ndarray) -> np.ndarray:
        """Diagonal kernel: y[n] = Σ_p Σ_m a_{pm} · x[n-m]^p"""
        N = len(x)
        out = np.zeros(N, dtype=float)
        for row, p in enumerate(self._orders):
            for m in range(self._M + 1):
                a = self._coeff[row, m]
                if a == 0.0:
                    continue
                if m == 0:
                    x_del = x
                else:
                    x_del = np.concatenate([np.zeros(m), x[:-m]])
                out += a * (x_del ** p)
        return out

    def _apply_full_kernel(self, x: np.ndarray) -> np.ndarray:
        """
        Full third-order Volterra: y = h1*x + triple sum of h3.
        Efficient implementation: for each (τ₁, τ₂), convolve h3[:, τ₁, τ₂] with x,
        then multiply by x[n-τ₁] · x[n-τ₂] and accumulate.
        Still O(N·M²) — keep M small.
        """
        N = len(x)
        M1 = len(self._h1)
        # Linear term
        y = fftconvolve(x, self._h1)[:N]

        # Third-order term
        M = self._h3.shape[0] - 1
        x_pad = np.concatenate([np.zeros(M), x])   # pad for delayed access

        for tau1 in range(M + 1):
            x1 = x_pad[M - tau1: M - tau1 + N]
            for tau2 in range(M + 1):
                x2 = x_pad[M - tau2: M - tau2 + N]
                for tau3 in range(M + 1):
                    h = self._h3[tau1, tau2, tau3]
                    if h == 0.0:
                        continue
                    x3 = x_pad[M - tau3: M - tau3 + N]
                    y += h * x1 * x2 * x3
        return y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_h1(self, h1: np.ndarray) -> None:
        """
        Set the linear kernel from an external source (e.g. SI transfer function).
        Called by the simulation engine when option='describing'.
        """
        self._h1 = np.asarray(h1, dtype=float)

    @property
    def memory_depth(self) -> int:
        return self._M

    @property
    def supports_baseband(self) -> bool:
        return False

    @property
    def supports_real_axis(self) -> bool:
        return True

    @property
    def small_signal_gain(self) -> Optional[float]:
        """
        Linear (k=1, m=0) diagonal coefficient for 'diagonal'/'describing'.
        Always exactly 1.0 for 'describing' (fixed at construction, not
        settable). For 'diagonal', reflects whatever the caller supplied in
        `coefficients` -- should be ≈1.0 by the same convention, but not
        enforced here (nonlinear/registry.py's build_nonlinear_nodes()
        warns at runtime if it deviates). Returns None for 'full_kernel',
        where gain is embedded in the h1 array rather than a scalar.
        """
        if self._option in ("diagonal", "describing") and 1 in self._orders:
            row = self._orders.index(1)
            return float(self._coeff[row, 0])
        return None

    @property
    def max_monotonic_amplitude(self) -> float:
        """
        Largest input amplitude for which this model still behaves like a
        plausible compressive amplifier (output doesn't turn down, and gain
        doesn't start expanding) -- see _compute_max_monotonic_amplitude().
        np.inf if unbounded (e.g. a pure linear model, 'full_kernel', or a
        model with non-zero memory taps beyond m=0, where this check isn't
        attempted).
        """
        return self._max_monotonic_amplitude

    def apply_real_axis(self, v: np.ndarray) -> np.ndarray:
        """
        Apply Volterra series to real RF waveform v(t).

        Dispatches to the appropriate internal implementation based on option.
        """
        v = np.asarray(v, dtype=float)
        if self._option in ("diagonal", "describing"):
            self._warn_if_overdriven(v)
            return self._apply_diagonal(v)
        elif self._option == "full_kernel":
            return self._apply_full_kernel(v)
        else:
            raise RuntimeError(f"Unknown option: {self._option}")

    def _warn_if_overdriven(self, x: np.ndarray) -> None:
        """Warn if x's peak amplitude exceeds max_monotonic_amplitude -- see
        _compute_max_monotonic_amplitude() for what that means physically."""
        if len(x) == 0 or not np.isfinite(self._max_monotonic_amplitude):
            return
        peak = float(np.max(np.abs(x)))
        if peak > self._max_monotonic_amplitude:
            warnings.warn(
                f"SI-QFI: VolterraModel input peak amplitude ({peak:.4g}) exceeds "
                f"the amplitude ({self._max_monotonic_amplitude:.4g}) beyond which "
                f"this polynomial stops behaving like a plausible compressive "
                f"amplifier (output turning down, or gain starting to expand "
                f"instead of compress). A truncated polynomial describing "
                f"function has no saturation mechanism, so it may predict "
                f"non-physical output out here. Consider a bounded model (e.g. "
                f"'saleh' -- SalehModel in complex_baseband mode, or "
                f"SalehRealAxisModel here in real_axis mode) if operation this "
                f"far into compression needs to be trusted.",
                stacklevel=3,
            )

    def __repr__(self) -> str:
        return f"VolterraModel(option='{self._option}', M={self._M})"
