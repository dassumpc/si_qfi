"""
si_qfi.noise.realization
========================
Generate bandlimited stochastic noise realization arrays from a PSD.

Two variants:
  - Real bandpass noise  (real-axis mode): real array at RF sample rate
  - Complex baseband noise (baseband mode): complex array at baseband sample rate

The generation method is spectral shaping of white Gaussian noise:
    1. Draw white Gaussian noise in the frequency domain
    2. Multiply by sqrt(S_v(f) · df) to shape the spectrum (generate_baseband_noise
       additionally applies the bandpass-to-envelope conversion -- see its own
       docstring)
    3. IFFT to get the time-domain realization
    4. Take real part (real-axis) or keep complex (baseband)

Absolute-scale history (both functions had a real, independent x2-in-power
bug that canceled in every real_axis-vs-baseband cross-check, so neither was
caught until the two were verified separately against an outside reference):
  - generate_rf_noise(): for a flat one-sided PSD S1 over the full Nyquist
    band [0,fs/2], Var must equal S1*fs/2 -- the plain Johnson-Nyquist
    "noise power in bandwidth" formula <v^2>=S1*B applied at B=fs/2, and
    confirmed directly against scipy.signal.periodogram (an independent
    implementation) on the function's own output. The previous version gave
    S1*fs (2x too high) -- traced to amp needing an extra /sqrt(2) beyond
    what the (correct, separately-derived) Parseval-mirroring bookkeeping
    already accounted for.
  - generate_baseband_noise(): demodulating a real bandpass source with
    one-sided PSD S1 down to a complex envelope of one-sided cutoff B
    should give Var=4*S1*B, not 8*S1*B. Derivation: w(t):=LPF[v_rf(t)*
    exp(-i*wc*t)] (i.e. quantum.demodulate()'s own mixing step BEFORE its
    x2 correction) keeps only the +fc image of v_rf's spectrum and drops
    the -fc image entirely -- for genuinely broadband/incoherent content
    there is no redundant symmetric half to "recover", so Var(w)=S1*B
    (not S1*2B). demodulate() then applies x2 amplitude (x4 power) on top
    of w, giving Var=4*Var(w)=4*S1*B. The previous version gave 8*S1*B (2x
    too high) -- confirmed by demodulating the now-correctly-scaled
    generate_rf_noise() output and comparing directly against
    generate_baseband_noise()'s own output for the same S1,B (ratio landed
    on ~1 with the fix, ~2 without it).
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def generate_baseband_noise(
    n_samples: int,
    fs: float,
    psd_v2_per_hz: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate one complex baseband noise realization.

    `psd_v2_per_hz` is a physical, one-sided PSD of a real (wideband) bandpass
    noise source -- the same convention SI's own Johnson-noise formula uses
    (`sqrt(4*k_B*T*R)`), and the same value that's also handed to
    generate_rf_noise() for real_axis mode. Producing the correct complex
    envelope of such a source requires a bandpass-to-envelope conversion --
    see the module docstring's "Absolute-scale history" for the exact
    derivation (Var = 4 * S_v * B for one-sided cutoff B, matching what
    demodulating an equally-corrected generate_rf_noise() realization gives
    for the identical physical source, confirmed directly by comparing the
    two).

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples to generate.
    fs : float
        Sample rate (Hz). Determines the frequency resolution df = fs/n_samples.
    psd_v2_per_hz : np.ndarray, shape (n_samples,) or (n_samples//2+1,)
        One-sided or two-sided noise PSD [V²/Hz] evaluated at the FFT frequencies.
        If one-sided (len = n_samples//2+1), it is mirrored internally --
        S(-f) := S(f), NOT the textbook S(-f) := S(f)/2 halved convention.
        If two-sided (len = n_samples), it is used as-is, so the caller
        must supply it under the SAME un-halved convention (S(f) at each
        of a mirrored pair should equal what the one-sided form would put
        at that |f|, not half of it) to get an equivalent result -- these
        two input forms are NOT interchangeable under the standard S2=S1/2
        physics convention (confirmed empirically: feeding a properly
        halved two-sided array gives exactly half the variance of the
        equivalent one-sided array). In practice this only matters for
        direct callers of this function -- the engine pipeline
        (noise/psd.py's psd_cache_for_noise_nodes()) always builds
        one-sided arrays via np.fft.rfftfreq(), so it never exercises the
        two-sided path.
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().

    Returns
    -------
    noise : np.ndarray, complex128, shape (n_samples,)
        Complex baseband noise realization -- for a one-sided input flat
        over [0,B] (mirrored to +-B), Var = 4 * psd_v2_per_hz * B; for the
        full Nyquist band (B=fs/2) that's Var = 2 * psd_v2_per_hz * fs --
        matching real_axis mode's demodulated noise for the same physical
        source (see module docstring).
    """
    if rng is None:
        rng = np.random.default_rng()

    N = int(n_samples)
    df = fs / N

    psd = np.asarray(psd_v2_per_hz, dtype=float)

    # If one-sided, build full two-sided PSD for rfft-style usage
    if psd.shape[0] == N // 2 + 1:
        # Two-sided: mirror (for complex signal, both sides carry noise).
        # Mirror length must be N - (N//2+1): for even N that's N//2-1
        # (excludes DC AND the standalone Nyquist bin); for odd N there is
        # no standalone Nyquist bin, so it's N//2 (excludes DC only) --
        # using the even-only slice here for both parities used to leave
        # the odd case one element short (psd_two_sided.shape[0] == N-1),
        # breaking the elementwise multiply below for any odd n_samples.
        mirror = psd[-2:0:-1] if N % 2 == 0 else psd[-1:0:-1]
        psd_two_sided = np.concatenate([psd, mirror])
    elif psd.shape[0] == N:
        psd_two_sided = psd
    else:
        raise ValueError(
            f"psd_v2_per_hz length {psd.shape[0]} must be n_samples={N} "
            f"or n_samples//2+1={N//2+1}."
        )

    # Spectral amplitude: sqrt(2 * S_v * df) -- the sqrt(2) is the
    # bandpass-to-envelope conversion (see module docstring's "Absolute-
    # scale history": Var=4*S_v*B, not the naive S_v*B a same-formula-as-
    # generate_rf_noise treatment would give, nor 8*S_v*B which an
    # unchecked x2-amplitude/x4-power guess -- matching demodulate()'s own
    # x2 with no further reasoning -- gives).
    amp = np.sqrt(np.maximum(psd_two_sided * df, 0.0) * 2.0)

    # White Gaussian noise in frequency domain (complex)
    noise_f = amp * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)

    # IFFT → complex baseband time-domain realization
    noise_t = np.fft.ifft(noise_f) * N   # scale to preserve variance

    return noise_t.astype(complex)


def generate_rf_noise(
    n_samples: int,
    fs: float,
    psd_v2_per_hz: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate one real RF noise realization.

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples.
    fs : float
        Sample rate (Hz).
    psd_v2_per_hz : np.ndarray, shape (n_samples//2+1,) or (n_samples,)
        One-sided noise PSD [V²/Hz]. If given a length-n_samples array, only
        the first n_samples//2+1 entries are used, taken directly as the
        one-sided values (NOT halved) -- so, as with
        generate_baseband_noise(), a length-n_samples input is only
        equivalent to the corresponding one-sided input under this
        codebase's un-halved mirroring convention, not the textbook
        S2=S1/2 one. Same caveat applies: the engine pipeline always
        passes a one-sided array, so this only matters for direct callers.
    rng : np.random.Generator, optional

    Returns
    -------
    noise : np.ndarray, float64, shape (n_samples,)
        Real-valued bandlimited noise realization -- for a flat one-sided
        PSD S1 over the full Nyquist band [0,fs/2], Var = S1*fs/2 (the
        plain Johnson-Nyquist "noise power in bandwidth" formula
        <v^2>=S1*B at B=fs/2 -- see module docstring's "Absolute-scale
        history").
    """
    if rng is None:
        rng = np.random.default_rng()

    N = int(n_samples)
    df = fs / N

    psd = np.asarray(psd_v2_per_hz, dtype=float)

    # Work with one-sided PSD via rfft
    n_rfft = N // 2 + 1
    if psd.shape[0] == N:
        psd_onesided = psd[:n_rfft]
    elif psd.shape[0] == n_rfft:
        psd_onesided = psd
    else:
        raise ValueError(
            f"psd_v2_per_hz length {psd.shape[0]} must be n_samples={N} "
            f"or n_samples//2+1={n_rfft}."
        )

    # Per-bin amplitude for a REAL (Hermitian-symmetric) spectrum. Deriving
    # this via Parseval (numpy's unnormalized-forward-FFT convention, X the
    # full length-N spectrum implied by mirroring): Var(noise_t) =
    # (1/N^2) * [|X[0]|^2 + |X[N/2]|^2 + 2*sum_{k=1}^{N/2-1}|X[k]|^2].
    # Constructing X[k] = amp*(g1+i*g2) for interior bins (g1,g2 iid
    # N(0,1)) gives E[|X[k]|^2] = 2*amp^2 -- so the Parseval mirroring
    # factor of 2 for interior bins and the "complex draw has 2x the
    # variance of a single real Gaussian" factor of 2 combine to a factor
    # of 4, not 2: naively using amp = sqrt(S_v*df) overshoots by 4x, so
    # amp = sqrt(S_v*df/4) = sqrt(S_v*df)/2 is the per-bin amplitude that
    # makes Var(noise_t) integrate to the target S1*B over whatever
    # one-sided band B the psd array covers (S1*fs/2 for the full Nyquist
    # band) -- confirmed against scipy.signal.periodogram directly on this
    # function's own output (see module docstring's "Absolute-scale
    # history"; an earlier version used sqrt(S_v*df/2), 2x too high in
    # variance, which passed every INTERNAL self-consistency check because
    # generate_baseband_noise had an independent, compensating 2x bug of
    # its own -- only caught by validating each function's absolute scale
    # against an outside reference rather than against each other).
    # The `* N` after irfft below corrects for numpy's own 1/N irfft
    # normalization, the same correction generate_baseband_noise applies
    # after its own (equally 1/N-normalized) ifft call.
    amp = np.sqrt(np.maximum(psd_onesided * df, 0.0) / 4.0)

    noise_f = amp * (rng.standard_normal(n_rfft) + 1j * rng.standard_normal(n_rfft))

    # Enforce Hermitian symmetry so IFFT is real -- DC and (for even N)
    # Nyquist have no mirror partner, so they must be purely real; folding
    # the dropped imaginary part's variance into the real part via *sqrt(2)
    # keeps each bin's total contributed power correct on its own (verified
    # as part of the overall Parseval derivation above, not in isolation).
    noise_f[0] = noise_f[0].real * np.sqrt(2)
    if N % 2 == 0:
        noise_f[-1] = noise_f[-1].real * np.sqrt(2)

    noise_t = np.fft.irfft(noise_f, n=N) * N   # undo irfft's own 1/N normalization
    return noise_t.astype(float)


def generate_phase_noise(
    n_samples: int,
    fs: float,
    psd_rad2_per_hz: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate one real phase-noise realization phi(t) [rad].

    A phase-noise process is mathematically identical in KIND to a real
    voltage-noise process -- both are real, bandlimited, stationary Gaussian
    processes from a one-sided PSD -- just in different physical units (rad
    vs V) and, critically, with a hard bandwidth cutoff already baked into
    `psd_rad2_per_hz` by the caller (see noise/psd.py's
    phase_noise_psd_from_spec(), which requires an explicit bandwidth_hz --
    unlike voltage noise, phase noise has no natural bandwidth of its own).
    This is therefore a thin, explicitly-named wrapper around
    generate_rf_noise() rather than a separate implementation -- reusing the
    exact same (independently scipy.signal.periodogram-validated, see that
    function's own docstring) amplitude scaling.

    Parameters
    ----------
    n_samples : int
    fs : float
        Sample rate (Hz) -- must match whatever grid this realization will
        be applied to (the source waveform's own envelope grid in
        complex_baseband mode, or the resampled envelope grid at the
        schematic's native rate in real_axis mode -- see
        simulation/engine.py's phase-noise injection).
    psd_rad2_per_hz : np.ndarray, shape (n_samples//2+1,) or (n_samples,)
        One-sided phase-noise PSD [rad^2/Hz], already bandwidth-cut (see
        above).
    rng : np.random.Generator, optional

    Returns
    -------
    phi : np.ndarray, float64, shape (n_samples,)
        Real-valued phase realization (radians).
    """
    return generate_rf_noise(n_samples, fs, psd_rad2_per_hz, rng=rng)


def generate_noise_ensemble(
    n_samples: int,
    fs: float,
    psd_v2_per_hz: np.ndarray,
    n_realizations: int,
    mode: str = "complex_baseband",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an ensemble of noise realizations.

    Parameters
    ----------
    n_samples, fs, psd_v2_per_hz : as per generate_baseband_noise / generate_rf_noise.
    n_realizations : int
        Number of independent realizations.
    mode : str
        'complex_baseband' or 'real_axis'.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    ensemble : np.ndarray, shape (n_realizations, n_samples)
        dtype complex128 for baseband, float64 for real_axis.
    """
    rng = np.random.default_rng(seed)
    gen = generate_baseband_noise if mode == "complex_baseband" else generate_rf_noise
    dtype = complex if mode == "complex_baseband" else float
    ensemble = np.empty((n_realizations, n_samples), dtype=dtype)
    for i in range(n_realizations):
        ensemble[i] = gen(n_samples, fs, psd_v2_per_hz, rng=rng)
    return ensemble
