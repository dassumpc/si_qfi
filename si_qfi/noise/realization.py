"""
si_qfi.noise.realization
========================
Generate bandlimited stochastic noise realization arrays from a PSD.

Two variants:
  - Real bandpass noise  (real-axis mode): real array at RF sample rate
  - Complex baseband noise (baseband mode): complex array at baseband sample rate

The generation method is spectral shaping of white Gaussian noise:
    1. Draw white Gaussian noise in the frequency domain
    2. Multiply by sqrt(S_v(f) · df) to shape the spectrum
    3. IFFT to get the time-domain realization
    4. Take real part (real-axis) or keep complex (baseband)
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

    Parameters
    ----------
    n_samples : int
        Number of time-domain samples to generate.
    fs : float
        Sample rate (Hz). Determines the frequency resolution df = fs/n_samples.
    psd_v2_per_hz : np.ndarray, shape (n_samples,) or (n_samples//2+1,)
        One-sided or two-sided noise PSD [V²/Hz] evaluated at the FFT frequencies.
        If one-sided (len = n_samples//2+1), it is mirrored internally.
        If two-sided (len = n_samples), it is used as-is.
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().

    Returns
    -------
    noise : np.ndarray, complex128, shape (n_samples,)
        Complex baseband noise realization with spectral density matching psd_v2_per_hz.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = int(n_samples)
    df = fs / N

    psd = np.asarray(psd_v2_per_hz, dtype=float)

    # If one-sided, build full two-sided PSD for rfft-style usage
    if psd.shape[0] == N // 2 + 1:
        # Two-sided: mirror (for complex signal, both sides carry noise)
        psd_two_sided = np.concatenate([psd, psd[-2:0:-1]])[:N]
    elif psd.shape[0] == N:
        psd_two_sided = psd
    else:
        raise ValueError(
            f"psd_v2_per_hz length {psd.shape[0]} must be n_samples={N} "
            f"or n_samples//2+1={N//2+1}."
        )

    # Spectral amplitude: sqrt(S_v · df)
    amp = np.sqrt(np.maximum(psd_two_sided * df, 0.0))

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
        One-sided noise PSD [V²/Hz].
    rng : np.random.Generator, optional

    Returns
    -------
    noise : np.ndarray, float64, shape (n_samples,)
        Real-valued bandlimited noise realization.
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

    # One-sided amplitude; factor √2 for one-sided → two-sided conversion
    amp = np.sqrt(np.maximum(psd_onesided * df, 0.0))

    noise_f = amp * (rng.standard_normal(n_rfft) + 1j * rng.standard_normal(n_rfft))

    # Enforce Hermitian symmetry so IFFT is real
    noise_f[0] = noise_f[0].real * np.sqrt(2)
    if N % 2 == 0:
        noise_f[-1] = noise_f[-1].real * np.sqrt(2)

    noise_t = np.fft.irfft(noise_f, n=N)
    return noise_t.astype(float)


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
