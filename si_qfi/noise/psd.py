"""
si_qfi.noise.psd
================
Compute one-sided noise power spectral density S_v(f) [V²/Hz] for each
entry in a `noise` annotation dict.

Each noise annotation key names a statistical-noise-source device declared
in the SI schematic (see schematic/loader.py's noise_source_names /
schematic/noise.py). Two ways to get that source's S_v(f):

  - Default (empty dict `{}`, or any dict without an override key): SI's own
    computed spectral density for that device (Johnson/shot/white/etc, per
    its own schematic-configured Type/Resistance/Temperature/... properties)
    — see schematic.noise.get_noise_source_psd().
  - Override (`single_sided_psd_v2_per_hz`, `single_sided_psd_dbm_hz`, or
    `type` key present): skip SI's physical model for this run, inject a PSD
    at the given magnitude instead — still propagated via that source's own
    SI-derived location/transfer function to the qubit plane, only the
    magnitude is overridden. `single_sided_psd_v2_per_hz` accepts either a
    plain number (flat PSD, the common case) or a callable freqs->S_v(freqs)
    for a COLORED (frequency-shaped) PSD -- e.g. a quasi-static (narrowband-
    at-DC) source, or a narrow probe centered away from DC, both used in
    examples/noise_filter_function_demo.py to trace out the qubit's own
    frequency-dependent noise susceptibility empirically. `single_sided_
    psd_dbm_hz` stays flat-only (a colored dBm/Hz callable would need a
    frequency-dependent impedance reference to convert sensibly, not just
    the fixed 50 ohm assumed here -- not needed by any current caller).
    `type` ('noise_figure' / 'thermal' / 'noise_density') is a higher-level,
    physically-parameterized alternative to specifying a raw PSD number
    directly -- see psd_from_override()'s own docstring for the exact specs.
    All Johnson-noise-derived PSDs here (both `type='thermal'` and
    `type='noise_figure'`) use S_v = 4*k_B*T*R, matching the SAME 4kTR
    convention SI's own native computation uses (cross-checked directly in
    tests/test_engine_noise.py::test_johnson_psd_matches_4kTR_formula) --
    NOT the un-factored `k_B*T*R` an earlier draft of this project's PRD
    (SI_Quantum_Fidelity_Plugin_PRD.md §7.1) mistakenly wrote; that formula
    was never implemented and would have silently undersized every
    noise-figure-derived PSD by exactly 4x relative to every other noise
    source in this codebase.
"""

from __future__ import annotations

import numpy as np
from typing import Any

_KB = 1.380649e-23
_NOISE_FIGURE_REFERENCE_K = 290.0   # standard IEEE noise-figure reference temperature (T0)
_DEFAULT_RESISTANCE_OHMS = 50.0     # matches single_sided_psd_dbm_hz's existing fixed-50-ohm convention


def psd_from_override(spec: dict[str, Any], freqs: np.ndarray) -> np.ndarray:
    """
    One-sided PSD S_v(f) [V²/Hz] from an explicit override value.

    Accepts either a raw-PSD key:
      'single_sided_psd_v2_per_hz'  — a number (flat PSD, already in V²/Hz)
        or a callable freqs->S_v(freqs) for a colored/frequency-shaped PSD
        (must return an array the same shape as `freqs`, non-negative).
      'single_sided_psd_dbm_hz'     — a number, in dBm/Hz, converted to
        V²/Hz into 50 Ω (flat only -- see module docstring for why).

    ...or a higher-level, physically-parameterized `'type'` key (all flat --
    none of these support a callable/colored shape, matching
    single_sided_psd_dbm_hz's own flat-only precedent):
      'type': 'thermal'
        Plain Johnson-Nyquist noise at a specified physical temperature --
        an explicit way to force a Johnson source's PSD regardless of how
        (or whether) the underlying schematic device's own Type/Temperature
        properties are configured. Requires 'temperature_k'. Optional
        'resistance_ohms' (default 50.0). S_v = 4*k_B*T*R.
      'type': 'noise_figure'
        The excess noise a component with a given noise figure contributes,
        referred to its input -- the standard textbook noise-figure-to-PSD
        conversion. Requires 'noise_figure_db'. Optional 'temperature_k'
        (default 290.0 -- the standard IEEE noise-figure reference
        temperature T0, NOT the component's own physical operating
        temperature; override it only if you have a specific reason the
        figure was specified against a non-standard reference) and
        'resistance_ohms' (default 50.0).
        T_excess = temperature_k * (10**(noise_figure_db/10) - 1)
        S_v = 4*k_B*T_excess*R
      'type': 'noise_density'
        An explicit-type alias for 'single_sided_psd_v2_per_hz' -- must
        also supply that key in the same dict. Any accompanying
        'temperature_k' is documentary only (the PSD is given directly,
        not derived from it).

    Parameters
    ----------
    spec : dict
        Must contain 'single_sided_psd_v2_per_hz', 'single_sided_psd_dbm_hz',
        or 'type' (with that type's own required keys, see above).
    freqs : np.ndarray, shape (F,)
        Frequency grid (Hz) the returned array is evaluated on.

    Returns
    -------
    S_v : np.ndarray, float64, shape (F,)
    """
    if "type" in spec:
        kind = spec["type"]
        if kind == "thermal":
            if "temperature_k" not in spec:
                raise ValueError("Noise override type='thermal' requires 'temperature_k'.")
            t = float(spec["temperature_k"])
            r = float(spec.get("resistance_ohms", _DEFAULT_RESISTANCE_OHMS))
            S_v = 4.0 * _KB * t * r
            return np.full_like(freqs, S_v, dtype=float)
        elif kind == "noise_figure":
            if "noise_figure_db" not in spec:
                raise ValueError("Noise override type='noise_figure' requires 'noise_figure_db'.")
            nf_db = float(spec["noise_figure_db"])
            t_ref = float(spec.get("temperature_k", _NOISE_FIGURE_REFERENCE_K))
            r = float(spec.get("resistance_ohms", _DEFAULT_RESISTANCE_OHMS))
            t_excess = t_ref * (10.0 ** (nf_db / 10.0) - 1.0)
            S_v = 4.0 * _KB * t_excess * r
            return np.full_like(freqs, S_v, dtype=float)
        elif kind == "noise_density":
            if "single_sided_psd_v2_per_hz" not in spec:
                raise ValueError(
                    "Noise override type='noise_density' requires 'single_sided_psd_v2_per_hz'."
                )
            # Falls through to the plain-key handling below.
        else:
            raise ValueError(
                f"Unknown noise override type {kind!r}. Use 'noise_figure', 'thermal', or 'noise_density'."
            )

    if "single_sided_psd_v2_per_hz" in spec:
        val = spec["single_sided_psd_v2_per_hz"]
        if callable(val):
            S_v = np.asarray(val(freqs), dtype=float)
            if S_v.shape != freqs.shape:
                raise ValueError(
                    f"single_sided_psd_v2_per_hz callable must return an array "
                    f"of shape {freqs.shape} (got {S_v.shape})."
                )
            if np.any(S_v < 0):
                raise ValueError("single_sided_psd_v2_per_hz callable returned a negative PSD value.")
            return S_v
        return np.full_like(freqs, float(val), dtype=float)
    elif "single_sided_psd_dbm_hz" in spec:
        psd_dbm_hz = float(spec["single_sided_psd_dbm_hz"])
        # Convert dBm/Hz → W/Hz → V²/Hz  (into 50 Ω: P = V²/R, V² = P·R)
        S_w = 10.0 ** (psd_dbm_hz / 10.0) * 1e-3
        S_v = S_w * 50.0
        return np.full_like(freqs, S_v, dtype=float)
    else:
        raise ValueError(
            "Noise override must contain 'single_sided_psd_v2_per_hz', "
            "'single_sided_psd_dbm_hz', or 'type' (see psd_from_override()'s docstring)."
        )


def psd_cache_for_noise_nodes(
    schematic,
    noise: dict[str, dict[str, Any]],
    n_samples: int,
    fs: float,
    h_lengths: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Build the {noise_source_name: S_v(f) [V²/Hz]} cache for an entire
    `noise` annotation dict — one call per engine.run(), not per realization
    (the PSD itself is deterministic; only the realization draw is
    stochastic, see noise/realization.py).

    Each source's PSD is evaluated on ITS OWN frequency grid
    (np.fft.rfftfreq(n_draw, d=1/fs), n_draw = n_samples + h_lengths[label]
    - 1) rather than a single shared grid — this must match
    NoisePropagator.generate_realization()'s own per-source draw length
    exactly (see its docstring for why: each source draws n_samples +
    len(h)-1 samples and uses fftconvolve(..., mode="valid") so every one
    of the n_samples outputs is a full-overlap convolution sum, and
    different noise sources can have differently-lengthed h).

    Parameters
    ----------
    schematic : SISchematic
        Loaded schematic (used for SI-sourced lookups — see
        schematic.noise.get_noise_source_psd()).
    noise : dict
        {noise_source_name: override_dict}. override_dict is `{}` (or any
        dict without an override key) to use SI's own computed density for
        that source, or `{"single_sided_psd_v2_per_hz": ...}` /
        `{"single_sided_psd_dbm_hz": ...}` / `{"type": ...}` to override it —
        see psd_from_override().
    n_samples : int
        Target total length (matches v_nl_qubit's own length this run —
        see engine.py).
    fs : float
        Sample rate (Hz) actually used for convolution this run.
    h_lengths : dict
        {noise_source_name: len(h_{name→qubit})} — the impulse response
        length for each annotated source (from
        schematic.noise.extract_noise_source_transfer_functions() +
        compute_impulse_response(), already computed by the engine before
        this is called).

    Returns
    -------
    dict mapping noise_source_name → np.ndarray, shape (n_draw//2+1,), V²/Hz
    (n_draw specific to that source — see above).
    """
    from ..schematic.noise import get_noise_source_psd

    cache: dict[str, np.ndarray] = {}
    for label, override in noise.items():
        n_draw = n_samples + h_lengths[label] - 1
        freqs = np.fft.rfftfreq(n_draw, d=1.0 / fs)
        if (
            "single_sided_psd_v2_per_hz" in override
            or "single_sided_psd_dbm_hz" in override
            or "type" in override
        ):
            cache[label] = psd_from_override(override, freqs)
        else:
            cache[label] = get_noise_source_psd(schematic, label, freqs)
    return cache


def noise_bandwidth_hz(
    source_waveform,
    mode: str,
    carrier_freq_hz: float,
) -> tuple[float, float]:
    """
    Return the (f_low, f_high) noise bandwidth in Hz appropriate for the mode.

    Complex baseband: bandwidth centred at DC, width = source Fs/2.
    Real axis: full one-sided bandwidth 0 → source Fs/2.
    """
    fs = source_waveform.fs
    if mode == "complex_baseband":
        bw = fs / 2.0
        return (-bw, bw)
    else:
        return (0.0, fs / 2.0)


def phase_noise_psd_from_spec(spec: dict[str, Any], freqs: np.ndarray) -> np.ndarray:
    """
    One-sided phase-noise PSD S_phi(f) [rad^2/Hz] from an `engine.run(...,
    phase_noise={...})` spec dict.

    Unlike VN1-style voltage noise, phase noise (LO/oscillator phase jitter,
    riding on the drive's own carrier) has NO natural bandwidth of its own --
    voltage noise is generated directly on whatever grid the simulation
    already propagates on (its own fs answers "how wide" automatically), but
    a real oscillator's phase-noise spectrum L(f) does not roll off to zero
    at high offset -- it flattens to a nonzero floor that, taken alone,
    extends indefinitely. What actually bounds the TOTAL effect is the
    qubit's own frequency-dependent sensitivity (see quantum/snr.py's
    "Absolute-scale history"-adjacent filter-function derivation and
    examples/noise_filter_function_demo.py's empirical measurement of it),
    NOT anything about the phase-noise source itself -- so this function
    requires an explicit `bandwidth_hz`, with no default, rather than
    silently inheriting the simulation's own sample rate the way voltage
    noise does. See examples/phase_noise_case_study_demo.py for a worked
    example of choosing one by convergence (double it, confirm the result
    doesn't change), rather than by a rule of thumb.

    Spec dict must contain 'bandwidth_hz' plus exactly one of:
      'single_sided_psd_rad2_per_hz' -- a number (flat) or a callable
        freqs->S_phi(freqs) [rad^2/Hz], the raw/native form -- same
        contract as noise/psd.py's single_sided_psd_v2_per_hz.
      'dbc_hz' -- a callable freqs->L(freqs) [dBc/Hz] (callable ONLY -- a
        flat dBc/Hz number is not physically meaningful for phase noise,
        which is essentially never flat: real L(f) curves fall off sharply
        close to the carrier and only flatten far out). Converted via the
        standard IEEE Std 1139 small-angle relation
        S_phi(f) = 2 * 10**(L(f)/10), valid whenever L(f) is comfortably
        below 0 dBc (true for any usable oscillator).

    Parameters
    ----------
    spec : dict
    freqs : np.ndarray, shape (F,)
        Frequency grid (Hz) the returned array is evaluated on.

    Returns
    -------
    S_phi : np.ndarray, float64, shape (F,)
        Zero for |f| > bandwidth_hz (hard cutoff -- the caller is
        responsible for having already reconciled bandwidth_hz against
        whatever this grid can actually represent, e.g. clipping it to the
        simulation's own Nyquist in complex_baseband mode and warning if it
        had to -- see simulation/engine.py).
    """
    if "bandwidth_hz" not in spec:
        raise ValueError(
            "phase_noise spec requires 'bandwidth_hz' (the max offset frequency, "
            "Hz, to generate phase noise out to) -- there is no physically honest "
            "default, since real phase noise has no natural bandwidth ceiling of "
            "its own (see phase_noise_psd_from_spec()'s docstring)."
        )
    bandwidth_hz = float(spec["bandwidth_hz"])
    if bandwidth_hz <= 0:
        raise ValueError(f"phase_noise 'bandwidth_hz' must be positive, got {bandwidth_hz}.")

    has_raw = "single_sided_psd_rad2_per_hz" in spec
    has_dbc = "dbc_hz" in spec
    if has_raw and has_dbc:
        raise ValueError(
            "phase_noise spec must contain exactly one of "
            "'single_sided_psd_rad2_per_hz' or 'dbc_hz', not both."
        )

    if has_raw:
        val = spec["single_sided_psd_rad2_per_hz"]
        if callable(val):
            S_phi = np.asarray(val(freqs), dtype=float)
            if S_phi.shape != freqs.shape:
                raise ValueError(
                    f"single_sided_psd_rad2_per_hz callable must return an array "
                    f"of shape {freqs.shape} (got {S_phi.shape})."
                )
            if np.any(S_phi < 0):
                raise ValueError("single_sided_psd_rad2_per_hz callable returned a negative PSD value.")
        else:
            S_phi = np.full_like(freqs, float(val), dtype=float)
    elif has_dbc:
        val = spec["dbc_hz"]
        if not callable(val):
            raise ValueError(
                "phase_noise 'dbc_hz' must be a callable freqs->L(freqs) [dBc/Hz] -- "
                "a flat number isn't physically meaningful for phase noise (see "
                "phase_noise_psd_from_spec()'s docstring)."
            )
        L = np.asarray(val(freqs), dtype=float)
        if L.shape != freqs.shape:
            raise ValueError(
                f"dbc_hz callable must return an array of shape {freqs.shape} (got {L.shape})."
            )
        S_phi = 2.0 * 10.0 ** (L / 10.0)
    else:
        raise ValueError(
            "phase_noise spec must contain 'single_sided_psd_rad2_per_hz' or 'dbc_hz' "
            "(plus 'bandwidth_hz')."
        )

    return np.where(np.abs(freqs) <= bandwidth_hz, S_phi, 0.0)
