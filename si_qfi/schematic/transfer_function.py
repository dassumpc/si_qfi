"""
si_qfi.schematic.transfer_function
====================================
Extract voltage transfer functions H(f) between adjacent probe pairs from a
SignalIntegrity schematic. Extraction is waveform-agnostic — it depends only
on the schematic, never on any particular drive waveform's sample rate,
carrier frequency, or mode. Converting a raw TransferFunction to a
time-domain impulse response h(τ) (which does need those things) is a
separate, later step: see compute_impulse_response().

SI API notes (verified against the installed SignalIntegrity source, not
guessed)
---------------------------------------------------------------------------
There is no `si_app.SParameters()` method. Two real methods exist, and which
applies depends on schematic topology:
  - `si_app.CalculateSParameters()` — only works for schematics built from
    `Port`-type devices with no `Output` devices (network-analyzer style).
  - `si_app.TransferParameters()` — works for VoltageSource + Output
    schematics, which is SI-QFI's convention. This is the one used here.
    Returns a `Result` (dict subclass) with keys 'source names',
    'output waveform labels', 'transfer matrices'. `Result.FrequencyResponse
    (from_name, to_name)` looks up a `FrequencyResponse` by name directly —
    no manual port-index bookkeeping needed.

TransferParameters() only exposes *source-referenced* responses (source →
each probe), never probe-to-probe directly. To get H(f) for a segment
between two probes A → B where neither is the source, this module computes:

    H_{A→B}(f) = H_{source→B}(f) / H_{source→A}(f)     [see _extract_single_tf]

which is exact by linearity (both are responses of the same linear network
to the same source). TransferParameters() re-simulates the whole network, so
its result is cached per si_app (see _get_transfer_parameters) rather than
re-fetched for every segment.

Reuse of SI's own FrequencyResponse machinery
---------------------------------------------------------------------------
`Result.FrequencyResponse()` returns a real SI `FrequencyResponse` object,
which already implements `/` (frequency-domain division, used for the ratio
above) and `.ImpulseResponse(td)` (real-axis mode's impulse response — see
compute_impulse_response()). TransferFunction stores this object as
`si_frequency_response` (an untyped/`Any` field — no SI import needed to
declare it) and derives `freqs`/`H` from it as properties rather than
storing them separately — TransferFunction is always constructed from a
schematic (see _extract_single_tf), so there is never a second, independent
source for this data to reconcile. TransferFunction deliberately does *not*
inherit from FrequencyResponse, though: that would require importing SI at
module level here, and this module is imported unconditionally by
simulation/engine.py → si_qfi/__init__.py, so it would make SignalIntegrity
a hard dependency just to `import si_qfi` — breaking the "core math works
without SI installed" property every other SI touchpoint in this codebase
preserves by importing SI lazily inside function bodies instead.

Extraction vs. impulse response (PRD §3.3)
---------------------------------------------------------------------------
extract_all_transfer_functions() / extract_noise_transfer_functions() return
raw, frequency-domain-only TransferFunction objects (h/dt left as None).
Converting to a time-domain impulse response — which requires a target
sample rate, a mode, and (for baseband) a carrier frequency — is the
caller's job, via compute_impulse_response(), done once a concrete
SourceWaveform is available (the engine does this). This keeps extraction
coupled only to the schematic, never to any specific drive waveform.

The two modes disagree on *whose* sample rate governs the impulse response:
  - Real-axis mode: the schematic has one natural, waveform-independent
    sample rate (native_sample_rate() below, derived from the schematic's
    own frequency sweep — matches SI's own CalculationProperties.
    UserSampleRate). h(τ) is computed directly at that rate (no
    interpolation needed), and the *drive waveform* is resampled to match
    it (SourceWaveform.rf_waveform_at()) — mirroring SI's own
    Waveform.Adapt() idiom for "fit a waveform to a target time grid".
  - Complex baseband mode: there is no schematic-intrinsic baseband rate —
    it's fundamentally a pulse-bandwidth choice (PRD §4.1's whole point is
    running two orders of magnitude below real-axis rate). So baseband mode
    keeps interpolating H(f) onto the target grid implied by the envelope's
    own fs/carrier, as before.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TransferFunction:
    """
    Voltage transfer function between two schematic nodes.

    Always constructed from a schematic (see _extract_single_tf) — there is
    no supported way to build one without a real SI FrequencyResponse, so
    freqs/H are derived from si_frequency_response rather than stored
    independently (see module docstring).

    Attributes
    ----------
    label_in : str   Source probe label (input node).
    label_out : str  Destination probe label (output node).
    si_frequency_response : Any
        The native SI FrequencyResponse this was extracted from (set by
        _extract_single_tf). Untyped so this module never needs to import
        SI's class to declare the field. freqs/H (below) are computed from
        it on access; compute_impulse_response() also uses its
        .ImpulseResponse() directly for real-axis mode. repr suppressed —
        it would otherwise dump the entire complex-value list.
    h : np.ndarray or None
        Time-domain impulse response. None until compute_impulse_response()
        has been called on this object (extraction alone never populates
        this — see module docstring).
    dt : float or None
        Sample interval of h (seconds). None until compute_impulse_response().
    """
    label_in: str
    label_out: str
    si_frequency_response: Any = field(repr=False)
    h: Optional[np.ndarray] = None     # time domain — set by compute_impulse_response()
    dt: Optional[float] = None

    @property
    def freqs(self) -> np.ndarray:
        """Frequency array (Hz), float64, shape (F,)."""
        return np.array(self.si_frequency_response.Frequencies(), dtype=float)

    @property
    def H(self) -> np.ndarray:
        """Complex transfer function, complex128, shape (F,)."""
        return np.array(self.si_frequency_response.Response(), dtype=complex)


def extract_all_transfer_functions(
    schematic,           # SISchematic
    nl_labels: list[str],
) -> dict[tuple[str, str], TransferFunction]:
    """
    Extract raw (frequency-domain-only) transfer functions for all required
    segment pairs.

    Node pairs needed:
      Segment pairs: (source_label, NL_1), (NL_1, NL_2), ...,
                      (NL_n, qubit_probe_label)

    The engine calls this once during setup. Results are cached and reused
    for all realizations. Returned TransferFunctions have h=None, dt=None —
    call compute_impulse_response() to populate those once a target sample
    rate/mode/carrier are known (see module docstring).

    Parameters
    ----------
    schematic : SISchematic
        Loaded schematic object (contains si_app, source_label,
        qubit_probe_label).
    nl_labels : list of str
        Nonlinear node probe labels, in propagation order (source_label →
        nl_labels → qubit_probe_label). This is the `nonlinear` annotation
        dict's key order, as validated by
        schematic.loader.validate_node_labels() — it is the single source of
        truth for segmentation, not anything derived from the schematic
        itself.

    Returns
    -------
    dict mapping (label_in, label_out) → TransferFunction (raw).
    """
    source_label = schematic.source_label
    qubit_label = schematic.qubit_probe_label

    all_labels = [source_label] + list(nl_labels) + [qubit_label]
    segment_pairs = list(zip(all_labels[:-1], all_labels[1:]))

    result: dict[tuple[str, str], TransferFunction] = {}

    for (lin, lout) in segment_pairs:
        tf = _extract_single_tf(schematic.si_app, lin, lout, source_label)
        result[(lin, lout)] = tf

    return result


def extract_noise_transfer_functions(
    schematic,
    noise_node_labels: list[str],
) -> dict[str, TransferFunction]:
    """
    Extract raw (frequency-domain-only) transfer functions h_{j→qubit} for
    each noise node j.

    Parameters
    ----------
    schematic : SISchematic
    noise_node_labels : list of str
        Labels of noise-annotated nodes (any schematic node, not just NL probes).

    Returns
    -------
    dict mapping noise_node_label → TransferFunction (raw; h=None, dt=None
    until compute_impulse_response() is called — see module docstring).
    """
    source_label = schematic.source_label
    qubit_label = schematic.qubit_probe_label

    result: dict[str, TransferFunction] = {}
    for label in noise_node_labels:
        result[label] = _extract_single_tf(schematic.si_app, label, qubit_label, source_label)
    return result


def compute_impulse_response(
    tf: TransferFunction,
    mode: str,
    *,
    fs: Optional[float] = None,
    carrier_hz: Optional[float] = None,
) -> TransferFunction:
    """
    Convert a raw (frequency-domain-only) TransferFunction into one with a
    time-domain impulse response.

    This is the deferred step described in the module docstring: extraction
    never needs fs/mode/carrier_hz, only this conversion does — call it once
    a concrete SourceWaveform's fs/carrier are known (the engine does this,
    per-run, not per-schematic).

    Real-axis mode ignores fs entirely: it reuses SI's own
    FrequencyResponse.ImpulseResponse() (via tf.si_frequency_response) with
    no target rate, so SI derives its own native sample rate directly from
    the FrequencyResponse's own frequency grid — not a hand-rolled IRFFT,
    and not the caller-supplied fs, which (per native_sample_rate()) would
    only ever equal that same native rate anyway. carrier_hz is also unused
    in this mode (real-axis has no baseband shift to apply). fs/carrier_hz
    are keyword-only and optional for this reason.

    Complex baseband mode still needs both explicitly — fs to set the
    baseband grid the envelope's own bandwidth implies, carrier_hz for the
    H(f) → H(f+carrier) shift (PRD §3.3) — and raises ValueError if either
    is missing.

    Returns a new TransferFunction with h/dt populated; freqs/H are
    unchanged (still derived from the same si_frequency_response — see
    _extract_single_tf).
    """
    if mode == "real_axis":
        ir = tf.si_frequency_response.ImpulseResponse()
        h = np.asarray(ir.Values(), dtype=float)
        dt = 1.0 / float(ir.td.Fs)
    elif mode == "complex_baseband":
        if fs is None or carrier_hz is None:
            raise ValueError(
                "compute_impulse_response(mode='complex_baseband') requires "
                "both fs and carrier_hz."
            )
        h, _unused_freq_grid, dt = _tf_to_baseband_impulse(tf.freqs, tf.H, fs, carrier_hz)
    else:
        raise ValueError(f"mode must be 'complex_baseband' or 'real_axis', got {mode!r}.")

    return TransferFunction(
        label_in=tf.label_in, label_out=tf.label_out,
        si_frequency_response=tf.si_frequency_response, h=h, dt=dt,
    )


def native_sample_rate(tf: TransferFunction) -> float:
    """
    The real-axis-mode sample rate implied by the schematic's own frequency
    sweep: 2x the top frequency in tf.freqs (Nyquist for the associated
    real time-domain signal — matches SI's own CalculationProperties.
    UserSampleRate for an evenly-spaced sweep from 0 to EndFrequency).

    Used in real-axis mode so the schematic's own resolution determines the
    impulse response and convolution grid; the drive waveform is resampled
    to match it (SourceWaveform.rf_waveform_at()) rather than the transfer
    function being interpolated onto the waveform's own fs — see module
    docstring.
    """
    return 2.0 * float(tf.freqs[-1])


def _get_transfer_parameters(si_app: Any):
    """
    Call si_app.TransferParameters() once and cache the Result on si_app
    (TransferParameters() re-simulates the whole network, and this is called
    once per segment/noise-node — caching avoids re-simulating for each).

    Raises
    ------
    RuntimeError
        If TransferParameters() fails (e.g. equations invalid, or the
        schematic doesn't have the VoltageSource + Output topology it
        requires — see module docstring).
    """
    cached = getattr(si_app, "_si_qfi_transfer_result", None)
    if cached is not None:
        return cached
    result = si_app.TransferParameters()
    if not result or result.get("transfer matrices") is None:
        raise RuntimeError(
            "SignalIntegrity TransferParameters() failed. Check that the "
            "schematic has a VoltageSource and Output-type probes, no Port "
            "devices, and no unresolved equation errors."
        )
    si_app._si_qfi_transfer_result = result
    return result


def _source_referenced_response(si_app: Any, source_label: str, label: str):
    """
    Return the FrequencyResponse from source_label to `label`, as computed
    by TransferParameters(). Raises a clear ValueError if either name isn't
    found (Result.FrequencyResponse would otherwise raise a bare ValueError
    from list.index()).
    """
    result = _get_transfer_parameters(si_app)
    if source_label not in result["source names"]:
        raise ValueError(
            f"Source '{source_label}' not found. Available sources: "
            f"{result['source names']}."
        )
    if label not in result["output waveform labels"]:
        raise ValueError(
            f"Probe '{label}' not found. Available probes: "
            f"{result['output waveform labels']}."
        )
    return result.FrequencyResponse(source_label, label)


def _extract_single_tf(
    si_app: Any,
    label_in: str,
    label_out: str,
    source_label: str,
) -> TransferFunction:
    """
    Extract a single raw (frequency-domain-only) transfer function
    H(f) = V_out(label_out)/V_in(label_in). h/dt are left as None — see
    compute_impulse_response() and the module docstring.

    TransferParameters() only exposes source-referenced responses, so this
    is computed as (see module docstring):
        label_in == source_label:  fr = FrequencyResponse(source_label → label_out)
        otherwise:                 fr = FrequencyResponse(source_label → label_out)
                                       / FrequencyResponse(source_label → label_in)
    using SI's own FrequencyResponse division (FrequencyDomain.__truediv__),
    which returns another proper FrequencyResponse — stored on the result as
    si_frequency_response for compute_impulse_response()'s real-axis path to
    reuse (and for the freqs/H properties to derive from).
    """
    fr_out = _source_referenced_response(si_app, source_label, label_out)

    if label_in == source_label:
        fr = fr_out
    else:
        fr_in = _source_referenced_response(si_app, source_label, label_in)
        fr = fr_out / fr_in

    return TransferFunction(label_in=label_in, label_out=label_out, si_frequency_response=fr)


def _tf_to_baseband_impulse(
    freqs: np.ndarray,
    H: np.ndarray,
    fs: float,
    carrier_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Shift H(f) to baseband H̃(f) = H(f + f_carrier) and IFFT.

    The baseband-equivalent transfer function is evaluated at:
        f_baseband = freq_grid - carrier_hz
    where freq_grid spans [-fs/2, fs/2].
    """
    # Build a uniformly-spaced frequency grid at the target sample rate
    N = int(round(fs / (freqs[1] - freqs[0]))) if len(freqs) > 1 else 1024
    N = max(N, 64)
    f_bb = np.fft.fftfreq(N, d=1.0 / fs)   # two-sided baseband freqs, FFT-bin order

    # Interpolate H(f) onto the shifted grid: H̃(f_bb) = H(f_bb + carrier).
    # f_bb (and therefore H_tilde) is already in FFT-native bin order (index 0
    # = baseband DC, matching np.fft.fftfreq's own convention) -- do NOT
    # ifftshift it before the ifft below, that would assume a centered/
    # ascending frequency order and rotate every bin by N/2.
    f_rf = f_bb + carrier_hz   # RF frequencies corresponding to baseband grid
    H_tilde = _interpolate_tf(freqs, H, f_rf)

    # IFFT -> complex baseband impulse response (discrete convolution kernel
    # convention, same as real-axis mode: sum(h) == H(f=0-equivalent), no
    # extra dt/fs scaling -- engine.py's fftconvolve(v, h) expects this).
    #
    # fftshift the result so index 0 is the most-negative representable
    # delay and the array is centered around tau=0 -- this matches SI's own
    # FrequencyResponse.ImpulseResponse() convention exactly (see its source:
    # it computes the raw ifft, then does `Y = Y[K//2:] + Y[:K//2]`, i.e. an
    # fftshift). Without this, real-axis mode's h (SI-native, centered) and
    # baseband mode's h (this function, previously left in native FFT-bin
    # order) would use two different conventions for "where is tau=0",
    # even though engine.py now convolves both the same way (full, untouched
    # -- no truncation, no offset bookkeeping -- see engine.py's
    # _nonlinear_pass). fftshift is a pure reordering (sum(h) is unaffected).
    h_tilde = np.fft.fftshift(np.fft.ifft(H_tilde))
    return h_tilde.astype(complex), f_bb, 1.0 / fs


def _interpolate_tf(
    f_known: np.ndarray,
    H_known: np.ndarray,
    f_query: np.ndarray,
) -> np.ndarray:
    """
    Interpolate complex transfer function H at query frequencies.
    Extrapolates as 0 outside the known range (transfer function is zero
    outside the measured bandwidth — conservative assumption).
    """
    from scipy.interpolate import interp1d
    # Interpolate magnitude and phase separately to avoid wrapping artefacts
    mag = np.abs(H_known)
    phase = np.unwrap(np.angle(H_known))

    interp_mag = interp1d(
        f_known, mag, kind="linear", bounds_error=False, fill_value=0.0
    )
    interp_phase = interp1d(
        f_known, phase, kind="linear", bounds_error=False, fill_value=0.0
    )

    abs_query = np.abs(f_query)   # for one-sided interpolation on two-sided grid
    H_out = interp_mag(abs_query) * np.exp(1j * np.sign(f_query) * interp_phase(abs_query))
    return H_out


def compute_isolation_db(
    tf_forward: TransferFunction,
    tf_reverse: TransferFunction,
    signal_band_hz: tuple[float, float],
) -> float:
    """
    Compute isolation (dB) as max reverse transfer in the signal band.

    Parameters
    ----------
    tf_forward : TransferFunction  A → B
    tf_reverse : TransferFunction  B → A
    signal_band_hz : (f_low, f_high)  Signal band for isolation check.

    Returns
    -------
    isolation_db : float  Max |H_reverse| in band, in dB. More negative = better.
    """
    f_low, f_high = signal_band_hz
    mask = (tf_reverse.freqs >= f_low) & (tf_reverse.freqs <= f_high)
    if not np.any(mask):
        return -np.inf
    H_rev_band = tf_reverse.H[mask]
    H_fwd_band = tf_forward.H[mask]
    # Normalise reverse by forward magnitude to get isolation relative to signal
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(H_rev_band) / np.maximum(np.abs(H_fwd_band), 1e-30)
    return float(20.0 * np.log10(np.max(ratio)))
