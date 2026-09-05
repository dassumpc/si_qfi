"""
si_qfi.simulation.engine
========================
Main SI-QFI simulation engine implementing the two-pass execution flow (PRD §8).

PASS 1 — NONLINEAR PASS (deterministic, runs once):
    Propagate the source waveform through the segmented NL chain.
    Output: v_nl_qubit — the distorted-but-noiseless waveform at the qubit plane.

PASS 2 — NOISE PASS (stochastic, runs N times):
    For each noise node j, independently draw a noise realization and propagate
    it via h_{j→qubit} to the qubit plane. Sum all contributions.

Final qubit waveform per realization:
    v_qubit_i = v_nl_qubit + v_noise_qubit_i

Then call the QuTiP fidelity module per realization.
"""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional
from scipy.signal import fftconvolve

from ..schematic.loader import SISchematic, validate_node_labels
from ..schematic.transfer_function import (
    extract_all_transfer_functions,
    compute_impulse_response,
    native_sample_rate,
    compute_isolation_db,
)
from ..schematic.noise import extract_noise_source_transfer_functions
from ..source.waveform import SourceWaveform
from ..nonlinear.registry import build_nonlinear_nodes
from ..noise.propagation import NoisePropagator
from ..noise.psd import psd_cache_for_noise_nodes


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Output of siq.run(). Contains the ensemble of qubit-plane waveforms
    and all diagnostic warnings. Pass to siq.quantum.gate_fidelity().
    """
    v_nl_qubit: np.ndarray          # Deterministic NL-distorted waveform, shape (N,) -- the
                                     # phi=0 (no phase noise), no-additive-noise baseline, always
                                     # from exactly one _nonlinear_pass() call regardless of
                                     # phase_noise_enabled (see run()'s module docstring)
    v_qubit_ensemble: list[np.ndarray]  # per realization -- only meaningful when noise_enabled (see below)
    fs: float                       # Sample rate (Hz) shared by every array in this result
    mode: str                       # 'complex_baseband' or 'real_axis'
    carrier_freq_hz: float
    noise_enabled: bool = False     # True iff v_qubit_ensemble holds real stochastic
                                     # realizations (additive noise and/or phase noise) rather
                                     # than being a v_nl_qubit stand-in -- lets gate_fidelity()
                                     # know whether to build a real ensemble result
    phase_noise_enabled: bool = False   # True iff a non-empty `phase_noise` dict was passed to
                                         # run() -- lets a caller tell WHY noise_enabled is True
                                         # (additive noise, phase noise, or both) without
                                         # re-deriving it; gate_fidelity() itself doesn't need
                                         # this distinction, only noise_enabled
    warnings: list[str] = field(default_factory=list)
    n_realizations: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    # Open-ended bag for anything that doesn't warrant its own named field --
    # e.g. extra["intermediate_waveforms"]: dict[str, np.ndarray], the
    # waveform at each node in the NL chain (source_label, each NL label,
    # qubit_label), not just the final qubit-plane output. Keeps the return
    # signature stable as the solver grows more diagnostics/metadata.
    #
    # No absolute time axis is tracked here (deliberately -- see
    # _nonlinear_pass's docstring): every array's own index 0 isn't pinned
    # to a specific time, only fs is assumed meaningful/shared across the
    # whole result. A caller needing a time axis for a specific array can
    # build one locally via np.arange(len(array)) / result.fs.

    def __repr__(self) -> str:
        return (
            f"SimulationResult(mode='{self.mode}', "
            f"N={len(self.v_nl_qubit)} samples, "
            f"n_realizations={self.n_realizations}, "
            f"warnings={len(self.warnings)})"
        )


# ---------------------------------------------------------------------------
# Diagnostic constants
# ---------------------------------------------------------------------------

_ISOLATION_THRESHOLD_DB = -20.0       # Warn if reverse coupling exceeds this
_HARMONIC_SUPPRESSION_DB = 30.0       # Warn if 3f_c attenuation is less than this
_NARROWBAND_RATIO_THRESHOLD = 0.05    # Warn if BW/carrier > 5%


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    schematic: SISchematic,
    source: SourceWaveform,
    nonlinear: Optional[dict[str, Any]] = None,
    noise: Optional[dict[str, Any]] = None,
    phase_noise: Optional[dict[str, Any]] = None,
    n_realizations: int = 100,
    mode: str = "complex_baseband",
    isolation_threshold_db: float = _ISOLATION_THRESHOLD_DB,
    harmonic_suppression_db: float = _HARMONIC_SUPPRESSION_DB,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Run the full SI-QFI two-pass simulation.

    Parameters
    ----------
    schematic : SISchematic
        Loaded schematic from siq.load_schematic().
    source : SourceWaveform
        Source drive waveform with carrier frequency.
    nonlinear : dict, optional
        Nonlinear node annotation dict {probe_label: spec_dict}.
        If None, no nonlinearity is applied.
    noise : dict, optional
        Noise annotation dict {noise_source_name: override_dict}. Keys name
        statistical-noise-source devices declared in the SI schematic (see
        schematic.noise_source_names) -- a different namespace than probe
        labels. Presence in the dict enables that source; override_dict is
        `{}` to use SI's own computed spectral density for that device
        (Johnson/shot/white/etc, per its own schematic-configured
        properties), or `{"single_sided_psd_v2_per_hz": ...}` /
        `{"single_sided_psd_dbm_hz": ...}` to inject a flat PSD instead
        (still propagated via that device's own schematic location/transfer
        function -- see noise/psd.py). If None/empty, all realizations are
        identical to v_nl_qubit (noiseless).
    phase_noise : dict, optional
        LO/oscillator phase-noise spec, e.g.
        {"single_sided_psd_rad2_per_hz": callable_or_number, "bandwidth_hz": ...}
        or {"dbc_hz": callable, "bandwidth_hz": ...} -- see
        noise/psd.py's phase_noise_psd_from_spec() for the full contract
        (in particular why 'bandwidth_hz' is required, with no default).
        Unlike `noise`, this is a SINGLE spec for the whole run, not keyed
        by schematic node -- phase noise belongs to the one LO/carrier the
        entire drive is built from, not to any particular schematic
        location. Physically distinct from `noise` (see
        examples/phase_noise_case_study_demo.py's module docstring):
        it's MULTIPLICATIVE (rides on the drive envelope itself,
        `ũ(t)·exp(j*phi(t))`, rather than adding independently of it) and,
        critically, it's injected at the SOURCE, before the nonlinear pass
        -- so when phase_noise is given, _nonlinear_pass() is re-run once
        per realization (each with its own phase draw) rather than once
        total. If None/empty, behavior (and cost) is identical to before
        this parameter existed.
    n_realizations : int
        Number of stochastic realizations (additive noise and/or phase
        noise). Ignored if both noise and phase_noise are None/empty.
    mode : str
        'complex_baseband' (default) or 'real_axis'.
    isolation_threshold_db : float
        Isolation check threshold. See §3.4 of PRD.
    harmonic_suppression_db : float
        Harmonic suppression check threshold (baseband mode only).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    SimulationResult
    """
    sim_warnings: list[str] = []
    nonlinear = nonlinear or {}
    noise = noise or {}
    phase_noise = phase_noise or {}

    # ------------------------------------------------------------------
    # Validate mode selection
    # ------------------------------------------------------------------
    _validate_mode(mode, source, nonlinear, sim_warnings)

    # ------------------------------------------------------------------
    # Validate that every annotated node actually exists in the schematic.
    # nonlinear/noise are the sole source of node identity (PRD §3.2) — the
    # schematic is not scanned for a naming convention, so a typo'd or
    # mismatched label here would otherwise be silently ignored downstream.
    # ------------------------------------------------------------------
    validate_node_labels(schematic, nonlinear.keys(), kind="nonlinear")
    validate_node_labels(
        schematic, noise.keys(), kind="noise", known=schematic.noise_source_names,
    )

    # nl_labels is the single source of truth for NL propagation order —
    # the nonlinear dict's key order (PRD §3.5) — passed to every downstream
    # step that needs to segment the channel.
    nl_labels = list(nonlinear.keys())
    source_label = schematic.source_label
    qubit_label = schematic.qubit_probe_label

    # ------------------------------------------------------------------
    # Build nonlinear node objects
    # ------------------------------------------------------------------
    nl_nodes = (
        build_nonlinear_nodes(nonlinear, mode, warnings_list=sim_warnings)
        if nonlinear else {}
    )

    # ------------------------------------------------------------------
    # SETUP: Extract raw (frequency-domain-only) transfer functions.
    # Waveform-agnostic -- depends only on the schematic, not on `source`'s
    # fs/carrier/mode. See schematic/transfer_function.py module docstring.
    # ------------------------------------------------------------------
    raw_segment_tfs = extract_all_transfer_functions(schematic, nl_labels)
    raw_noise_tfs = (
        extract_noise_source_transfer_functions(schematic, list(noise.keys()))
        if noise else {}
    )

    # ------------------------------------------------------------------
    # SETUP: Diagnostics that only need frequency-domain data (isolation,
    # harmonic suppression) -- these run on the raw TFs directly, before any
    # impulse-response conversion (they only ever touch .freqs/.H).
    # ------------------------------------------------------------------
    _run_isolation_checks(
        nl_labels, raw_segment_tfs, source, mode,
        isolation_threshold_db, harmonic_suppression_db, sim_warnings,
        source_label, qubit_label,
    )

    # ------------------------------------------------------------------
    # Determine the sample rate + initial waveform actually used for
    # convolution this run, and convert the raw TFs to impulse responses at
    # that rate (PRD §3.3):
    #   - real_axis: the schematic's own native rate. h(tau) is computed
    #     directly from it (no H(f) interpolation) and the drive waveform is
    #     resampled to match -- not the other way around.
    #   - complex_baseband: unchanged -- the envelope's own fs/carrier
    #     determine the grid H(f) is interpolated onto.
    # ------------------------------------------------------------------
    # t_resampled/env_resampled (the complex envelope BEFORE carrier
    # modulation, at fs_conv) are kept around uniformly across both modes --
    # not just for real_axis -- because phase-noise injection (below) needs
    # to rotate the envelope by exp(j*phi(t)) BEFORE modulation regardless
    # of mode; in complex_baseband mode there IS no separate modulation
    # step, so env_resampled/t_resampled and v_initial coincide exactly.
    if mode == "real_axis":
        fs_conv = native_sample_rate(next(iter(raw_segment_tfs.values())))
        source.check_sample_rate_for_real_axis(fs_conv, harmonic_order=3)
        t_resampled, env_resampled = source.resampled_envelope_at(fs_conv)
        carrier = np.exp(1j * 2 * np.pi * source.carrier_freq_hz * t_resampled)
        v_initial = np.real(env_resampled * carrier)
    else:
        fs_conv = source.fs
        t_resampled = source.t
        env_resampled = source.envelope_complex.copy()
        v_initial = env_resampled.copy()

    # fs_conv/carrier_hz are only actually used by compute_impulse_response()
    # in complex_baseband mode -- real_axis mode ignores both and derives its
    # own native rate from si_frequency_response directly (see its
    # docstring). Passed unconditionally here anyway since fs_conv is still
    # needed above for the waveform resampling/sample-rate check.
    segment_tfs = {
        key: compute_impulse_response(raw, mode, fs=fs_conv, carrier_hz=source.carrier_freq_hz)
        for key, raw in raw_segment_tfs.items()
    }
    noise_tfs_to_qubit: dict[str, np.ndarray] = {
        label: compute_impulse_response(raw, mode, fs=fs_conv, carrier_hz=source.carrier_freq_hz).h
        for label, raw in raw_noise_tfs.items()
    }

    # ------------------------------------------------------------------
    # PASS 1: NONLINEAR PASS (deterministic)
    # ------------------------------------------------------------------
    v_nl_qubit, intermediate_waveforms = _nonlinear_pass(
        v_initial, nl_labels, nl_nodes, segment_tfs, mode, sim_warnings,
        source_label, qubit_label,
    )

    # ------------------------------------------------------------------
    # PASS 2: NOISE PASS (stochastic)
    # ------------------------------------------------------------------
    noise_enabled = bool(noise)
    phase_noise_enabled = bool(phase_noise)
    ensemble_enabled = noise_enabled or phase_noise_enabled

    def _build_additive_noise_propagator() -> NoisePropagator:
        # Target length must match v_nl_qubit's own (grown) length, not
        # v_initial's pre-convolution length -- v_nl_qubit is longer by
        # len(h)-1 per segment (see _nonlinear_pass's docstring: "full",
        # untruncated linear convolution), and generate_realization()'s
        # output is added directly to a v_nl_qubit-length array below, so it
        # must land at that same length or the addition doesn't broadcast.
        # Noise is drawn over the qubit plane's own full duration,
        # independent of how long the deterministic drive's own convolution
        # happened to grow it -- physically correct (noise is present for as
        # long as the qubit exists in that window), not just a shape fix.
        # Valid to reuse across every realization (including the per-
        # phase-noise-draw ones below): every v_nl_qubit_i has the SAME
        # length as v_nl_qubit, since it's the same fixed segment_tfs/h_k
        # arrays applied to a same-length (only VALUES differ) input.
        h_lengths = {label: len(h) for label, h in noise_tfs_to_qubit.items()}
        psd_cache = psd_cache_for_noise_nodes(
            schematic, noise, len(v_nl_qubit), fs_conv, h_lengths,
        )
        return NoisePropagator(
            psd_cache=psd_cache,
            transfer_functions_to_qubit=noise_tfs_to_qubit,
            n_samples=len(v_nl_qubit),
            fs=fs_conv,
            mode=mode,
        )

    if not ensemble_enabled:
        # Neither kind of noise: v_qubit_ensemble is not a real ensemble --
        # a single v_nl_qubit stand-in, not n_realizations redundant copies
        # (that used to mean gate_fidelity() silently re-solved QuTiP
        # n_realizations times for an identical result). gate_fidelity()
        # uses v_nl_qubit directly for its noise-free result and only
        # touches v_qubit_ensemble when noise_enabled is True.
        v_qubit_ensemble = [v_nl_qubit.copy()]

    elif not phase_noise_enabled:
        # Additive noise only -- the original, cheap path: v_nl_qubit is
        # computed once (above) and reused for every realization, only the
        # additive draw varies.
        propagator = _build_additive_noise_propagator()
        rng = np.random.default_rng(seed)
        v_qubit_ensemble = [
            v_nl_qubit + propagator.generate_realization(rng)
            for _ in range(n_realizations)
        ]

    else:
        # Phase noise enabled (with or without additive noise on top): the
        # nonlinear pass genuinely depends on the phase draw (the
        # nonlinearity sees and reshapes the phase-perturbed waveform, and
        # even a purely LINEAR but dispersive channel treats phase-
        # modulation sidebands differently from the carrier -- see
        # run()'s phase_noise docstring), so it must be re-run once per
        # realization rather than once total. See examples/
        # phase_noise_case_study_demo.py for a direct, worked comparison
        # against the (physically wrong, for this reason) alternative of
        # adding phase noise post-hoc onto the shared v_nl_qubit.
        from ..noise.psd import phase_noise_psd_from_spec
        from ..noise.realization import generate_phase_noise

        bandwidth_hz = float(phase_noise["bandwidth_hz"])
        nyquist_hz = fs_conv / 2.0
        if bandwidth_hz > nyquist_hz:
            sim_warnings.append(
                f"phase_noise bandwidth_hz={bandwidth_hz:.3e} Hz exceeds what "
                f"mode='{mode}' can represent at fs={fs_conv:.3e} Hz "
                f"(Nyquist={nyquist_hz:.3e} Hz) -- clipped to the representable "
                f"range; the requested phase-noise content above {nyquist_hz:.3e} Hz "
                + ("is lost. Consider mode='real_axis' for a much higher native rate."
                   if mode == "complex_baseband" else "is lost.")
            )
            bandwidth_hz = nyquist_hz

        n_draw = len(v_initial)   # phi(t) is applied BEFORE convolution grows the array
        freqs_phi = np.fft.rfftfreq(n_draw, d=1.0 / fs_conv)
        phase_noise_spec = dict(phase_noise)
        phase_noise_spec["bandwidth_hz"] = bandwidth_hz   # possibly clipped above
        phase_psd = phase_noise_psd_from_spec(phase_noise_spec, freqs_phi)

        propagator = _build_additive_noise_propagator() if noise_enabled else None
        rng = np.random.default_rng(seed)

        v_qubit_ensemble = []
        for _ in range(n_realizations):
            phi_i = generate_phase_noise(n_draw, fs_conv, phase_psd, rng=rng)
            v_initial_i = _apply_phase_noise(mode, env_resampled, t_resampled, source.carrier_freq_hz, phi_i)
            # Throwaway warnings list: any warning _nonlinear_pass can raise
            # (e.g. "no transfer function for segment") is structural, about
            # the schematic/segmentation, not about a specific phase draw --
            # already captured once above when v_nl_qubit was computed, so
            # collecting it again n_realizations times here would just spam
            # duplicates.
            v_nl_qubit_i, _ = _nonlinear_pass(
                v_initial_i, nl_labels, nl_nodes, segment_tfs, mode, [],
                source_label, qubit_label,
            )
            if propagator is not None:
                v_nl_qubit_i = v_nl_qubit_i + propagator.generate_realization(rng)
            v_qubit_ensemble.append(v_nl_qubit_i)

    # ------------------------------------------------------------------
    # Emit collected warnings
    # ------------------------------------------------------------------
    for w in sim_warnings:
        warnings.warn(f"SI-QFI: {w}", stacklevel=2)

    return SimulationResult(
        v_nl_qubit=v_nl_qubit,
        v_qubit_ensemble=v_qubit_ensemble,
        fs=fs_conv,
        mode=mode,
        carrier_freq_hz=source.carrier_freq_hz,
        noise_enabled=ensemble_enabled,
        phase_noise_enabled=phase_noise_enabled,
        warnings=sim_warnings,
        n_realizations=n_realizations,
        extra={"intermediate_waveforms": intermediate_waveforms},
    )


# ---------------------------------------------------------------------------
# Nonlinear pass implementation
# ---------------------------------------------------------------------------

def _nonlinear_pass(
    initial_waveform: np.ndarray,
    nl_labels: list[str],
    nl_nodes: dict,
    segment_tfs: dict,
    mode: str,
    sim_warnings: list[str],
    source_label: str,
    qubit_label: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Propagate the source waveform through all NL segments deterministically.

    For each segment k:
      1. Convolve current waveform with h_k(τ).
      2. Apply nonlinearity at the NL node (if present).

    nl_labels gives the propagation order (the `nonlinear` dict's key order —
    see validate_node_labels() / PRD §3.5); it is the same list already used
    to extract segment_tfs, so segment boundaries line up exactly.

    Convolution is `fftconvolve(v, h, mode="full")` with NO truncation back
    down to len(v): this is a true zero-padded linear convolution (no
    circular/wraparound artifacts), and the output is allowed to grow at
    every segment rather than clipping off the tail of the channel's own
    response (which would silently discard real, physical late-arriving
    content -- e.g. reflections -- past the original waveform's duration).
    Each segment's output is therefore longer than its input by len(h)-1
    samples. This means every node's waveform in the chain has a different
    (growing) length -- expected, not a bug. No attempt is made here to
    track *which* absolute time each array's index 0 corresponds to (only
    the sample rate is assumed shared across the whole chain) -- that's a
    deliberate simplification for now (see PRD discussion), revisit if the
    growing array length becomes a real performance problem.

    Parameters
    ----------
    initial_waveform : np.ndarray
        Drive waveform samples to convolve through the segment chain --
        source.envelope_complex for baseband mode, or
        source.rf_waveform_at(fs_conv)[1] for real-axis mode (see run()).
        Passed in explicitly since only the caller knows which sample rate
        this run actually uses (PRD §3.3) -- real-axis mode may run at the
        schematic's native rate rather than source's own.

    Returns
    -------
    (v_final, intermediate_waveforms) : the waveform at qubit_label (also
    intermediate_waveforms[qubit_label]), and a dict of the waveform at
    every node in the chain (source_label, each NL label in order, and
    qubit_label), keyed by node label -- so a caller can inspect the signal
    at any intermediate NL probe, not just the final qubit-plane output.
    """
    v = initial_waveform.copy()
    intermediate_waveforms: dict[str, np.ndarray] = {source_label: v.copy()}

    # Build ordered segment list: source_label→NL_1, NL_1→NL_2, ..., NL_n→qubit_label
    # NOTE: This is ugly since we have this code duplicated when creating the segment list, figure out nice way to refactor so its not duplicated
    all_labels = [source_label] + list(nl_labels) + [qubit_label]
    segments = list(zip(all_labels[:-1], all_labels[1:]))

    for k, (lin, lout) in enumerate(segments):
        # Linear channel convolution
        tf = segment_tfs.get((lin, lout))
        if tf is not None:
            h = tf.h
            v = fftconvolve(v, h, mode="full")
        else:
            sim_warnings.append(
                f"No transfer function found for segment ({lin} → {lout}). "
                f"Assuming identity (no filtering)."
            )

        # NOTE: Kind of ugly how we have nl_nodes and nl_labels passed into this function, is that really necessary? Shouldnt the latter be contained in the former. 
        # Apply nonlinearity at the output node of this segment
        if lout in nl_nodes:
            nl = nl_nodes[lout]
            if mode == "complex_baseband":
                v = nl.apply_baseband(v)
            else:
                v = nl.apply_real_axis(v)

        intermediate_waveforms[lout] = v.copy()

    return v, intermediate_waveforms


def _apply_phase_noise(
    mode: str,
    env_resampled: np.ndarray,
    t_resampled: np.ndarray,
    carrier_hz: float,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Rotate the (pre-modulation) complex envelope by a phase-noise
    realization phi(t) [rad], then modulate onto the carrier if this mode
    needs a real RF waveform -- the source-side counterpart to
    run()'s `phase_noise` parameter.

    Mirrors quantum.demodulate()'s own physics in reverse: a noisy LO gives
    v(t) = Re{ũ(t)·exp(j(2*pi*f_c*t + phi(t)))} = Re{[ũ(t)·exp(j*phi(t))]·
    exp(j*2*pi*f_c*t)} -- i.e. phase noise is exactly a rotation of the
    envelope PRIOR to modulation, regardless of mode. In complex_baseband
    mode there is no separate modulation step at all (the "RF waveform" IS
    the envelope), so this reduces to just the rotation.

    Parameters
    ----------
    mode : str
        'complex_baseband' or 'real_axis'.
    env_resampled : np.ndarray, complex128
        The envelope at the sample rate actually used for propagation this
        run (fs_conv) -- equals v_initial itself in complex_baseband mode,
        or the resampled-but-not-yet-modulated envelope in real_axis mode
        (see SourceWaveform.resampled_envelope_at()).
    t_resampled : np.ndarray
        Time array (seconds) matching env_resampled -- only used to build
        the carrier in real_axis mode.
    carrier_hz : float
    phi : np.ndarray, float64
        Phase-noise realization (radians), same length as env_resampled.

    Returns
    -------
    np.ndarray
        Complex (complex_baseband) or real (real_axis) waveform, same
        length as env_resampled -- suitable as _nonlinear_pass()'s
        initial_waveform.
    """
    env_noisy = env_resampled * np.exp(1j * phi)
    if mode == "complex_baseband":
        return env_noisy
    carrier = np.exp(1j * 2 * np.pi * carrier_hz * t_resampled)
    return np.real(env_noisy * carrier)


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _validate_mode(
    mode: str,
    source: SourceWaveform,
    nonlinear: dict,
    sim_warnings: list[str],
) -> None:
    if mode not in ("complex_baseband", "real_axis"):
        raise ValueError(f"mode must be 'complex_baseband' or 'real_axis', got '{mode}'.")

    # Real-axis mode's sample rate check happens later in run(), once the
    # schematic's native sample rate is known (PRD §3.3) -- it can't be
    # checked here, before extraction, since it no longer depends on
    # source.fs alone.

    # Narrowband check for complex baseband mode
    if mode == "complex_baseband":
        ratio = source.narrowband_ratio()
        if ratio > _NARROWBAND_RATIO_THRESHOLD:
            sim_warnings.append(
                f"Narrowband ratio (BW/carrier) = {ratio:.3f} > "
                f"{_NARROWBAND_RATIO_THRESHOLD}. Complex baseband approximation "
                f"may be inaccurate. Consider using mode='real_axis'."
            )

    # Volterra models are only valid in real_axis mode
    for label, spec in nonlinear.items():
        if spec.get("model") == "volterra" and mode == "complex_baseband":
            raise ValueError(
                f"Node '{label}': model='volterra' requires mode='real_axis'."
            )


def _run_isolation_checks(
    nl_labels: list[str],
    segment_tfs: dict,
    source: SourceWaveform,
    mode: str,
    isolation_threshold_db: float,
    harmonic_suppression_db: float,
    sim_warnings: list[str],
    source_label: str,
    qubit_label: str,
) -> None:
    """
    Run isolation and harmonic suppression checks on extracted transfer functions.
    Appends warning strings to sim_warnings.
    """
    if len(nl_labels) < 2:
        return   # Need at least two NL nodes to check inter-stage isolation

    carrier_hz = source.carrier_freq_hz
    bw = source.bandwidth_hz
    signal_band = (carrier_hz - bw / 2, carrier_hz + bw / 2)

    all_labels = [source_label] + list(nl_labels) + [qubit_label]
    segments = list(zip(all_labels[:-1], all_labels[1:]))

    for k in range(len(nl_labels)):
        # Forward: NL_k → NL_{k+1}
        lin, lout = all_labels[k + 1], all_labels[k + 2]
        tf_fwd = segment_tfs.get((lin, lout))
        tf_rev = segment_tfs.get((lout, lin))

        if tf_fwd is None or tf_rev is None:
            continue

        iso_db = compute_isolation_db(tf_fwd, tf_rev, signal_band)
        if iso_db > isolation_threshold_db:
            sim_warnings.append(
                f"Isolation check FAIL: reverse coupling from {lout} → {lin} "
                f"is {iso_db:.1f} dB (threshold {isolation_threshold_db:.0f} dB). "
                f"Feedforward propagation model may be inaccurate. "
                f"Consider adding an isolator between these stages."
            )

        # Harmonic check (baseband mode only)
        if mode == "complex_baseband" and tf_fwd is not None:
            _check_harmonic_suppression(
                tf_fwd, carrier_hz, harmonic_suppression_db, lin, lout, sim_warnings
            )


def _check_harmonic_suppression(
    tf,
    carrier_hz: float,
    threshold_db: float,
    label_in: str,
    label_out: str,
    sim_warnings: list[str],
) -> None:
    """
    Check that H(3·f_carrier) is suppressed relative to H(f_carrier) by at least
    threshold_db. Warns and suggests real_axis mode if not.
    """
    if tf.freqs is None or len(tf.freqs) == 0:
        return

    from scipy.interpolate import interp1d
    mag = np.abs(tf.H)
    interp = interp1d(tf.freqs, mag, bounds_error=False, fill_value=0.0)

    H_at_carrier = float(interp(carrier_hz))
    H_at_3fc = float(interp(3.0 * carrier_hz))

    if H_at_carrier < 1e-12:
        return   # Can't compute meaningful ratio

    suppression_db = 20.0 * np.log10(H_at_3fc / H_at_carrier + 1e-30)
    if suppression_db > -threshold_db:
        sim_warnings.append(
            f"Harmonic suppression check FAIL for segment ({label_in} → {label_out}): "
            f"H(3·f_carrier) is only {-suppression_db:.1f} dB below H(f_carrier) "
            f"(threshold {threshold_db:.0f} dB). Third harmonic energy may reach the "
            f"qubit plane. Consider using mode='real_axis' for accurate simulation."
        )



# ---------------------------------------------------------------------------
# Cross-validation utility
# ---------------------------------------------------------------------------

def compare_modes(
    result_baseband: SimulationResult,
    result_realaxis: SimulationResult,
    tolerance_db: float = 1.0,
) -> dict:
    """
    Compare NL-only waveforms from complex baseband and real-axis runs.

    When the narrowband assumption holds, both modes should produce identical
    in-band output at the qubit plane. Disagreement indicates harmonic
    content is significant and real-axis mode should be preferred.

    Parameters
    ----------
    result_baseband : SimulationResult  from mode='complex_baseband'
    result_realaxis : SimulationResult  from mode='real_axis'
    tolerance_db : float  Maximum allowed RMS difference in dB before warning.

    Returns
    -------
    dict with keys 'rms_error_db', 'agree', 'message'.
    """
    v_bb = result_baseband.v_nl_qubit
    v_ra = result_realaxis.v_nl_qubit

    # Demodulate real-axis to baseband for fair comparison
    if result_realaxis.mode == "real_axis":
        from ..quantum import demodulate
        t = np.arange(len(v_ra)) / result_realaxis.fs
        fc = result_realaxis.carrier_freq_hz
        i_ra, q_ra = demodulate(np.real(v_ra), t, fc)
        v_ra_bb = i_ra + 1j * q_ra
    else:
        v_ra_bb = v_ra

    # Trim to same length
    N = min(len(v_bb), len(v_ra_bb))
    err = v_bb[:N] - v_ra_bb[:N]
    rms_err = np.sqrt(np.mean(np.abs(err) ** 2))
    rms_sig = np.sqrt(np.mean(np.abs(v_bb[:N]) ** 2))

    if rms_sig < 1e-15:
        rms_err_db = 0.0
    else:
        rms_err_db = 20.0 * np.log10(rms_err / rms_sig + 1e-30)

    agree = rms_err_db < -tolerance_db
    msg = (
        f"Mode comparison: RMS error = {rms_err_db:.1f} dB. "
        + ("Modes agree — narrowband assumption valid." if agree
           else f"Modes DISAGREE beyond {tolerance_db:.0f} dB tolerance. "
                f"Harmonic content may be significant. Prefer real_axis mode.")
    )

    if not agree:
        warnings.warn(f"SI-QFI: {msg}", stacklevel=2)

    return {"rms_error_db": rms_err_db, "agree": agree, "message": msg}
