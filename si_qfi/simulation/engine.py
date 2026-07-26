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
    extract_noise_transfer_functions,
    compute_impulse_response,
    native_sample_rate,
    compute_isolation_db,
)
from ..source.waveform import SourceWaveform
from ..nonlinear.registry import build_nonlinear_nodes
from ..noise.propagation import NoisePropagator


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Output of siq.run(). Contains the ensemble of qubit-plane waveforms
    and all diagnostic warnings. Pass to siq.quantum.gate_fidelity().
    """
    v_nl_qubit: np.ndarray          # Deterministic NL-distorted waveform, shape (N,)
    v_qubit_ensemble: list[np.ndarray]  # v_nl_qubit + noise, one per realization -- only meaningful when noise_enabled (see below)
    fs: float                       # Sample rate (Hz) shared by every array in this result
    mode: str                       # 'complex_baseband' or 'real_axis'
    carrier_freq_hz: float
    noise_enabled: bool = False     # True iff a non-empty `noise` dict was passed to run() --
                                     # lets gate_fidelity() know whether v_qubit_ensemble holds
                                     # real stochastic realizations or is just a v_nl_qubit stand-in
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
        Noise node annotation dict {probe_label: spec_dict}.
        If None, all realizations are identical to v_nl_qubit (noiseless).
    n_realizations : int
        Number of stochastic noise realizations. Ignored if noise is None.
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
    validate_node_labels(schematic, noise.keys(), kind="noise")

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
        extract_noise_transfer_functions(schematic, list(noise.keys()))
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
    if mode == "real_axis":
        fs_conv = native_sample_rate(next(iter(raw_segment_tfs.values())))
        source.check_sample_rate_for_real_axis(fs_conv, harmonic_order=3)
        _, v_initial = source.rf_waveform_at(fs_conv)
    else:
        fs_conv = source.fs
        v_initial = source.envelope_complex.copy()

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
    if noise_enabled:
        propagator = NoisePropagator(
            noise_annotation=noise,
            transfer_functions_to_qubit=noise_tfs_to_qubit,
            n_samples=len(v_initial),
            fs=fs_conv,
            mode=mode,
        )
        rng = np.random.default_rng(seed)
        v_qubit_ensemble = [
            v_nl_qubit + propagator.generate_realization(rng)
            for _ in range(n_realizations)
        ]
    else:
        # No noise: v_qubit_ensemble is not a real ensemble -- a single
        # v_nl_qubit stand-in, not n_realizations redundant copies (that
        # used to mean gate_fidelity() silently re-solved QuTiP
        # n_realizations times for an identical result). gate_fidelity()
        # uses v_nl_qubit directly for its noise-free result and only
        # touches v_qubit_ensemble when noise_enabled is True.
        v_qubit_ensemble = [v_nl_qubit.copy()]

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
        noise_enabled=noise_enabled,
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
