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

from ..schematic.loader import SISchematic
from ..schematic.transfer_function import (
    extract_all_transfer_functions,
    extract_noise_transfer_functions,
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
    v_qubit_ensemble: list[np.ndarray]  # v_nl_qubit + noise, one per realization
    t_array: np.ndarray             # Shared time axis (seconds)
    mode: str                       # 'complex_baseband' or 'real_axis'
    carrier_freq_hz: float
    warnings: list[str] = field(default_factory=list)
    n_realizations: int = 0

    def __repr__(self) -> str:
        return (
            f"SimulationResult(mode='{self.mode}', "
            f"N={len(self.t_array)} samples, "
            f"n_realizations={self.n_realizations}, "
            f"warnings={len(self.warnings)})"
        )


# ---------------------------------------------------------------------------
# Diagnostic constants
# ---------------------------------------------------------------------------

_ISOLATION_THRESHOLD_DB = -20.0       # Warn if reverse coupling exceeds this
_HARMONIC_SUPPRESSION_DB = 30.0       # Warn if 3f_c attenuation is less than this
_NARROWBAND_RATIO_THRESHOLD = 0.05    # Warn if BW/carrier > 5%
_MEMORY_FLAG_RATIO = 0.1              # Warn if τ_reflection/T_pulse > 0.1


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
    # Build nonlinear node objects
    # ------------------------------------------------------------------
    nl_nodes = build_nonlinear_nodes(nonlinear, mode) if nonlinear else {}

    # ------------------------------------------------------------------
    # SETUP: Extract transfer functions
    # ------------------------------------------------------------------
    # Segment TFs (SOURCE → NL_1 → ... → QUBIT_PROBE)
    segment_tfs = extract_all_transfer_functions(schematic, source, mode)

    # Noise TFs: each noise node → QUBIT_PROBE (direct, full path)
    noise_tfs_to_qubit: dict[str, np.ndarray] = {}
    if noise:
        noise_tfs_to_qubit = extract_noise_transfer_functions(
            schematic, list(noise.keys()), source, mode
        )

    # ------------------------------------------------------------------
    # SETUP: Diagnostics
    # ------------------------------------------------------------------
    _run_isolation_checks(
        schematic, segment_tfs, source, mode,
        isolation_threshold_db, harmonic_suppression_db, sim_warnings
    )

    # ------------------------------------------------------------------
    # PASS 1: NONLINEAR PASS (deterministic)
    # ------------------------------------------------------------------
    v_nl_qubit = _nonlinear_pass(
        source, schematic, nl_nodes, segment_tfs, mode, sim_warnings
    )

    # ------------------------------------------------------------------
    # PASS 2: NOISE PASS (stochastic)
    # ------------------------------------------------------------------
    if noise:
        propagator = NoisePropagator(
            noise_annotation=noise,
            transfer_functions_to_qubit=noise_tfs_to_qubit,
            source_waveform=source,
            mode=mode,
        )
        rng = np.random.default_rng(seed)
        v_qubit_ensemble = [
            v_nl_qubit + propagator.generate_realization(rng)
            for _ in range(n_realizations)
        ]
    else:
        # No noise: all realizations are identical to the NL waveform
        v_qubit_ensemble = [v_nl_qubit.copy() for _ in range(n_realizations)]
        if n_realizations > 1:
            sim_warnings.append(
                "No noise nodes annotated; all realizations are identical. "
                "Gate fidelity std will be zero."
            )

    # ------------------------------------------------------------------
    # Emit collected warnings
    # ------------------------------------------------------------------
    for w in sim_warnings:
        warnings.warn(f"SI-QFI: {w}", stacklevel=2)

    return SimulationResult(
        v_nl_qubit=v_nl_qubit,
        v_qubit_ensemble=v_qubit_ensemble,
        t_array=source.t,
        mode=mode,
        carrier_freq_hz=source.carrier_freq_hz,
        warnings=sim_warnings,
        n_realizations=n_realizations,
    )


# ---------------------------------------------------------------------------
# Nonlinear pass implementation
# ---------------------------------------------------------------------------

def _nonlinear_pass(
    source: SourceWaveform,
    schematic: SISchematic,
    nl_nodes: dict,
    segment_tfs: dict,
    mode: str,
    sim_warnings: list[str],
) -> np.ndarray:
    """
    Propagate the source waveform through all NL segments deterministically.

    For each segment k:
      1. Convolve current waveform with h_k(τ).
      2. Apply nonlinearity at the NL node (if present).

    Returns the final waveform at QUBIT_PROBE.
    """
    # Initial waveform from source
    if mode == "complex_baseband":
        v = source.envelope_complex.copy()
    else:
        v = source.rf_waveform.copy()

    N = len(v)

    # Build ordered segment list: SOURCE→NL_1, NL_1→NL_2, ..., NL_n→QUBIT_PROBE
    nl_labels = schematic.nl_probe_labels
    source_label = "SOURCE"
    qubit_label = "QUBIT_PROBE"
    all_labels = [source_label] + nl_labels + [qubit_label]
    segments = list(zip(all_labels[:-1], all_labels[1:]))

    for k, (lin, lout) in enumerate(segments):
        # Linear channel convolution
        tf = segment_tfs.get((lin, lout))
        if tf is not None:
            h = tf.h
            v = fftconvolve(v, h, mode="full")[:N]
        else:
            sim_warnings.append(
                f"No transfer function found for segment ({lin} → {lout}). "
                f"Assuming identity (no filtering)."
            )

        # Apply nonlinearity at the output node of this segment
        if lout in nl_nodes:
            nl = nl_nodes[lout]
            if mode == "complex_baseband":
                v = nl.apply_baseband(v)
            else:
                v = nl.apply_real_axis(v)

            # Memory regime warning
            if tf is not None:
                _check_memory_regime(tf, source, nl, lout, sim_warnings)

    return v


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

    if mode == "real_axis":
        source.check_sample_rate_for_real_axis(harmonic_order=3)

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
    schematic: SISchematic,
    segment_tfs: dict,
    source: SourceWaveform,
    mode: str,
    isolation_threshold_db: float,
    harmonic_suppression_db: float,
    sim_warnings: list[str],
) -> None:
    """
    Run isolation and harmonic suppression checks on extracted transfer functions.
    Appends warning strings to sim_warnings.
    """
    nl_labels = schematic.nl_probe_labels
    if len(nl_labels) < 2:
        return   # Need at least two NL nodes to check inter-stage isolation

    carrier_hz = source.carrier_freq_hz
    bw = source.bandwidth_hz
    signal_band = (carrier_hz - bw / 2, carrier_hz + bw / 2)

    all_labels = ["SOURCE"] + nl_labels + ["QUBIT_PROBE"]
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


def _check_memory_regime(
    tf,
    source: SourceWaveform,
    nl_node,
    label: str,
    sim_warnings: list[str],
) -> None:
    """
    Check τ_reflection / T_pulse and warn if memory effects may be significant.
    """
    # Estimate dominant reflection delay from TDR of S11
    # (h(τ) has a peak at the reflection delay)
    h = np.real(tf.h) if np.iscomplexobj(tf.h) else tf.h
    if len(h) < 4:
        return

    # Peak of the impulse response (excluding t=0 direct path)
    h_delayed = np.abs(h[1:])
    if np.max(h_delayed) < 1e-6 * np.max(np.abs(h)):
        return   # No significant reflections

    tau_idx = np.argmax(h_delayed) + 1
    tau_s = tau_idx * tf.dt

    # Estimate pulse width as duration where |envelope| > 10% of peak
    env = np.abs(source.envelope_complex)
    threshold = 0.1 * np.max(env)
    active = np.where(env > threshold)[0]
    if len(active) < 2:
        return
    T_pulse_s = (active[-1] - active[0]) * source.dt

    ratio = tau_s / T_pulse_s if T_pulse_s > 0 else 0.0

    if ratio > 1.0:
        sim_warnings.append(
            f"Memory regime CRITICAL at node '{label}': τ_reflection/T_pulse = "
            f"{ratio:.2f} > 1.0. Reflection arrives after the pulse ends and will "
            f"appear as a spurious drive event in the QuTiP simulation."
        )
    elif ratio > _MEMORY_FLAG_RATIO:
        if nl_node.memory_depth == 0:
            sim_warnings.append(
                f"Memory regime WARNING at node '{label}': τ_reflection/T_pulse = "
                f"{ratio:.2f} > {_MEMORY_FLAG_RATIO}. Memoryless NL model may be "
                f"insufficient. Consider using model='memory_polynomial' with "
                f"memory_depth >= {max(1, int(np.ceil(ratio * 10)))}."
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
        t = result_realaxis.t_array
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
