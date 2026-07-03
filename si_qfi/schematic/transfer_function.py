"""
si_qfi.schematic.transfer_function
====================================
Extract voltage transfer functions H(f) between adjacent probe pairs from a
SignalIntegrity schematic, and convert to time-domain impulse responses h(τ).

Two output forms:
  - Complex baseband: H̃(f) = H(f + f_carrier), impulse response h̃(τ)
  - Real-axis: H(f) directly, impulse response h(τ) (real-valued)

# --- CURSOR NOTE (HIGH PRIORITY) ---
# This is the most critical SI API integration point in the codebase.
#
# The standard SI pattern for computing transfer functions between ports:
#
#   from SignalIntegrity.App.SignalIntegrityAppHeadless import (
#       SignalIntegrityAppHeadless,
#   )
#   app = SignalIntegrityAppHeadless()
#   app.OpenProjectFile("myschematic.si")
#
#   # Get S-parameters of the full schematic:
#   (sp, name) = app.SParameters()
#   # sp is a SParameters object with:
#   #   sp.m  — number of ports
#   #   sp.f() — frequency list
#   #   sp[i][j] — S_{i,j} at each frequency (list of complex values)
#   #
#   # Transfer function from port j to port i:
#   #   H_{ij}(f) = S_{ij}(f)
#   #
#   # Port numbering corresponds to the order probes appear in the schematic.
#   # Need to map probe labels → port indices.
#   # Check: app.schematic.deviceList ordering or a dedicated port-name method.
#
#   # Alternative: if SI supports extracting a sub-network between two named probes,
#   # use that directly. Check SI documentation / source for port naming API.
# -------------------
"""

from __future__ import annotations

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Any
from scipy.signal import fftconvolve


@dataclass
class TransferFunction:
    """
    Voltage transfer function between two schematic nodes.

    Attributes
    ----------
    label_in : str   Source probe label (input node).
    label_out : str  Destination probe label (output node).
    freqs : np.ndarray, float64, shape (F,)   Frequency array (Hz).
    H : np.ndarray, complex128, shape (F,)    Complex transfer function.
    h : np.ndarray, float64 or complex128     Time-domain impulse response.
    dt : float  Sample interval of h (seconds).
    """
    label_in: str
    label_out: str
    freqs: np.ndarray
    H: np.ndarray      # frequency domain
    h: np.ndarray      # time domain
    dt: float


def extract_all_transfer_functions(
    schematic,           # SISchematic
    source_waveform,     # SourceWaveform
    mode: str,
) -> dict[tuple[str, str], TransferFunction]:
    """
    Extract transfer functions for all required node pairs.

    Node pairs needed:
      Segment pairs: (SOURCE, NL_1), (NL_1, NL_2), ..., (NL_n, QUBIT_PROBE)
      Noise pairs:   (noise_node_j, QUBIT_PROBE) for each j — full path

    The engine calls this once during setup. Results are cached and reused
    for all realizations.

    Parameters
    ----------
    schematic : SISchematic
        Loaded schematic object (contains si_app).
    source_waveform : SourceWaveform
        Used to determine sample rate and carrier frequency.
    mode : str
        'complex_baseband' or 'real_axis'.

    Returns
    -------
    dict mapping (label_in, label_out) → TransferFunction.

    # --- CURSOR NOTE ---
    # Implementation outline:
    #
    # 1. Call si_app.SParameters() to get the full S-parameter matrix.
    # 2. Build a mapping of probe_label → port_index from the schematic.
    # 3. For each required (label_in, label_out) pair:
    #    a. Look up port indices i_in, i_out.
    #    b. Extract H(f) = sp[i_out][i_in] at each frequency.
    #    c. If mode == 'complex_baseband': shift to baseband (§3.3 of PRD).
    #    d. IFFT to get impulse response h(τ).
    # 4. Return dict of TransferFunction objects.
    #
    # Frequency array from SI:
    #   freqs = sp.f()   # list → np.array
    # Transfer function (voltage gain, not power):
    #   H_ij = np.array([sp[f_idx][i_out][i_in] for f_idx in range(len(freqs))])
    #   (confirm indexing order: sp[port_out][port_in] vs sp[port_in][port_out])
    # -------------------
    """
    # Build list of all required pairs
    nl_labels = schematic.nl_probe_labels
    source_label = "SOURCE"       # synthetic label for the voltage source port
    qubit_label = "QUBIT_PROBE"

    # Segment pairs for NL pass
    all_labels = [source_label] + nl_labels + [qubit_label]
    segment_pairs = list(zip(all_labels[:-1], all_labels[1:]))

    # Noise pairs: source → each possible noise injection point → QUBIT_PROBE
    # These span potentially multiple segments; we extract them as full paths.
    # For now we return segment pairs only; the engine composes for longer paths.
    # --- CURSOR NOTE ---
    # For noise propagation we need h_{j→qubit} directly (not composed from segments).
    # Extract these as separate port pairs from the full schematic S-parameters.
    # The full S-parameter matrix gives us any-to-any transfer function directly.
    # -------------------

    result: dict[tuple[str, str], TransferFunction] = {}

    for (lin, lout) in segment_pairs:
        tf = _extract_single_tf(
            schematic.si_app, lin, lout,
            source_waveform.fs, mode, source_waveform.carrier_freq_hz
        )
        result[(lin, lout)] = tf

    return result


def extract_noise_transfer_functions(
    schematic,
    noise_node_labels: list[str],
    source_waveform,
    mode: str,
) -> dict[str, np.ndarray]:
    """
    Extract impulse responses h_{j→qubit}(τ) for each noise node j.

    Parameters
    ----------
    schematic : SISchematic
    noise_node_labels : list of str
        Labels of noise-annotated nodes (any schematic node, not just NL probes).
    source_waveform : SourceWaveform
    mode : str

    Returns
    -------
    dict mapping noise_node_label → impulse response array h_{j→qubit}.
    """
    result: dict[str, np.ndarray] = {}
    for label in noise_node_labels:
        tf = _extract_single_tf(
            schematic.si_app, label, "QUBIT_PROBE",
            source_waveform.fs, mode, source_waveform.carrier_freq_hz
        )
        result[label] = tf.h
    return result


def _extract_single_tf(
    si_app: Any,
    label_in: str,
    label_out: str,
    fs: float,
    mode: str,
    carrier_hz: float,
) -> TransferFunction:
    """
    Extract a single transfer function H(f) between two named probes.

    # --- CURSOR NOTE ---
    # This is where the SI API call lives. Implement:
    #
    #   (sp, name) = si_app.SParameters()
    #   freqs = np.array(sp.f())
    #   i_in  = _label_to_port_index(si_app, label_in)
    #   i_out = _label_to_port_index(si_app, label_out)
    #   H_raw = np.array([sp[k][i_out][i_in] for k in range(len(freqs))], dtype=complex)
    #   # Note: sp indexing may be sp[freq_index][port_out][port_in] — verify.
    #
    # Then call _to_impulse_response(freqs, H_raw, fs, mode, carrier_hz).
    # -------------------
    """
    raise NotImplementedError(
        f"_extract_single_tf('{label_in}' → '{label_out}'): "
        "Implement using SignalIntegrity SParameters API. See CURSOR NOTE."
    )


def _label_to_port_index(si_app: Any, label: str) -> int:
    """
    Map a probe label to its port index in the SI S-parameter matrix.

    # --- CURSOR NOTE ---
    # Port ordering in SI's S-parameter output matches the order probes appear
    # in the schematic device list. Verify by:
    #   for i, device in enumerate(si_app.schematic.deviceList):
    #       print(i, device.partname, device.propertiesByName.get('ref', '?'))
    # Build the mapping once and cache it.
    # -------------------
    """
    raise NotImplementedError(
        f"_label_to_port_index('{label}'): implement using SI schematic device list."
    )


def _to_impulse_response(
    freqs: np.ndarray,
    H: np.ndarray,
    fs_target: float,
    mode: str,
    carrier_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert a frequency-domain transfer function to a time-domain impulse response.

    Parameters
    ----------
    freqs : np.ndarray   Frequency array from SI (Hz), one-sided (f ≥ 0).
    H : np.ndarray       Complex transfer function at freqs.
    fs_target : float    Target sample rate of the time-domain output (Hz).
    mode : str           'complex_baseband' or 'real_axis'.
    carrier_hz : float   Carrier frequency (Hz) for baseband shifting.

    Returns
    -------
    h : np.ndarray   Time-domain impulse response.
    freqs_out : np.ndarray  Frequency array used for IFFT.
    dt : float   Sample interval of h.
    """
    if mode == "complex_baseband":
        return _tf_to_baseband_impulse(freqs, H, fs_target, carrier_hz)
    else:
        return _tf_to_realaxis_impulse(freqs, H, fs_target)


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
    df = fs / N
    f_bb = np.fft.fftfreq(N, d=1.0 / fs)   # two-sided baseband freqs

    # Interpolate H(f) onto the shifted grid: H̃(f_bb) = H(f_bb + carrier)
    f_rf = f_bb + carrier_hz   # RF frequencies corresponding to baseband grid
    H_tilde = _interpolate_tf(freqs, H, f_rf)

    # IFFT → complex baseband impulse response
    h_tilde = np.fft.ifft(np.fft.ifftshift(H_tilde)) * N * df
    return h_tilde.astype(complex), f_bb, 1.0 / fs


def _tf_to_realaxis_impulse(
    freqs: np.ndarray,
    H: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build the full two-sided (real-axis) transfer function and IFFT to get h(τ).
    """
    N = int(round(fs / (freqs[1] - freqs[0]))) if len(freqs) > 1 else 1024
    N = max(N, 64)
    df = fs / N
    f_full = np.fft.rfftfreq(N, d=1.0 / fs)

    H_interp = _interpolate_tf(freqs, H, f_full)
    h = np.fft.irfft(H_interp, n=N)
    return h.astype(float), f_full, 1.0 / fs


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
