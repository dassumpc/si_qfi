"""
si_qfi.noise.psd
================
Compute one-sided noise power spectral density S_v(f) [V²/Hz] from
user-supplied noise specifications.

Three spec types are supported (matching the noise_nodes annotation format):

  'noise_figure'  — amplifier characterised by noise figure NF (dB) + temperature
  'noise_density' — direct PSD specification in V²/Hz
  'thermal'       — Johnson-Nyquist thermal noise at temperature T through a
                    resistive element. Loss in dB can be supplied explicitly or
                    inferred from the SI transfer function magnitude (done by
                    the engine, not here).
"""

from __future__ import annotations

import numpy as np
from typing import Any

# Boltzmann constant (J/K)
_KB = 1.380649e-23


def psd_from_spec(
    spec: dict[str, Any],
    freqs: np.ndarray,
    source_impedance_ohm: float = 50.0,
) -> np.ndarray:
    """
    Compute one-sided noise PSD S_v(f) [V²/Hz] over the given frequency array.

    Parameters
    ----------
    spec : dict
        Noise node specification. Must contain 'type' key.
    freqs : np.ndarray, shape (F,)
        Positive frequency values at which to evaluate PSD (Hz).
        Zero frequency may be included but is usually excluded for RF.
    source_impedance_ohm : float
        Source impedance used to convert noise temperature → voltage PSD.
        Default 50 Ω.

    Returns
    -------
    S_v : np.ndarray, float64, shape (F,)
        One-sided noise voltage PSD [V²/Hz] at each frequency.
    """
    noise_type = spec.get("type")
    if noise_type is None:
        raise ValueError("Noise spec must contain a 'type' key.")

    if noise_type == "noise_figure":
        return _psd_noise_figure(spec, freqs, source_impedance_ohm)
    elif noise_type == "noise_density":
        return _psd_noise_density(spec, freqs)
    elif noise_type == "thermal":
        return _psd_thermal(spec, freqs, source_impedance_ohm)
    else:
        raise ValueError(
            f"Unknown noise type '{noise_type}'. "
            f"Use 'noise_figure', 'noise_density', or 'thermal'."
        )


def _psd_noise_figure(
    spec: dict,
    freqs: np.ndarray,
    R: float,
) -> np.ndarray:
    """
    Noise figure NF (dB) + physical temperature T_phys (K).

    Excess noise temperature:
        T_eff = T_phys · (10^(NF/10) - 1)

    One-sided voltage PSD into impedance R:
        S_v(f) = 4 · k_B · T_eff · R   [V²/Hz]
        (one-sided: factor 4 rather than 2; standard convention for
         noise voltage referred to a matched source)
    """
    nf_db = float(spec["noise_figure_db"])
    t_phys = float(spec.get("temperature_k", 290.0))
    t_eff = t_phys * (10.0 ** (nf_db / 10.0) - 1.0)
    S_v = 4.0 * _KB * t_eff * R
    return np.full_like(freqs, S_v, dtype=float)


def _psd_noise_density(
    spec: dict,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Direct one-sided PSD specification.

    Accepts either:
      'single_sided_psd_v2_per_hz'  — already in V²/Hz
      'single_sided_psd_dbm_hz'     — in dBm/Hz, converted to V²/Hz into 50 Ω
    """
    if "single_sided_psd_v2_per_hz" in spec:
        S_v = float(spec["single_sided_psd_v2_per_hz"])
    elif "single_sided_psd_dbm_hz" in spec:
        psd_dbm_hz = float(spec["single_sided_psd_dbm_hz"])
        # Convert dBm/Hz → W/Hz → V²/Hz  (into 50 Ω: P = V²/R, V² = P·R)
        S_w = 10.0 ** (psd_dbm_hz / 10.0) * 1e-3
        S_v = S_w * 50.0
    else:
        raise ValueError(
            "noise_density spec must contain 'single_sided_psd_v2_per_hz' "
            "or 'single_sided_psd_dbm_hz'."
        )
    return np.full_like(freqs, S_v, dtype=float)


def _psd_thermal(
    spec: dict,
    freqs: np.ndarray,
    R: float,
) -> np.ndarray:
    """
    Johnson-Nyquist thermal noise at temperature T (K).

    Optional 'loss_db' key applies additional resistive loss (attenuator noise
    model): an attenuator with loss L (linear) at temperature T contributes
    T_eff = T · (L - 1), so S_v = 4·k_B·T·(L-1)·R.

    If loss_db is absent, returns flat thermal PSD: S_v = 4·k_B·T·R.
    """
    t_k = float(spec.get("temperature_k", 290.0))
    loss_db = float(spec.get("loss_db", 0.0))
    loss_linear = 10.0 ** (loss_db / 10.0)   # ≥ 1 for passive loss
    t_eff = t_k * max(loss_linear - 1.0, 0.0) if loss_db > 0 else t_k
    S_v = 4.0 * _KB * t_eff * R
    return np.full_like(freqs, S_v, dtype=float)


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
