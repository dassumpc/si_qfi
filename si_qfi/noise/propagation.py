"""
si_qfi.noise.propagation
========================
For each annotated noise node j, propagate noise realizations to the qubit
plane via the precomputed transfer function h_{j→qubit}(τ).

Because noise is linear, propagation is independent of the nonlinear
segmentation and runs as a separate, simpler pass (PRD §4, Design Principle 4).

The total noise at the qubit plane for one realization is:
    v_noise_qubit(t) = Σ_j  [h_{j→qubit} * v_noise_j](t)

All noise contributions are summed and added to the deterministic NL waveform
before the QuTiP simulation.
"""

from __future__ import annotations

import numpy as np
from typing import Any
from scipy.signal import fftconvolve

from .psd import psd_from_spec
from .realization import generate_baseband_noise, generate_rf_noise


class NoisePropagator:
    """
    Manages per-node noise generation and propagation to QUBIT_PROBE.

    Parameters
    ----------
    noise_annotation : dict
        User-supplied noise_nodes dict: {probe_label: spec_dict}.
    transfer_functions_to_qubit : dict
        Precomputed impulse responses from each noise node j to QUBIT_PROBE:
        {probe_label: np.ndarray of h_{j→qubit}(τ)}.
        Keys must be a superset of noise_annotation keys.
        Supplied by the simulation engine after SI schematic analysis.
    n_samples : int
        Number of samples per realization — the length of the waveform
        actually used for convolution this run (real-axis mode may run at
        the schematic's native sample rate rather than the drive waveform's
        own, so this is passed explicitly rather than read off a
        SourceWaveform — see source/waveform.py's rf_waveform_at()).
    fs : float
        Sample rate (Hz) actually used for convolution this run — same
        reasoning as n_samples.
    mode : str
        'complex_baseband' or 'real_axis'.
    source_impedance_ohm : float
        Used for PSD calculations. Default 50 Ω.
    """

    def __init__(
        self,
        noise_annotation: dict[str, dict[str, Any]],
        transfer_functions_to_qubit: dict[str, np.ndarray],
        n_samples: int,
        fs: float,
        mode: str = "complex_baseband",
        source_impedance_ohm: float = 50.0,
    ) -> None:
        self._annotation = noise_annotation
        self._h_to_qubit = transfer_functions_to_qubit
        self._mode = mode
        self._R = float(source_impedance_ohm)

        self._N = n_samples
        self._fs = fs

        # Validate: every noise node must have a precomputed h_{j→qubit}
        missing = set(noise_annotation) - set(transfer_functions_to_qubit)
        if missing:
            raise KeyError(
                f"Transfer functions to QUBIT_PROBE missing for noise nodes: {missing}. "
                f"Ensure the SI schematic analysis computed h_{{j→qubit}} for all "
                f"annotated noise nodes before constructing NoisePropagator."
            )

        # Precompute PSD arrays for each node (flat over frequency — see psd.py)
        self._psd_cache: dict[str, np.ndarray] = {}
        freqs = np.fft.rfftfreq(self._N, d=1.0 / self._fs)
        for label, spec in noise_annotation.items():
            self._psd_cache[label] = psd_from_spec(spec, freqs, self._R)

    # ------------------------------------------------------------------
    # Core method: generate one noise realization at the qubit plane
    # ------------------------------------------------------------------

    def generate_realization(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate one summed noise realization at the qubit plane.

        For each noise node j:
          1. Draw noise v_noise_j(t) from S_v_j(f).
          2. Convolve with h_{j→qubit}(τ) → contribution at qubit plane.
        Sum all contributions.

        Parameters
        ----------
        rng : np.random.Generator
            Caller-supplied RNG for reproducibility across the ensemble.

        Returns
        -------
        v_noise_qubit : np.ndarray, shape (N,)
            dtype complex128 (baseband) or float64 (real_axis).
        """
        dtype = complex if self._mode == "complex_baseband" else float
        total = np.zeros(self._N, dtype=dtype)

        gen = (
            generate_baseband_noise
            if self._mode == "complex_baseband"
            else generate_rf_noise
        )

        for label, psd in self._psd_cache.items():
            h = self._h_to_qubit[label]
            v_noise_j = gen(self._N, self._fs, psd, rng=rng)
            propagated = fftconvolve(v_noise_j, h, mode="full")[: self._N]
            total += propagated.astype(dtype)

        return total

    # ------------------------------------------------------------------
    # Ensemble generation (convenience wrapper)
    # ------------------------------------------------------------------

    def generate_ensemble(
        self,
        n_realizations: int,
        seed: int | None = None,
    ) -> list[np.ndarray]:
        """
        Generate n_realizations independent noise realizations at the qubit plane.

        Returns
        -------
        list of np.ndarray, each shape (N,), one per realization.
        """
        rng = np.random.default_rng(seed)
        return [self.generate_realization(rng) for _ in range(n_realizations)]

    @property
    def n_noise_nodes(self) -> int:
        return len(self._annotation)

    @property
    def node_labels(self) -> list[str]:
        return list(self._annotation.keys())
