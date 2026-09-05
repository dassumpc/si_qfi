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

NoisePropagator itself only handles this mechanical realize+propagate+sum
step — it takes a precomputed {node: S_v(f)} PSD cache directly rather than
deriving PSDs itself (that responsibility now lives in
noise.psd.psd_cache_for_noise_nodes(), which sources each node's PSD from
SI's own statistical-noise-source computation, or a Python-side override —
see noise/psd.py's module docstring). This keeps "how do we know the PSD"
separate from "how do we turn a PSD into an actual noisy waveform".
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .realization import generate_baseband_noise, generate_rf_noise


class NoisePropagator:
    """
    Manages per-node noise generation and propagation to QUBIT_PROBE.

    Parameters
    ----------
    psd_cache : dict
        Precomputed {noise_source_name: S_v(f) [V²/Hz]} — see
        noise.psd.psd_cache_for_noise_nodes(). Each array must be evaluated
        on np.fft.rfftfreq(n_samples, d=1/fs) (matching this instance's own
        n_samples/fs) — the caller is responsible for that consistency.
    transfer_functions_to_qubit : dict
        Precomputed impulse responses from each noise node j to QUBIT_PROBE:
        {noise_source_name: np.ndarray of h_{j→qubit}(τ)}.
        Keys must be a superset of psd_cache keys.
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
    """

    def __init__(
        self,
        psd_cache: dict[str, np.ndarray],
        transfer_functions_to_qubit: dict[str, np.ndarray],
        n_samples: int,
        fs: float,
        mode: str = "complex_baseband",
    ) -> None:
        self._psd_cache = psd_cache
        self._h_to_qubit = transfer_functions_to_qubit
        self._mode = mode

        self._N = n_samples
        self._fs = fs

        # Validate: every noise node must have a precomputed h_{j→qubit}
        missing = set(psd_cache) - set(transfer_functions_to_qubit)
        if missing:
            raise KeyError(
                f"Transfer functions to QUBIT_PROBE missing for noise nodes: {missing}. "
                f"Ensure the SI schematic analysis computed h_{{j→qubit}} for all "
                f"annotated noise nodes before constructing NoisePropagator."
            )

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

        Noise is modeled as a stationary process present continuously at
        the source, convolved with h_j and evaluated over the window of
        interest [0, self._N) -- so each source draws self._N + len(h_j) - 1
        samples (extending "before" index 0 by h_j's own memory length) and
        uses `fftconvolve(..., mode="valid")`, giving exactly self._N output
        samples, EVERY one of which is a full-overlap convolution sum (not
        a partial/zero-padded edge). This matters because complex-baseband
        impulse responses are built via an fftshift (see
        schematic/transfer_function.py's _tf_to_baseband_impulse), which
        centers a near-delta response at index len(h_j)//2 rather than
        index 0 -- confirmed directly that drawing only self._N samples and
        using mode="full" (truncated or not) leaves most of the [0,self._N)
        window with ~zero overlap with h_j's shifted peak, understating the
        realized variance by roughly the fraction of the window that
        genuinely overlaps (observed: variance off by the same factor as
        the overlap fraction, not statistical noise).

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
            n_draw = self._N + len(h) - 1
            v_noise_j = gen(n_draw, self._fs, psd, rng=rng)
            propagated = fftconvolve(v_noise_j, h, mode="valid")
            assert len(propagated) == self._N   # by construction, see docstring
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
        return len(self._psd_cache)

    @property
    def node_labels(self) -> list[str]:
        return list(self._psd_cache.keys())
