# si_qfi.noise

Everything noise: computing a PSD from a `noise=`/`phase_noise=` spec,
turning that PSD into an actual stochastic realization, and propagating
each realization to the qubit plane. Two physically distinct mechanisms
live here — see `simulation/engine.py`'s module docstring and
Investigation 10 in `INVESTIGATIONS.md` for why they're different:
*additive* drive-line noise (`noise=`, independent of the drive, generated
once per realization and summed onto the qubit-plane waveform) and
*multiplicative* LO phase noise (`phase_noise=`, proportional to the drive
envelope itself, injected at the source before the nonlinear pass).

- **`psd.py`** — `psd_from_override()` (flat, colored/callable, `dBm/Hz`,
  or `type="thermal"`/`"noise_figure"`/`"noise_density"` specs for additive
  noise), `psd_cache_for_noise_nodes()` (builds the whole `noise=` dict's
  PSDs for one `engine.run()` call), `phase_noise_psd_from_spec()` (raw
  `rad^2/Hz` or `dbc_hz` specs for phase noise, with a required
  `bandwidth_hz` — see its own docstring for why there's no default).
- **`realization.py`** — `generate_baseband_noise()`/`generate_rf_noise()`
  (bandlimited Gaussian realizations, mode-dependent absolute scale — see
  the module docstring's "Absolute-scale history" for two real bugs found
  and fixed here, and Investigation 8), `generate_phase_noise()` (a thin
  wrapper around `generate_rf_noise()` for the phase-noise case).
- **`propagation.py`** — `NoisePropagator`: per-node noise generation +
  propagation to the qubit plane via each node's own SI-derived transfer
  function, independent of nonlinear segmentation.

See `tests/test_noise.py` (unit tests, numpy/scipy only — includes the
`scipy.signal.periodogram`-verified absolute-scale checks) and
`tests/test_engine_noise.py` / `tests/test_engine_phase_noise.py`
(SI/QuTiP-backed integration tests).
