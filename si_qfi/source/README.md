# si_qfi.source

Wraps the drive pulse: a complex baseband envelope plus a carrier
frequency, with helpers to resample/modulate it for whichever simulation
mode is in use.

- **`waveform.py`** — `SourceWaveform` (carrier + envelope, resampling and
  real-axis modulation via `rf_waveform_at()`/`resampled_envelope_at()`),
  `source_from_envelope_array()` (build one directly from a numpy array,
  the common case), and `build_gaussian_envelope()`/`build_drag_envelope()`
  (DRAG, suppresses leakage to higher transmon levels).

See `tests/test_source_waveform.py`. Used by every `examples/*.py` demo to
build the drive pulse before calling `siq.run()`.
