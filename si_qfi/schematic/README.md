# si_qfi.schematic

Everything that talks to SignalIntegrity directly: loading a `.si` file,
extracting transfer functions from it, and reading a statistical-noise-
source device's own configured PSD. No sample rate, carrier, or waveform
involved at this layer — extraction is purely schematic-derived (PRD §3.3);
converting a raw transfer function into a time-domain impulse response at a
specific rate/mode happens later, once `engine.run()` has an actual
`SourceWaveform` to convolve.

- **`loader.py`** — `load_schematic()`: opens the project file, validates
  the source/qubit-probe labels exist, exposes the full probe-name list for
  annotation validation, and supports overriding the schematic's own
  `<Variables>` via SI's native `OpenProjectFile(args=...)` mechanism.
- **`transfer_function.py`** — extracts `H_k(f)` between adjacent probes
  (source-referenced ratio, since SI's headless API only exposes
  source-to-probe responses directly), and converts to a time-domain
  impulse response per mode (`compute_impulse_response()`) — real-axis
  reuses SI's own `FrequencyResponse.ImpulseResponse()` at the schematic's
  native sample rate; complex-baseband shifts+interpolates onto the
  envelope's own grid.
- **`noise.py`** — `get_noise_source_psd()` (SI's own native Johnson/shot/
  white PSD computation for a declared noise-source device) and
  `extract_noise_source_transfer_functions()` (that device's own transfer
  function to the qubit plane).

See `tests/test_schematic_hookup.py` for the real-SI-API integration
checks.
