"""
si_qfi.schematic.noise
=======================
Extract statistical-noise-source content from a SignalIntegrity schematic:
per-source transfer functions (noise-source device -> qubit probe) and SI's
own computed noise spectral density for each declared source, via SI's
"statistical noise" feature (SignalIntegrity/App/Device.py's
DeviceVoltageStatisticalNoiseSource / DeviceCurrentStatisticalNoiseSource,
and the Johnson/shot/white-noise physics in
SignalIntegrity/App/StatisticalNoisePreferencesFile.py).

Why this calls si_app.Simulate(), and what that requires of the schematic
---------------------------------------------------------------------------
`si_app.TransferParameters()` (what transfer_function.py uses for the
signal path) is `Simulate(TransferMatricesOnly=True)` under the hood --
verified directly against SignalIntegrity/App/SignalIntegrityAppHeadless.py:
the `TransferMatricesOnly` branch explicitly does
`transferMatrices.Remove(outputs, sources, [], noiseSourceNames)`, i.e. it
DELETES every noise-source column before returning. So noise sources are
invisible to transfer_function.py's extraction path by design.

An earlier version of this module tried to route around that by
hand-reimplementing TransferParameters()'s internal call sequence (netlist
build, SimulatorNumericParser, ...) with just that one `.Remove()` step
omitted. That's fragile -- it duplicates SI's own private internal
machinery outside SI's maintained code path, and would silently drift out
of sync with any future change to Simulate()'s own implementation. This
module calls the REAL `si_app.Simulate()` instead (no special flags -- not
TransferMatricesOnly, not EyeDiagrams; confirmed headless, no Tk/GUI/eye-
diagram requirement by reading the method body), which keeps both the
noise-source columns AND automatically builds a StatisticalNoiseAnalysis
(`result['noise']`) with each source's own computed spectral density.

The real cost of this choice: unlike TransferParameters(), Simulate()
(without TransferMatricesOnly) also calls
`Drawing.schematic.InputWaveforms()`, which requires every source device
in the schematic to have a RESOLVABLE waveform (wftype != 'file' with an
empty/missing wffile, the convention every si_qfi schematic otherwise
uses, will raise). This is a real, deliberate requirement, not a bug to
work around: **any schematic that declares a statistical-noise-source
device must also give its drive source device a resolvable waveform**
(e.g. wftype='impulse' with amplitude 0 -- see
tests/test_schematic_noise.si's VSource for the pattern -- si_qfi never
reads or uses that waveform's actual content, since the engine always
injects its own drive waveform at run time; it only needs to exist so
Simulate() can complete). Schematics with no noise sources are entirely
unaffected -- this module, and therefore Simulate(), is only ever invoked
when `noise` is non-empty.

Simulate() is heavier than TransferParameters() (it also processes the
ideal signal waveform, which this module discards), so its Result is
cached on si_app exactly like transfer_function.py's
_get_transfer_parameters() caches TransferParameters() -- once per
(schematic, variables) combination, not per noise source or per
realization.

Name-based lookups, not manual index bookkeeping
---------------------------------------------------------------------------
A noise source is registered as an ordinary extra network INPUT (same
TransferMatrices machinery as the real signal source -- verified via
SignalIntegrity/Lib/Parsers/SimulatorParser.py's
`AddVoltageSource`/`AddCurrentSource` dispatch, which treats
'voltagenoisesource'/'currentnoisesource' netlist lines identically to
ordinary sources). So, unlike the signal path's `_extract_single_tf()`
(which needs the source-referenced-division trick because
TransferParameters() only exposes source→probe responses, never
probe→probe), a noise source's transfer function to the qubit probe is
available DIRECTLY via `Result.FrequencyResponse(source_name, qubit_label)`
(SignalIntegrity/App/Result.py), which resolves both names via `.index()`
against the SAME 'source names'/'output waveform labels' lists returned
alongside the transfer matrices -- safe regardless of any internal netlist
column-ordering quirks (SI's own commit history flags a real ordering bug
for 4-port differential/common-mode noise sources, whose netlist lines get
deferred to the end of the generated netlist text; name-based lookup
sidesteps it, and those 4-port variants are out of scope here anyway).

Scope: only 1-port and 2-port single-ended statistical noise sources
---------------------------------------------------------------------------
`_NOISE_SOURCE_PARTNAMES` in schematic/loader.py deliberately excludes the
4-port differential-mode/common-mode noise source variants (which get
netlist-rewritten into a 6-port inserter network plus a separately-declared
2-port source, per SignalIntegrity/App/NetList.py) and the '...Project'
variants (which pull SpectralDensity() from another .si project's own noise
output, via SignalIntegrityAppHeadless.ProjectNoise() -- a hierarchical-
noise-budget mechanism unrelated to what this codebase needs). Both are
straightforward to add later following the same pattern here, if needed.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from .transfer_function import TransferFunction


def _get_simulate_result(si_app: Any):
    """
    Call si_app.Simulate() once and cache the Result on si_app, mirroring
    transfer_function.py's _get_transfer_parameters() caching pattern (see
    module docstring for why Simulate() rather than TransferParameters(),
    and what it requires of the schematic's drive source).

    Raises
    ------
    RuntimeError
        If Simulate() fails -- e.g. equations invalid, the schematic
        doesn't have the VoltageSource + Output topology it requires, OR
        (the new requirement noise sources bring) the drive source device
        has no resolvable waveform (wftype='file' with an empty wffile,
        which every OTHER si_qfi schematic uses, will fail here -- give it
        a resolvable waveform, e.g. wftype='impulse' with amplitude 0; its
        content is never read, see module docstring).
    """
    cached = getattr(si_app, "_si_qfi_simulate_result", None)
    if cached is not None:
        return cached
    result = si_app.Simulate()
    if not result or result.get("transfer matrices") is None:
        raise RuntimeError(
            "SignalIntegrity Simulate() failed. This schematic declares a "
            "statistical-noise-source device, which requires Simulate() "
            "(not just TransferParameters()) -- check that the schematic "
            "has a VoltageSource and Output-type probes, no Port devices, "
            "no unresolved equation errors, AND that its drive source has "
            "a resolvable waveform (e.g. wftype='impulse', amplitude 0 -- "
            "see schematic.noise module docstring)."
        )
    si_app._si_qfi_simulate_result = result
    return result


def extract_noise_source_transfer_functions(
    schematic,
    noise_source_names: list[str],
) -> dict[str, TransferFunction]:
    """
    Extract raw (frequency-domain-only) transfer functions h_{source→qubit}
    for each named statistical-noise-source device.

    Noise sources propagate straight to the qubit plane, bypassing any
    nonlinear segmentation -- the same simplifying assumption
    noise/propagation.py's NoisePropagator already made for the old
    Python-spec noise model (see its module docstring): noise is linear, so
    its propagation is independent of the nonlinear chain, and is not
    modeled as passing through any intervening nonlinearity. This module
    doesn't change that assumption, just where the transfer function and
    spectral density come from.

    Parameters
    ----------
    schematic : SISchematic
    noise_source_names : list of str
        Statistical-noise-source device names -- validated by
        schematic.loader.validate_node_labels(schematic, names, kind="noise",
        known=schematic.noise_source_names) before this is called.

    Returns
    -------
    dict mapping noise_source_name → TransferFunction (raw; h=None, dt=None
    until compute_impulse_response() is called, same contract as the
    signal-path TransferFunction objects from transfer_function.py).
    """
    result = _get_simulate_result(schematic.si_app)
    qubit_label = schematic.qubit_probe_label

    tfs: dict[str, TransferFunction] = {}
    for name in noise_source_names:
        if name not in result["source names"]:
            raise ValueError(
                f"Noise source '{name}' not found in Simulate() result's "
                f"source list. Available sources: {result['source names']}."
            )
        fr = result.FrequencyResponse(name, qubit_label)
        tfs[name] = TransferFunction(
            label_in=name, label_out=qubit_label, si_frequency_response=fr,
        )
    return tfs


def get_noise_source_psd(
    schematic,
    noise_source_name: str,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    Return SI's own computed one-sided noise voltage PSD S_v(f) [V²/Hz] for
    a declared statistical-noise-source device (Johnson/shot/white/etc, per
    that device's own schematic-configured Type/Resistance/Temperature/...
    properties), resampled onto `freqs`.

    Comes from the same Simulate() call as
    extract_noise_source_transfer_functions() (result['noise'], a
    StatisticalNoiseAnalysis -- SignalIntegrity/App/StatisticalNoiseAnalysis.py
    / SignalIntegrity/Lib/Noise/NoiseAnalysis.py), so there's no extra
    Simulate() cost beyond what the transfer-function extraction already
    pays, and SI's own Enable/Lanes handling on the device is respected
    automatically (see StatisticalNoisePreferencesFile.py's
    VoltageNoiseConfiguration.SpectralDensity()) rather than re-derived here.

    SI's SpectralDensity stores AMPLITUDE spectral density (V/sqrt(Hz), not
    power) -- verified directly against
    SignalIntegrity/Lib/FrequencyDomain/SpectralDensity.py's `Values()`
    docstring/implementation. `.Resample(FrequencyList)` linearly
    interpolates that amplitude density onto a new frequency grid (points
    beyond SI's own solved bandwidth clamp to zero, per its docstring).
    Converting to si_qfi's psd_v2_per_hz convention (matching
    noise/realization.py's generate_baseband_noise/generate_rf_noise, which
    both expect V²/Hz) is then just squaring the resampled amplitude values.

    Parameters
    ----------
    schematic : SISchematic
    noise_source_name : str
        A statistical-noise-source device name.
    freqs : np.ndarray
        Target frequency grid (Hz) to resample SI's own computed density
        onto -- typically np.fft.rfftfreq(N, d=1/fs) for the convolution
        grid actually in use this run (see noise/propagation.py).

    Returns
    -------
    S_v : np.ndarray, float64, shape matching `freqs`
        One-sided noise voltage PSD [V²/Hz] at each frequency.
    """
    from SignalIntegrity.Lib.FrequencyDomain.FrequencyList import FrequencyList

    result = _get_simulate_result(schematic.si_app)
    noise_result = result.get("noise") or {}
    input_psd = noise_result.get("input_noise_spectral_density", {})
    if noise_source_name not in input_psd:
        raise ValueError(
            f"Noise source '{noise_source_name}' has no computed spectral "
            f"density in Simulate()'s noise result (is 'Enable' set on that "
            f"device in the schematic?). Available: {sorted(input_psd)}."
        )
    spectrum = input_psd[noise_source_name]["spectrum"]

    target_fd = FrequencyList(list(np.asarray(freqs, dtype=float)))
    resampled = spectrum.Resample(target_fd)
    amplitude_asd = np.asarray(resampled.Values("V/sqrt(Hz)"), dtype=float)
    return amplitude_asd ** 2
