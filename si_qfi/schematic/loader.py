"""
si_qfi.schematic.loader
=======================
Load and validate a SignalIntegrity schematic.

Node identity convention (PRD §3.2, revised)
---------------------------------------------
The schematic itself does not declare which probes are "nonlinear nodes" —
there is no `NL_` prefix auto-detection here. Nonlinear and noise nodes are
declared entirely by the `nonlinear` / `noise` dicts passed to `siq.run()`;
their keys must simply name probes that exist in the loaded schematic.
`validate_node_labels()` below is what the engine calls to check that, and
propagation order for nonlinear nodes is the `nonlinear` dict's key order
(the user is responsible for listing keys in signal-flow order — see PRD
§3.5). This module's only job is to load the schematic and expose the full
list of probe labels (`port_names`) so the engine can validate against it.

The qubit-plane probe and the drive source are also configurable labels
(`qubit_probe_label`, `source_label`, defaulting to 'VQubit' / 'VSource')
rather than hardcoded — `source_label` is both the reference point for every
relative transfer function computed in transfer_function.py and the point
where the drive waveform is injected.

SI API notes (verified against the installed SignalIntegrity source, not
guessed):
  - Devices live at `si_app.Drawing.schematic.deviceList`, populated as soon
    as OpenProjectFile() succeeds.
  - Device properties are accessed as `device['keyword'].GetValue()`
    (`device.__getitem__` returns None for a missing keyword, never raises).
  - Probes are devices whose `device['partname'].GetValue()` is one of
    _PROBE_PARTNAMES below (there is no 'VoltageProbe' partname in this SI
    version — a schematic VoltageProbe/Output device has partname 'Output').
  - The source is *not* identified by partname at all — it's identified by
    `device.netlist['DeviceName']` being one of _SOURCE_DEVICE_NAMES below
    (see SignalIntegrity/App/NetList.py's source-name collection logic).
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterable

# Device partname values that mark a probe (SI's VoltageProbe-equivalent).
_PROBE_PARTNAMES = (
    "Output",
    "DifferentialVoltageOutput",
    "CurrentOutput",
    "EyeProbe",
    "DifferentialEyeProbe",
    "EyeWaveform",
    "Waveform",
)

# netlist DeviceName values that count as a voltage/current source.
_SOURCE_DEVICE_NAMES = ("voltagesource", "currentsource", "networkanalyzerport")

# Device partname values that mark a statistical noise source (SI's own
# physically-grounded noise device library -- Johnson/shot/white/etc., see
# SignalIntegrity/App/Device.py's DeviceVoltageStatisticalNoiseSource /
# DeviceCurrentStatisticalNoiseSource). Deliberately excludes the 4-port
# differential-mode/common-mode variants and the '...Project' (hierarchical,
# noise-pulled-from-another-.si-file) variants -- both are out of scope for
# now (see schematic/noise.py's module docstring).
_NOISE_SOURCE_PARTNAMES = (
    "VoltageStatisticalNoiseSource",
    "CurrentStatisticalNoiseSource",
)

_DEFAULT_QUBIT_PROBE_LABEL = "VQubit"
_DEFAULT_SOURCE_LABEL = "VSource"


@dataclass
class SISchematic:
    """
    Loaded and validated SignalIntegrity schematic.

    Attributes
    ----------
    path : Path
        Path to the .si schematic file.
    si_app : Any
        The underlying SignalIntegrity headless app object
        (SignalIntegrityAppHeadless).
    qubit_probe_label : str
        Label of the probe marking the qubit plane (PRD §3.1). Configurable
        per schematic via load_schematic(qubit_probe_label=...).
    source_label : str
        ref of the schematic's voltage/current source device. Configurable
        per schematic via load_schematic(source_label=...). Used as the
        reference point for every relative transfer function (see
        transfer_function.py) and as the drive injection point.
    has_qubit_probe : bool
        True if qubit_probe_label is present (load_schematic() always
        guarantees this — it raises otherwise).
    port_names : list of str
        All probe labels in the schematic. This is the set that
        nonlinear annotation labels are validated against — see
        validate_node_labels().
    noise_source_names : list of str
        All statistical-noise-source device names (refs) in the schematic
        (see _NOISE_SOURCE_PARTNAMES) — a DIFFERENT namespace than
        port_names (these are voltagenoisesource/currentnoisesource
        devices, not Output-type probes). This is the set that `noise`
        annotation keys are validated against — see validate_node_labels().
    """
    path: Path
    si_app: Any
    qubit_probe_label: str = _DEFAULT_QUBIT_PROBE_LABEL
    source_label: str = _DEFAULT_SOURCE_LABEL
    has_qubit_probe: bool = False
    port_names: list[str] = field(default_factory=list)
    noise_source_names: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SISchematic('{self.path.name}', "
            f"probes={self.port_names}, source='{self.source_label}', "
            f"qubit_probe='{self.qubit_probe_label}')"
        )


def load_schematic(
    path: str | Path,
    qubit_probe_label: str = _DEFAULT_QUBIT_PROBE_LABEL,
    source_label: str = _DEFAULT_SOURCE_LABEL,
    variables: dict[str, Any] | None = None,
) -> SISchematic:
    """
    Load a SignalIntegrity schematic from disk and validate it for SI-QFI use.

    Parameters
    ----------
    path : str or Path
        Path to the .si schematic file. Relative paths are accepted and
        resolved to absolute internally -- SignalIntegrity's own
        OpenProjectFile() silently fails (returns False, no exception) on a
        relative path, so passing one straight through would raise a
        misleading "failed to open" error for a file that plainly exists.
    qubit_probe_label : str
        Label of the probe that marks the qubit plane. Default 'VQubit'.
    source_label : str
        ref of the schematic's voltage/current source device — the
        reference point for relative transfer functions and the drive
        injection point. Default 'VSource'.
    variables : dict, optional
        Overrides for the schematic's own named SI project variables (a
        schematic-level parametrization mechanism -- see PartProperty
        values of the form `=VariableName` in the .si XML, and the
        top-level `<Variables><Items><Variable>...` section that declares
        them, e.g. tests/test_schematic_impedance_mismatch.si). Passed
        straight through to `SignalIntegrityAppHeadless.OpenProjectFile()`'s
        own `args` parameter -- SI's OWN mechanism for this (verified
        directly against SignalIntegrity/App/SignalIntegrityAppHeadless.py:
        `OpenProjectFile(filename, args={})` calls `self.SetVariables(args,
        True)` right after reading the project, matching each key against
        the schematic's declared `<Variables>` names). Keys not matching
        any declared variable are silently ignored by SI itself (not an
        error here) -- SI's own SetVariables() only *prints* a warning for
        an unmatched key (reportMissing=True), it doesn't raise; this
        wrapper doesn't add stricter validation on top since a schematic's
        variable set isn't otherwise introspectable from outside SI's own
        Project object. Applied once, before any transfer function is
        extracted (extraction is cached per si_app -- see
        schematic/transfer_function.py's _get_transfer_parameters -- so
        this is the only safe time to set them; changing a variable on an
        already-extracted-from si_app would silently use the stale cache).
        None (default) behaves exactly as before this parameter existed.

    Returns
    -------
    SISchematic

    Raises
    ------
    FileNotFoundError : if the path does not exist.
    ValueError : if the file fails to open, or qubit_probe_label /
        source_label are not found (or source_label names a non-source
        device).
    ImportError : if SignalIntegrity is not installed.
    """
    # .resolve() is load-bearing, not cosmetic: SignalIntegrity's
    # OpenProjectFile() returns False (no exception) for a relative path,
    # which would surface below as a confusing "failed to open" on a file
    # that exists. See the `path` parameter's docstring.
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schematic not found: {path}")

    try:
        from SignalIntegrity.App.SignalIntegrityAppHeadless import (
            SignalIntegrityAppHeadless,
        )
    except ImportError as e:
        raise ImportError(
            "SignalIntegrity is required. "
            "Install from https://github.com/TeledyneLeCroy/SignalIntegrity"
        ) from e

    app = SignalIntegrityAppHeadless()
    if not app.OpenProjectFile(str(path), args=(variables or {})):
        raise ValueError(f"Failed to open SignalIntegrity project file: {path}")

    port_names = _extract_port_names(app)
    noise_source_names = _extract_noise_source_names(app)

    if qubit_probe_label not in port_names:
        raise ValueError(
            f"Schematic '{path.name}' must contain a probe labelled "
            f"'{qubit_probe_label}' (qubit_probe_label). Available probes: "
            f"{port_names}."
        )
    _check_voltage_source_present(app, source_label)

    return SISchematic(
        path=path,
        si_app=app,
        qubit_probe_label=qubit_probe_label,
        source_label=source_label,
        has_qubit_probe=True,
        port_names=port_names,
        noise_source_names=noise_source_names,
    )


def _extract_port_names(si_app: Any) -> list[str]:
    """Return all probe labels (Output-type devices) from the SI schematic."""
    names = []
    for device in si_app.Drawing.schematic.deviceList:
        partname_prop = device["partname"]
        if partname_prop is None:
            continue
        if partname_prop.GetValue() in _PROBE_PARTNAMES:
            ref_prop = device["ref"]
            if ref_prop is not None:
                names.append(ref_prop.GetValue())
    return names


def _extract_noise_source_names(si_app: Any) -> list[str]:
    """Return all statistical-noise-source device refs (see
    _NOISE_SOURCE_PARTNAMES) from the SI schematic."""
    names = []
    for device in si_app.Drawing.schematic.deviceList:
        partname_prop = device["partname"]
        if partname_prop is None:
            continue
        if partname_prop.GetValue() in _NOISE_SOURCE_PARTNAMES:
            ref_prop = device["ref"]
            if ref_prop is not None:
                names.append(ref_prop.GetValue())
    return names


def _check_voltage_source_present(si_app: Any, source_label: str) -> None:
    """
    Verify that a voltage/current source device with ref == source_label
    exists in the schematic.
    """
    for device in si_app.Drawing.schematic.deviceList:
        ref_prop = device["ref"]
        if ref_prop is None or ref_prop.GetValue() != source_label:
            continue
        device_name = device.netlist["DeviceName"]
        if device_name not in _SOURCE_DEVICE_NAMES:
            raise ValueError(
                f"Device '{source_label}' exists in the schematic but is not "
                f"a voltage/current source (found device type '{device_name}')."
            )
        return
    raise ValueError(
        f"Schematic must contain a source device with ref '{source_label}' "
        f"(source_label). No device with that ref was found."
    )


def validate_node_labels(
    schematic: SISchematic,
    labels: Iterable[str],
    kind: str = "node",
    known: Iterable[str] | None = None,
) -> None:
    """
    Verify that every label in `labels` names something present in the
    loaded schematic. Called by the engine to check `nonlinear` and `noise`
    annotation keys before using them (e.g. before building nonlinear node
    models or extracting transfer functions).

    Parameters
    ----------
    schematic : SISchematic
        Loaded schematic.
    labels : iterable of str
        Annotation keys to validate (e.g. nonlinear.keys() or noise.keys()).
    kind : str
        Human-readable label used in the error message, e.g. 'nonlinear'
        or 'noise' — identifies which annotation dict the bad label came from.
    known : iterable of str, optional
        The namespace to validate `labels` against. Defaults to
        schematic.port_names (the original, probe-label behavior) — pass
        schematic.noise_source_names for `noise` annotation keys, which name
        statistical-noise-source devices, a different namespace than probes.

    Raises
    ------
    ValueError
        If any label is not in `known`. Lists all missing labels and the
        full set of valid names.
    """
    known_set = set(known if known is not None else schematic.port_names)
    missing = [label for label in labels if label not in known_set]
    if missing:
        raise ValueError(
            f"{kind} node(s) {missing} not found in schematic "
            f"'{schematic.path.name}'. Available: {sorted(known_set)}."
        )
