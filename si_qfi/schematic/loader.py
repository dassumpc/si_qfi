"""
si_qfi.schematic.loader
=======================
Load and validate a SignalIntegrity schematic.

# --- CURSOR NOTE (HIGH PRIORITY) ---
# This module is the primary SI API integration point.
# The SignalIntegrity schematic loading API needs to be verified against the
# actual SI repo. Common patterns in the SI codebase:
#
#   from SignalIntegrity.App.SignalIntegrityAppHeadless import SignalIntegrityAppHeadless
#   app = SignalIntegrityAppHeadless()
#   app.OpenProjectFile("myschematic.si")
#   # or:
#   from SignalIntegrity.Lib.SParameters.SParameterFile import SParameterFile
#
# For headless use (no GUI), the typical pattern is:
#   app = SignalIntegrityAppHeadless()
#   app.OpenProjectFile(path)
#   (sp, name) = app.SParameters()   # compute S-parameters from schematic
#
# Key things to verify:
#   1. Correct import path for the headless app
#   2. How to enumerate probes/devices in the schematic programmatically
#   3. How to get the list of port names / probe labels
#   4. How to extract a transfer function H(f) between two named ports
#   5. How to replace the voltage source waveform before running
# -------------------
"""

from __future__ import annotations

import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional


# Required probe labels in every SI-QFI schematic
_REQUIRED_OUTPUT_PROBE = "QUBIT_PROBE"
_NL_PROBE_PREFIX = "NL_"


@dataclass
class SISchematic:
    """
    Loaded and validated SignalIntegrity schematic.

    Attributes
    ----------
    path : Path
        Path to the .si schematic file.
    si_app : Any
        The underlying SignalIntegrity headless app object.
        Type: SignalIntegrityAppHeadless (not imported here to keep SI optional).
    nl_probe_labels : list of str
        All probe labels starting with 'NL_', in topological propagation order.
    has_qubit_probe : bool
        True if QUBIT_PROBE is present.
    port_names : list of str
        All named ports/probes in the schematic.
    """
    path: Path
    si_app: Any
    nl_probe_labels: list[str] = field(default_factory=list)
    has_qubit_probe: bool = False
    port_names: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SISchematic('{self.path.name}', "
            f"NL_nodes={self.nl_probe_labels}, "
            f"QUBIT_PROBE={'yes' if self.has_qubit_probe else 'NO — INVALID'})"
        )


def load_schematic(path: str | Path) -> SISchematic:
    """
    Load a SignalIntegrity schematic from disk and validate it for SI-QFI use.

    Parameters
    ----------
    path : str or Path
        Path to the .si schematic file.

    Returns
    -------
    SISchematic

    Raises
    ------
    FileNotFoundError : if the path does not exist.
    ValueError : if required probes are missing.
    ImportError : if SignalIntegrity is not installed.

    # --- CURSOR NOTE ---
    # Replace the stub body below with actual SI API calls.
    # Pattern to follow:
    #
    #   from SignalIntegrity.App.SignalIntegrityAppHeadless import (
    #       SignalIntegrityAppHeadless
    #   )
    #   app = SignalIntegrityAppHeadless()
    #   app.OpenProjectFile(str(path))
    #   port_names = _extract_port_names(app)
    #   nl_labels, ordered = _extract_nl_probes(app, port_names)
    #   _validate(port_names, nl_labels)
    #   return SISchematic(path=Path(path), si_app=app,
    #                      nl_probe_labels=ordered,
    #                      has_qubit_probe=True,
    #                      port_names=port_names)
    # -------------------
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schematic not found: {path}")

    try:
        # --- CURSOR NOTE: replace with correct SI import ---
        from SignalIntegrity.App.SignalIntegrityAppHeadless import (
            SignalIntegrityAppHeadless,
        )
    except ImportError as e:
        raise ImportError(
            "SignalIntegrity is required. "
            "Install from https://github.com/TeledyneLeCroy/SignalIntegrity"
        ) from e

    app = SignalIntegrityAppHeadless()
    app.OpenProjectFile(str(path))

    # --- CURSOR NOTE ---
    # Implement _extract_port_names to enumerate all voltage probe labels
    # from the schematic. The exact API depends on how SI stores device names.
    # Likely: iterate app.schematic.deviceList and check device type.
    # -------------------
    port_names = _extract_port_names(app)

    # Filter NL probes and sort in topological order
    nl_labels = [p for p in port_names if p.startswith(_NL_PROBE_PREFIX)]
    nl_ordered = _topological_sort_probes(app, nl_labels)

    # Validate required elements
    if _REQUIRED_OUTPUT_PROBE not in port_names:
        raise ValueError(
            f"Schematic '{path.name}' must contain a VoltageProbe "
            f"labelled '{_REQUIRED_OUTPUT_PROBE}'."
        )
    _check_voltage_source_present(app, port_names)

    return SISchematic(
        path=path,
        si_app=app,
        nl_probe_labels=nl_ordered,
        has_qubit_probe=True,
        port_names=port_names,
    )


def _extract_port_names(si_app: Any) -> list[str]:
    """
    Return all named probe/port labels from the SI schematic.

    # --- CURSOR NOTE ---
    # Inspect si_app.schematic or si_app.project to find device list.
    # Voltage probes are typically a specific device type in SI.
    # Example pattern (verify against SI source):
    #   labels = []
    #   for device in si_app.schematic.deviceList:
    #       if device.partname == 'VoltageProbe':
    #           labels.append(device.propertiesByName['ref'].value)
    #   return labels
    # -------------------
    """
    raise NotImplementedError(
        "_extract_port_names: implement using SignalIntegrity schematic API. "
        "See CURSOR NOTE in si_qfi/schematic/loader.py."
    )


def _check_voltage_source_present(si_app: Any, port_names: list[str]) -> None:
    """
    Verify that at least one VoltageSource device is present in the schematic.

    # --- CURSOR NOTE ---
    # Check si_app.schematic.deviceList for a VoltageSource device type.
    # Raise ValueError if none found.
    # -------------------
    """
    pass   # placeholder — remove when implemented


def _topological_sort_probes(si_app: Any, nl_labels: list[str]) -> list[str]:
    """
    Sort NL probe labels in signal-flow order from VoltageSource to QUBIT_PROBE.

    Returns the sorted list. Raises ValueError if ordering is ambiguous.

    # --- CURSOR NOTE ---
    # This requires tracing signal flow through the SI schematic graph.
    # If SI provides a netlist or graph structure, use it.
    # Otherwise, a simple heuristic: sort by the position of the probe
    # in the schematic (X coordinate) as a proxy for signal flow order.
    # For a properly drawn left-to-right schematic this works well.
    # More robust: trace from VoltageSource through connected nets to QUBIT_PROBE
    # and record the order in which NL probes are encountered.
    # -------------------
    """
    # Placeholder: return labels unsorted (user is responsible for naming order)
    warnings.warn(
        "NL probe topological sort not implemented; using annotation dict order. "
        "Ensure NL_ probes are listed in signal-flow order in the schematic.",
        stacklevel=3,
    )
    return nl_labels
