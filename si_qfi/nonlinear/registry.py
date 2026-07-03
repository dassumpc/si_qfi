"""
si_qfi.nonlinear.registry
=========================
Parse the user-supplied nonlinear_nodes annotation dict and return a mapping
of probe label → NonlinearNode instance.

Supported model strings
-----------------------
  'saleh'            → SalehModel
  'amam_ampm'        → TabulatedAMAM
  'memory_polynomial'→ MemoryPolynomial
  'volterra'         → VolterraModel

All keys that are not recognised model parameters are passed through as kwargs
to the model constructor so that future models can add their own parameters
without changes here.

Gain convention check (PRD §3.6)
---------------------------------
The SI schematic is the sole source of a device's linear gain; nonlinear
node models should be normalized to unity small-signal gain (see
nonlinear/base.py). After building each node, build_nonlinear_nodes() checks
node.small_signal_gain and warns if it deviates from unity by more than
_SMALL_SIGNAL_GAIN_TOLERANCE_DB — a likely sign the device's real gain was
fit directly into the nonlinear model and will be double-counted if that
device is also represented in the schematic.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

from .base import NonlinearNode
from .saleh import SalehModel
from .amam_ampm import TabulatedAMAM
from .memory_polynomial import MemoryPolynomial
from .volterra import VolterraModel

_SMALL_SIGNAL_GAIN_TOLERANCE_DB = 3.0


def build_nonlinear_nodes(
    nonlinear_annotation: dict[str, dict[str, Any]],
    mode: str,
    warnings_list: Optional[list[str]] = None,
) -> dict[str, NonlinearNode]:
    """
    Parse the nonlinear_nodes annotation dict and instantiate model objects.

    Parameters
    ----------
    nonlinear_annotation : dict
        Keys are probe labels (e.g. 'NL_AMP1_OUT').
        Values are dicts containing at minimum a 'model' key.
    mode : str
        'complex_baseband' or 'real_axis'. Used to validate that the selected
        model supports the chosen simulation mode.
    warnings_list : list of str, optional
        If given, gain-convention warnings (see module docstring) are
        appended here instead of being emitted immediately via
        warnings.warn() — used by the engine to collect all diagnostics
        together. If None, warnings.warn() is called directly.

    Returns
    -------
    dict mapping probe_label → NonlinearNode instance.
    """
    nodes: dict[str, NonlinearNode] = {}
    for label, spec in nonlinear_annotation.items():
        if not label.startswith("NL_"):
            raise ValueError(
                f"Nonlinear node label '{label}' must start with 'NL_'."
            )
        spec = dict(spec)   # copy so we can pop 'model'
        model_name = spec.pop("model", None)
        if model_name is None:
            raise ValueError(f"Nonlinear node '{label}' has no 'model' key.")

        node = _build_model(label, model_name, spec, mode)
        _check_small_signal_gain(label, node, warnings_list)
        nodes[label] = node

    return nodes


def _check_small_signal_gain(
    label: str,
    node: NonlinearNode,
    warnings_list: Optional[list[str]],
) -> None:
    """
    Warn if a built node's small-signal gain deviates materially from unity.

    See "Gain convention check" in the module docstring.
    """
    gain = node.small_signal_gain
    if gain is None or gain <= 0:
        return
    gain_db = 20.0 * np.log10(gain)
    if abs(gain_db) > _SMALL_SIGNAL_GAIN_TOLERANCE_DB:
        msg = (
            f"Nonlinear node '{label}' has small-signal gain {gain_db:+.1f} dB "
            f"relative to unity (expected ~0 dB). Per the SI-QFI gain "
            f"convention (PRD §3.6), a device's linear gain should be "
            f"captured by the SI schematic (e.g. a small-signal S-parameter "
            f"block), and this nonlinear model should represent only the "
            f"amplitude-dependent deviation from that response. If this "
            f"device is also present in the schematic, its gain is likely "
            f"being applied twice."
        )
        if warnings_list is not None:
            warnings_list.append(msg)
        else:
            warnings.warn(f"SI-QFI: {msg}", stacklevel=3)


def _build_model(
    label: str,
    model_name: str,
    params: dict[str, Any],
    mode: str,
) -> NonlinearNode:
    """Instantiate a single NonlinearNode from its spec dict."""

    if model_name == "saleh":
        node = _build_saleh(params)
    elif model_name == "amam_ampm":
        node = _build_amam_ampm(params)
    elif model_name == "memory_polynomial":
        node = _build_memory_polynomial(params)
    elif model_name == "volterra":
        node = _build_volterra(params)
    else:
        raise ValueError(
            f"Unknown nonlinear model '{model_name}' for node '{label}'. "
            f"Supported: 'saleh', 'amam_ampm', 'memory_polynomial', 'volterra'."
        )

    # Validate mode compatibility
    if mode == "complex_baseband" and not node.supports_baseband:
        raise ValueError(
            f"Model '{model_name}' for node '{label}' does not support "
            f"complex_baseband mode. Use 'saleh', 'amam_ampm', or 'memory_polynomial'."
        )
    if mode == "real_axis" and not node.supports_real_axis:
        raise ValueError(
            f"Model '{model_name}' for node '{label}' does not support "
            f"real_axis mode. Use 'volterra'."
        )

    return node


def _build_saleh(p: dict) -> SalehModel:
    if "p1db_amplitude" in p and "ip3_amplitude" in p:
        return SalehModel.from_p1db_ip3(
            p1db_amplitude=p["p1db_amplitude"],
            ip3_amplitude=p["ip3_amplitude"],
            small_signal_gain=p.get("small_signal_gain", 1.0),
            enable_am_pm=p.get("enable_am_pm", False),
            am_pm_peak_deg=p.get("am_pm_peak_deg", 0.0),
        )
    return SalehModel(
        alpha_a=p["alpha_a"],
        beta_a=p["beta_a"],
        alpha_phi=p.get("alpha_phi", 0.0),
        beta_phi=p.get("beta_phi", 0.0),
    )


def _build_amam_ampm(p: dict) -> TabulatedAMAM:
    import numpy as np
    return TabulatedAMAM(
        amp_in=np.asarray(p["amam_curve"])[:, 0],
        amp_out=np.asarray(p["amam_curve"])[:, 1],
        phase_shift_rad=(
            np.asarray(p["ampm_curve"])[:, 1] if "ampm_curve" in p else None
        ),
        extrapolate=p.get("extrapolate", "clip"),
    )


def _build_memory_polynomial(p: dict) -> MemoryPolynomial:
    import numpy as np
    if "coefficients" in p:
        return MemoryPolynomial(
            coefficients=np.asarray(p["coefficients"], dtype=complex),
            orders=p.get("orders", [1, 3, 5]),
        )
    if "p1db_amplitude" in p and "ip3_amplitude" in p:
        return MemoryPolynomial.from_p1db_ip3(
            p1db_amplitude=p["p1db_amplitude"],
            ip3_amplitude=p["ip3_amplitude"],
            small_signal_gain=p.get("small_signal_gain", 1.0),
            memory_depth=p.get("memory_depth", 0),
            orders=p.get("orders", [1, 3, 5]),
        )
    raise ValueError(
        "memory_polynomial requires either 'coefficients' or "
        "'p1db_amplitude' + 'ip3_amplitude'."
    )


def _build_volterra(p: dict) -> VolterraModel:
    import numpy as np
    option = p.get("option", "describing")
    return VolterraModel(
        h1=np.asarray(p["h1"]) if "h1" in p else None,
        option=option,
        coefficients=np.asarray(p["coefficients"]) if "coefficients" in p else None,
        orders=p.get("orders", None),
        h3=np.asarray(p["h3"]) if "h3" in p else None,
        p1db_amplitude=p.get("p1db_amplitude"),
        ip3_amplitude=p.get("ip3_amplitude"),
        small_signal_gain=p.get("small_signal_gain", 1.0),
        memory_depth=p.get("memory_depth", 5),
    )
