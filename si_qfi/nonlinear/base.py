"""
si_qfi.nonlinear.base
=====================
Abstract base class for all nonlinear node models.

Gain convention (PRD §3.6)
---------------------------
The SI schematic is the sole source of a device's linear (small-signal) gain,
phase, and frequency response — including active devices such as amplifiers,
which are represented in the schematic by their small-signal S-parameters.
Nonlinear node models therefore apply ONLY the amplitude-dependent deviation
from that already-captured linear response, and must be normalized so their
own small-signal gain is unity (G[A] → 1 as A → 0). Fitting a model directly
from a device's full measured response (gain included) double-counts that
gain if the same device is also present in the schematic. See
small_signal_gain below and nonlinear/registry.py for the runtime check.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class NonlinearNode(ABC):
    """
    Abstract base for nonlinear models applied at a designated probe node.

    Subclasses implement apply_baseband() (complex baseband mode) and/or
    apply_real_axis() (full real-axis mode).

    The memory_depth property returns 0 for memoryless models and M > 0
    for models that depend on past samples (memory polynomial, Volterra).
    """

    @property
    def memory_depth(self) -> int:
        """Number of past samples the model depends on (0 = memoryless)."""
        return 0

    @property
    def small_signal_gain(self) -> Optional[float]:
        """
        Linear (zero-amplitude) magnitude gain of this node, if defined.

        Per the SI-QFI gain convention (PRD §3.6), this should be ≈ 1.0 for
        every node: the SI schematic already supplies a device's linear gain,
        so this model should contribute only the amplitude-dependent
        deviation from it. nonlinear/registry.py warns if a built node's
        small_signal_gain deviates materially from 1.0.

        Returns None if not meaningful for this model (e.g. a full Volterra
        kernel with no single scalar gain).
        """
        return None

    @property
    def supports_baseband(self) -> bool:
        """True if this model can be used in complex baseband mode."""
        return False

    @property
    def supports_real_axis(self) -> bool:
        """True if this model can be used in real-axis mode."""
        return False

    def apply_baseband(self, u: np.ndarray) -> np.ndarray:
        """
        Apply nonlinearity to complex baseband envelope.

        Parameters
        ----------
        u : np.ndarray, complex128, shape (N,)
            Input complex envelope ũ(t).

        Returns
        -------
        np.ndarray, complex128, shape (N,)
            Output complex envelope after nonlinearity.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support complex baseband mode."
        )

    def apply_real_axis(self, v: np.ndarray) -> np.ndarray:
        """
        Apply nonlinearity to full real RF waveform.

        Parameters
        ----------
        v : np.ndarray, float64, shape (N,)
            Input real RF waveform v(t).

        Returns
        -------
        np.ndarray, float64, shape (N,)
            Output real RF waveform after nonlinearity.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support real-axis mode."
        )
