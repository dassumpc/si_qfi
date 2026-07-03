"""
si_qfi.nonlinear.base
=====================
Abstract base class for all nonlinear node models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
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
