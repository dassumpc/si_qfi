"""
SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin
"""
from setuptools import setup, find_packages

setup(
    name="si_qfi",
    version="0.1.0-dev",
    description="Bridges SignalIntegrity drive-chain simulation with QuTiP gate fidelity analysis",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    extras_require={
        "quantum": ["qutip>=5.0"],
        "qubits": ["scqubits>=4.0"],
        "si": ["SignalIntegrity"],
        "dev": ["pytest", "pytest-cov"],
    },
)
