"""si_qfi.output — plotting and report stubs (Phase 3)."""

def plot_waveform(result, realization_idx=0):
    """Plot the qubit-plane waveform for one realization."""
    import matplotlib.pyplot as plt
    v = result.v_qubit_ensemble[realization_idx]
    t_ns = result.t_array * 1e9
    fig, axes = plt.subplots(2, 1, sharex=True)
    if result.mode == "complex_baseband":
        axes[0].plot(t_ns, np.real(v), label="I")
        axes[1].plot(t_ns, np.imag(v), label="Q", color="orange")
        axes[0].set_ylabel("I (V)")
        axes[1].set_ylabel("Q (V)")
    else:
        import numpy as np
        axes[0].plot(t_ns, v)
        axes[0].set_ylabel("v(t) (V)")
        axes[1].set_ylabel("")
    axes[-1].set_xlabel("Time (ns)")
    axes[0].set_title(f"Qubit-plane waveform (realization {realization_idx})")
    plt.tight_layout()
    plt.show()
