"""
si_qfi.output — plotting helpers.

Two working utilities live here: plot_waveform() (the qubit-plane waveform
for one realization) and plot_nonlinearity() (a NonlinearNode's swept
input/output curve, for eyeballing a model's shape and compression before
committing to a full simulation). The richer full-report generation
described in the PRD is not built yet.
"""

def plot_waveform(result, realization_idx=0):
    """Plot the qubit-plane waveform for one realization."""
    import numpy as np
    import matplotlib.pyplot as plt
    v = result.v_qubit_ensemble[realization_idx]
    t_ns = np.arange(len(v)) / result.fs * 1e9
    fig, axes = plt.subplots(2, 1, sharex=True)
    if result.mode == "complex_baseband":
        axes[0].plot(t_ns, np.real(v), label="I")
        axes[1].plot(t_ns, np.imag(v), label="Q", color="orange")
        axes[0].set_ylabel("I (V)")
        axes[1].set_ylabel("Q (V)")
    else:
        axes[0].plot(t_ns, v)
        axes[0].set_ylabel("v(t) (V)")
        axes[1].set_ylabel("")
    axes[-1].set_xlabel("Time (ns)")
    axes[0].set_title(f"Qubit-plane waveform (realization {realization_idx})")
    plt.tight_layout()
    plt.show()


def plot_nonlinearity(node, mode, amplitude_max=None, n_points=400, settle_samples=None):
    """
    Sweep a NonlinearNode's input amplitude and plot its input/output curve
    -- a quick visual sanity check of a model's shape, compression, and
    overdrive behavior before running a full simulation.

    For mode='real_axis': one plot, output vs. input, swept over
    [-amplitude_max, +amplitude_max] (a real waveform can be either
    polarity), with the small-signal (linear) gain line overlaid for
    reference.
    For mode='complex_baseband': two subplots -- AM-AM (|output| vs input
    amplitude) and AM-PM (output phase, degrees, vs input amplitude),
    swept over [0, amplitude_max] (an envelope amplitude is non-negative
    by definition).

    Each sweep point is evaluated by applying the model to a constant-
    amplitude array long enough for any memory taps to settle
    (settle_samples), then reading off the last (steady-state) sample --
    this is what makes the utility work correctly for both memoryless
    models (SalehModel, SalehRealAxisModel) and models with real memory
    (Volterra 'full_kernel'/'diagonal' with m>0 taps) without the caller
    needing to know which kind `node` is. It does NOT
    attempt to show frequency-dependent (memory) behavior itself -- only
    the steady-state (DC-like) input/output curve.

    Parameters
    ----------
    node : NonlinearNode
        Any si_qfi.nonlinear model (SalehModel, SalehRealAxisModel,
        VolterraModel).
    mode : str
        'complex_baseband' or 'real_axis' -- must be a mode `node` actually
        supports (checked against node.supports_baseband/.supports_real_axis).
    amplitude_max : float, optional
        Sweep amplitude up to this value. Defaults to
        1.5 * node.max_monotonic_amplitude if that attribute exists and is
        finite (VolterraModel 'diagonal'/'describing' only), else 2.0.
    n_points : int
        Number of sweep points.
    settle_samples : int, optional
        Extra constant-input samples before reading the steady-state
        output. Defaults to node.memory_depth + 5.

    Returns
    -------
    matplotlib Figure, or None if matplotlib isn't installed (prints
    instead of raising, matching plot_waveform's convention).

    Note on units: amplitude_max and the plotted axes are all in the
    model's own INPUT-referred terms -- i.e. this plots the actual internal
    input/output mechanics of the fitted model, not the (output-referred)
    op1db_amplitude/oip3_amplitude a caller may have supplied when
    constructing it (see nonlinear/volterra.py's module docstring for the
    output-referred convention this codebase uses for those constructor
    arguments).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting.")
        return None
    import numpy as np

    if mode not in ("complex_baseband", "real_axis"):
        raise ValueError(f"mode must be 'complex_baseband' or 'real_axis', got {mode!r}.")
    if mode == "complex_baseband" and not node.supports_baseband:
        raise ValueError(f"{type(node).__name__} does not support complex_baseband mode.")
    if mode == "real_axis" and not node.supports_real_axis:
        raise ValueError(f"{type(node).__name__} does not support real_axis mode.")

    if amplitude_max is None:
        max_mono = getattr(node, "max_monotonic_amplitude", float("inf"))
        amplitude_max = 1.5 * max_mono if np.isfinite(max_mono) else 2.0
    if settle_samples is None:
        settle_samples = node.memory_depth + 5

    def steady_state(amp):
        dtype = complex if mode == "complex_baseband" else float
        x = np.full(settle_samples + 1, amp, dtype=dtype)
        y = node.apply_baseband(x) if mode == "complex_baseband" else node.apply_real_axis(x)
        return y[-1]

    gain0 = node.small_signal_gain
    if gain0 is None:
        gain0 = 1.0

    if mode == "real_axis":
        amps = np.linspace(-amplitude_max, amplitude_max, n_points)
        outputs = np.array([steady_state(a) for a in amps])

        fig, ax = plt.subplots()
        ax.plot(amps, outputs, label="model output")
        ax.plot(amps, amps * gain0, "--", color="gray", label="small-signal (linear) gain")
        ax.set_xlabel("Input amplitude")
        ax.set_ylabel("Output amplitude")
        ax.set_title(f"{type(node).__name__} input/output (real_axis)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # complex_baseband: AM-AM and AM-PM, amplitude >= 0 only.
    amps = np.linspace(0.0, amplitude_max, n_points)
    outputs = np.array([steady_state(a) for a in amps])
    amp_out = np.abs(outputs)
    phase_deg = np.degrees(np.angle(outputs))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(amps, amp_out, label="model output")
    ax1.plot(amps, amps * gain0, "--", color="gray", label="small-signal (linear) gain")
    ax1.set_ylabel("Output amplitude")
    ax1.set_title(f"{type(node).__name__} AM-AM / AM-PM (complex_baseband)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(amps, phase_deg)
    ax2.set_xlabel("Input amplitude")
    ax2.set_ylabel("Output phase (deg)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
