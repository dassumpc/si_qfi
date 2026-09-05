# SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin

Bridges **SignalIntegrity** classical microwave drive-chain simulation with
**QuTiP** quantum gate fidelity analysis. You define a qubit's control line
(amplifiers, transmission lines, mismatches, noise sources) as an ordinary
SignalIntegrity `.si` schematic; SI-QFI extracts real transfer functions
from it, propagates a drive pulse through the chain (applying any
nonlinearity and noise you've annotated), and hands the resulting waveform
at the qubit plane to QuTiP to compute gate/state fidelity.

**Status:** the SI↔QuTiP bridge is fully implemented, tested, and has been
used to answer 10 physics questions about how real drive-chain impairments
(amplifier compression, dispersion, impedance mismatches, transmon
leakage, T1/T2, drive-line noise of several kinds, and LO phase noise) cost
gate fidelity — see `INVESTIGATIONS.md`.

## Which doc do I want?

- **This README** — what the package does, how to install it, a runnable
  example.
- **`SI_Quantum_Fidelity_Plugin_PRD.md`** — the full design specification:
  the two simulation modes' math, the nonlinearity models' derivations, the
  noise-injection architecture, the target API. Read this for *why* things
  are built the way they are.
- **`SI_QFI_Cursor_Handoff.md`** — implementation-status detail: exactly
  which SignalIntegrity/QuTiP API calls were used and how they were
  resolved against the installed library versions, plus a few real bugs
  found along the way (worth reading if you're extending the SI or QuTiP
  integration points specifically).
- **`INVESTIGATIONS.md`** — a running log of physics questions answered
  using this codebase, each backed by a runnable demo in `examples/`, a
  regression test, and a generated figure. Read this for worked examples
  of what the tool is actually for, and for several real bugs/gotchas
  (absolute noise-scale errors, calibration search failures, a QuTiP
  solver step-size trap) that are worth knowing about before you trust a
  number this codebase gives you.

---

## Package structure

```
si_qfi/
├── __init__.py                Top-level API: load_schematic, SourceWaveform, run, compare_modes
├── source/
│   └── waveform.py             SourceWaveform, DRAG/Gaussian envelope generators
├── nonlinear/
│   ├── base.py                 NonlinearNode abstract base class
│   ├── saleh.py                 SalehModel (baseband) + SalehRealAxisModel (real-axis)
│   ├── volterra.py              Volterra series (real-axis mode; describing/diagonal/full_kernel)
│   └── registry.py              Annotation dict -> NonlinearNode objects, small-signal-gain check
├── noise/
│   ├── psd.py                   PSD from flat/colored override, type=noise_figure/thermal/
│   │                             noise_density, and phase-noise PSD specs (dBc/Hz or raw)
│   ├── realization.py           Bandlimited Gaussian noise + phase-noise realization generation
│   └── propagation.py           NoisePropagator: per-node noise -> qubit plane
├── simulation/
│   └── engine.py                Two-pass NL + noise engine, phase-noise injection, compare_modes()
├── schematic/
│   ├── loader.py                SI schematic loading, incl. <Variables> overrides
│   ├── transfer_function.py     SI S-parameter -> transfer function extraction
│   └── noise.py                 SI statistical-noise-source PSD + transfer function extraction
├── quantum/
│   ├── models.py                QubitBase, QubitModel, Transmon, from_scqubits
│   ├── hamiltonian.py           build_hamiltonian(), demodulate()
│   ├── fidelity.py              gate_fidelity(), tuneup_amplitude(), FidelityResult
│   └── snr.py                   pulse_snr() -- effective SNR of a noisy simulation result
├── output/                      plot_waveform(), plot_nonlinearity() (Phase 3: full report/plots pending)
├── sweep/                       Parameter sweep / fidelity budget utilities (Phase 3, not yet built)
├── examples/                    Runnable investigation demos, one per INVESTIGATIONS.md section
├── notebooks/                   Standalone derivation notebooks (e.g. noise PSD scaling, zero si_qfi imports)
└── tests/                       196 tests: unit tests (numpy/scipy only) + SI/QuTiP-backed integration tests
```

---

## What's implemented

**Two simulation modes.** `complex_baseband` (default): propagates the
complex envelope at the pulse's own bandwidth — efficient, natural fit for
the QuTiP rotating frame, valid when the drive is narrowband relative to
the carrier. `real_axis`: propagates the full real RF waveform at the
schematic's own native sample rate — exact, tracks harmonics and
inter-harmonic mixing, needed when the narrowband assumption breaks down.
`compare_modes()` cross-validates the two directly. A "narrowband ratio"
diagnostic warns automatically when `complex_baseband` may be inaccurate
for a given pulse/carrier combination.

**Nonlinearity.** `SalehModel` (complex-baseband AM-AM/AM-PM) and
`SalehRealAxisModel` (the same bounded-rational curve applied directly to
the real waveform); `VolterraModel` (real-axis, three parameterizations:
`describing` from a single OP1dB/OIP3 point, `diagonal` from caller-supplied
coefficients, or a `full_kernel` 3-tap Volterra series). All models are
normalized to unity small-signal gain by convention (the SI schematic
supplies a device's actual linear gain) — `engine.run()` warns if a node's
measured small-signal gain deviates from 0dB by more than 3dB, a common
double-counted-gain mistake.

**Noise**, three distinct mechanisms:
- *Additive drive-line noise* (`noise={"NODE": {...}}`, keyed by SI
  statistical-noise-source device name): SI's own native Johnson/shot
  computation from the device's configured Type/Resistance/Temperature, or
  an override — a flat PSD number, a `dBm/Hz` value, a **colored/callable**
  PSD (`freqs -> S_v(freqs)`, e.g. a quasi-static or narrowband source), or
  a physically-parameterized spec (`type="thermal"`, `type="noise_figure"`,
  `type="noise_density"`). Propagated via that device's own SI-derived
  transfer function to the qubit plane, independent of any nonlinear
  segmentation.
- *LO/oscillator phase noise* (`phase_noise={...}`, one spec for the whole
  run): multiplicative (rides on the drive envelope itself), injected at
  the source *before* the nonlinear pass — the engine re-runs the
  nonlinear pass once per Monte Carlo realization when this is enabled,
  since a compressing amplifier genuinely responds differently to a
  phase-perturbed drive than to a clean one added-to afterward (see
  Investigation 10). Spec by raw PSD (`single_sided_psd_rad2_per_hz`,
  flat or colored) or by the standard oscillator-datasheet form
  (`dbc_hz`, a callable L(f) curve); `bandwidth_hz` is required (a real
  oscillator's phase-noise floor never rolls off to zero, so there's no
  physically honest default to inherit).
- *Intrinsic qubit decoherence* (`gate_fidelity(T1_us=, T2_us=)`): Lindblad
  collapse operators, independent of drive-chain noise, so the two
  contributions can be budgeted separately by comparing runs with and
  without each enabled.

**Quantum backend.** `gate_fidelity()` — average gate fidelity to a named
or custom target unitary, and/or state fidelity from a chosen initial
state, over a noise ensemble; returns `.noise_free` (one deterministic
solve) and `.noise` (the stochastic ensemble, `None` if no noise was
configured) separately, with propagators and final states available at no
extra solve cost. `tuneup_amplitude()` — searches a pulse's amplitude scale
to hit a target fidelity through the actual (possibly nonlinear) chain,
replacing hand-rolled calibration loops. `pulse_snr()` — effective SNR
(signal power / noise power, bandwidth-matched across modes) of a noisy
result. `Transmon` (analytic anharmonic-oscillator qubit model) and
`from_scqubits()` (unverified against a real scqubits install — see
`quantum/models.py`).

**Envelope generation.** `build_gaussian_envelope()`, `build_drag_envelope()`
(DRAG, suppresses leakage to higher transmon levels).

---

## Setup

`setup.py` lives at the repository root (one level above this `si_qfi/`
package directory — the repo root and the package directory share the same
name, which is easy to confuse). Install once, in editable mode, from the
repo root:

```bash
cd ..            # to the repo root, where setup.py lives (not this si_qfi/ folder)
pip install -e .
```

After that, `import si_qfi` works from any directory, in any script or
test run.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

196 tests total. Tests that need SignalIntegrity or QuTiP `pytest.
importorskip` themselves and are skipped automatically if those packages
aren't installed; the nonlinear/noise/PSD unit tests run with only
numpy/scipy.

---

## Quick usage

```python
import qutip
import si_qfi as siq
from si_qfi.source.waveform import build_gaussian_envelope, source_from_envelope_array

# 1. Load a schematic (the drive chain: source -> amplifier -> qubit line).
schematic = siq.load_schematic("tests/test_schematic_noise.si")

# 2. Build a Gaussian envelope and calibrate its amplitude to a true X gate
#    through the actual (possibly nonlinear) chain -- see quantum.tuneup_amplitude.
qubit = siq.quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)
eta = 2 * 3.14159265 * 10e6   # rad/(s*V), drive coupling strength
ref_shape = build_gaussian_envelope(duration_s=20e-9, sigma_s=20e-9 / 6, sample_rate_hz=4e9, amp=1.0)

tuned = siq.quantum.tuneup_amplitude(
    schematic, ref_shape, fs_envelope=4e9, carrier_ghz=5.0,
    qubit=qubit, coupling_strength_per_volt=eta, ideal_gate="X",
)

# 3. Run the full simulation with noise enabled -- here, a noise-figure spec
#    injected at VN1's own schematic location.
source = source_from_envelope_array(tuned.scale * ref_shape, fs=4e9, carrier_ghz=5.0)
result = siq.run(
    schematic=schematic, source=source, nonlinear=None,
    noise={"VN1": {"type": "noise_figure", "noise_figure_db": 3.0}},
    n_realizations=100, mode="complex_baseband", seed=42,
)

# 4. Gate fidelity: .noise_free (one deterministic solve) and .noise (the
#    stochastic ensemble, populated because `noise` above was non-empty).
fid = siq.quantum.gate_fidelity(result, qubit, coupling_strength_per_volt=eta, ideal_gate="X")
print(f"Noise-free F_avg: {fid.noise_free.F_avg:.5f}")
print(f"Ensemble F_avg:   {fid.noise.F_avg:.5f} +/- {fid.noise.F_sem:.6f}  (N={fid.noise.n_realizations})")

# 5. Effective SNR of the same result.
snr = siq.quantum.pulse_snr(result)
print(f"Effective SNR: {snr.snr:.3e} ({snr.snr_db:.1f} dB)")
```

`load_schematic()` also accepts a `variables={...}` dict to override any
`<Variables>` declared in the `.si` schematic file itself (see
`tests/test_schematic_impedance_mismatch.si` and Investigation 5 in
`INVESTIGATIONS.md`). Real-axis mode, more nonlinearity options, and phase
noise all follow the same `siq.run(...)` entry point — see `examples/` for
complete, runnable end-to-end demos of each, and `INVESTIGATIONS.md` for
the physics findings behind them.
