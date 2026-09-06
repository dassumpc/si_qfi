# SI-QFI: Signal Integrity Quantum Fidelity Impact Plugin

Toolset to analyze the performance of drive-chains on quantum systems.

Specifically bridges **SignalIntegrity** classical microwave drive-chain simulation with
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
gate fidelity — see [`si_qfi/INVESTIGATIONS.md`](si_qfi/INVESTIGATIONS.md).

## Which doc do I want?

- **This README** — what the package does, how to install it, a runnable
  example.
- **[`si_qfi/SI_Quantum_Fidelity_Plugin_PRD.md`](si_qfi/SI_Quantum_Fidelity_Plugin_PRD.md)**
  — the design specification and derivations: the two simulation modes'
  math, the nonlinearity models' derivations, the noise-injection
  architecture. Read this for *why* things are built the way they are. It's
  the original forward-looking spec, so **where it and the code disagree the
  code is correct**; its own header says which parts are unbuilt and flags
  the one formula it gets wrong. The code cites its section numbers heavily
  (~44 docstring references), which is why the numbering is kept stable.
- **[`si_qfi/SI_QFI_Cursor_Handoff.md`](si_qfi/SI_QFI_Cursor_Handoff.md)** —
  implementation-status detail: exactly which SignalIntegrity/QuTiP API
  calls were used and how they were resolved against the installed library
  versions, plus a few real bugs found along the way (worth reading if
  you're extending the SI or QuTiP integration points specifically).
- **[`si_qfi/INVESTIGATIONS.md`](si_qfi/INVESTIGATIONS.md)** — a running log
  of physics questions answered using this codebase, each backed by a
  runnable demo in [`si_qfi/examples/`](si_qfi/examples/), a regression
  test, and a generated figure. Read this for worked examples of what the
  tool is actually for, and for several real bugs/gotchas (absolute
  noise-scale errors, calibration search failures, a QuTiP solver step-size
  trap) that are worth knowing about before you trust a number this
  codebase gives you.

Each subpackage also carries its own short `README.md` describing its files
— e.g. [`si_qfi/noise/`](si_qfi/noise/),
[`si_qfi/nonlinear/`](si_qfi/nonlinear/),
[`si_qfi/quantum/`](si_qfi/quantum/).

---

## Package structure

```
.                              (repo root: setup.py, pyproject.toml, LICENSE, this README)
└── si_qfi/                    the importable package -- note it shares the repo's name
├── __init__.py                Top-level API: load_schematic, SourceWaveform, run, compare_modes
├── source/
│   └── waveform.py             SourceWaveform, DRAG/Gaussian envelope generators
├── nonlinear/
│   ├── base.py                 NonlinearNode abstract base class
│   ├── saleh.py                 SalehModel (baseband) + SalehRealAxisModel (real-axis)
│   ├── volterra.py              Volterra series (real-axis mode; describing/diagonal/full_kernel)
│   ├── tabulated.py             TabulatedModel: generic AM-AM/AM-PM from a caller-supplied table
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
├── output/                      plot_waveform(), plot_nonlinearity() (full report generation pending)
├── examples/                    Runnable investigation demos, one per INVESTIGATIONS.md section
├── notebooks/                   Standalone derivation notebooks (e.g. noise PSD scaling, zero si_qfi imports)
├── demo_prompts/                Self-contained briefs for building signal-link design demos on top of si_qfi
└── tests/                       220 tests: unit tests (numpy/scipy only) + SI/QuTiP-backed integration tests
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
coefficients, or a `full_kernel` 3-tap Volterra series); `TabulatedModel`
(both modes) — a generic AM-AM/AM-PM nonlinearity interpolated directly from
a caller-supplied amplitude table, for devices whose response doesn't fit
either parametric form. All models are
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
[`si_qfi/quantum/models.py`](si_qfi/quantum/models.py)).

**Envelope generation.** `build_gaussian_envelope()`, `build_drag_envelope()`
(DRAG, suppresses leakage to higher transmon levels).

---

## Setup

Install in editable mode from the repo root (where `setup.py` and
`pyproject.toml` live). Note the repo root and the package directory share
the name `si_qfi`, which is easy to confuse — run this from the *outer* one:

```bash
pip install -e .
```

After that, `import si_qfi` works from any directory, in any script or
test run.

The two backends are **optional extras**, so a bare install pulls in only
numpy/scipy/matplotlib:

```bash
pip install -e ".[quantum]"      # QuTiP -- required for any fidelity calculation
pip install -e ".[si]"           # SignalIntegrity -- required to load .si schematics
pip install -e ".[quantum,si,dev]"   # everything, incl. pytest
```

> **Licensing note:** SI-QFI itself is MIT-licensed (see [`LICENSE`](LICENSE)).
> SignalIntegrity, its schematic backend, is **GPLv3+** and is *not* bundled
> or vendored here — it's an optional dependency you install separately. If
> you redistribute a combined work that includes both, the GPL's terms apply
> to that combined work.

## Running tests

```bash
pip install pytest
pytest si_qfi/tests/ -v
```

220 tests total. Tests that need SignalIntegrity or QuTiP `pytest.
importorskip` themselves and are skipped automatically if those packages
aren't installed; the nonlinear/noise/PSD unit tests run with only
numpy/scipy.

---

## Quick usage

```python
import qutip
import si_qfi as siq
from si_qfi.source.waveform import build_gaussian_envelope, source_from_envelope_array

FS = 450e6          # envelope sample rate
CARRIER_GHZ = 5.0   # keeps bandwidth/carrier at 0.045 -- inside the narrowband regime

# 1. Load a schematic (the drive chain: source -> amplifier -> qubit line).
schematic = siq.load_schematic("si_qfi/tests/test_schematic_noise.si")

# 2. Build a Gaussian envelope and calibrate its amplitude to a true X gate
#    through the actual (possibly nonlinear) chain -- see quantum.tuneup_amplitude.
qubit = siq.quantum.QubitModel(H0=0 * qutip.qeye(2), n_levels=2)
eta = 2 * 3.14159265 * 10e6   # rad/(s*V), drive coupling strength
ref_shape = build_gaussian_envelope(duration_s=40e-9, sigma_s=40e-9 / 6, sample_rate_hz=FS, amp=1.0)

tuned = siq.quantum.tuneup_amplitude(
    schematic, ref_shape, fs_envelope=FS, carrier_ghz=CARRIER_GHZ,
    qubit=qubit, coupling_strength_per_volt=eta, ideal_gate="X",
)

# 3. Run the full simulation with noise enabled, injected at VN1's own
#    schematic location. The density here is deliberately elevated so the
#    effect is visible in a quick-start; a realistic room-temperature
#    thermal/noise-figure spec on this chain (e.g.
#    {"type": "noise_figure", "noise_figure_db": 3.0}) sits ~6 orders of
#    magnitude lower and costs essentially no fidelity -- which is itself a
#    real result, just a dull demo. See INVESTIGATIONS.md #8 for the sweep.
source = source_from_envelope_array(tuned.scale * ref_shape, fs=FS, carrier_ghz=CARRIER_GHZ)
result = siq.run(
    schematic=schematic, source=source, nonlinear=None,
    noise={"VN1": {"single_sided_psd_v2_per_hz": 1e-12}},
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

Running it from the repo root prints:

```
Noise-free F_avg: 0.99996
Ensemble F_avg:   0.99991 +/- 0.000008  (N=100)
Effective SNR: 1.936e+04 (42.9 dB)
```

i.e. the chain itself is near-ideal, and the injected drive-line noise costs
about 5e-5 in average gate fidelity — roughly six times the ensemble's own
standard error, so it's a real effect rather than Monte Carlo scatter.

`load_schematic()` also accepts a `variables={...}` dict to override any
`<Variables>` declared in the `.si` schematic file itself (see
[`si_qfi/tests/test_schematic_impedance_mismatch.si`](si_qfi/tests/test_schematic_impedance_mismatch.si)
and Investigation 5 in [`si_qfi/INVESTIGATIONS.md`](si_qfi/INVESTIGATIONS.md)).
Real-axis mode, more nonlinearity options, and phase noise all follow the
same `siq.run(...)` entry point — see
[`si_qfi/examples/`](si_qfi/examples/) for complete, runnable end-to-end
demos of each, and `INVESTIGATIONS.md` for the physics findings behind them.

---

## License

MIT — see [`LICENSE`](LICENSE). Note that SignalIntegrity, the optional
schematic backend, is separately licensed under GPLv3+ and is not
distributed with this project; see the licensing note under
[Setup](#setup).
