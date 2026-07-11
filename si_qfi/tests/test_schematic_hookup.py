"""
Tests for the SignalIntegrity hookup in si_qfi.schematic.loader and
si_qfi.schematic.transfer_function, run against tests/test_schematic_basic.si.

test_schematic_basic.si topology:

    VSource (DeviceVoltageSource) -> R1 (50 ohm) -> T1 (1ns/50 ohm line)
        -> D1 (DeviceVoltageAmplifierTwoPort, gain=10.0, zi=zo=50 ohm)
        -> ViDriver probe (D1's input node)
        -> DriverOutput probe (D1's output node, pre-T2)
        -> T2 (1ns/50 ohm line) -> R2 (50 ohm, shunt/matched termination)
        -> VQubit probe (post-T2, at the R2 node)

Everything is impedance-matched at 50 ohm (source, lines, amp zi/zo,
termination), so both transmission lines are lossless and reflection-free:
they add pure delay, never attenuation. Expected transfer functions (flat
across frequency, in linear voltage units):

    VSource -> VQubit        : 0.5 (R1/D1-input divider) * 10.0 (D1 gain)
                                * 0.5 (D1-output/R2 divider) = 2.5
    DriverOutput -> VQubit   : 1.0 (T2 is matched at both ends -- no
                                additional attenuation, only ~1ns delay)

These numbers were confirmed independently against SignalIntegrity's
VoltageAmplifierTwoPort S-parameter model (SignalIntegrity/Lib/Devices/
VoltageAmplifier.py): S21 = 2*G*Zi*Z0/((Zi+Z0)*(Zo+Z0)) = G/2 = 5 when
Zi=Zo=Z0=50, i.e. DriverOutput/ViDriver = 5, and ViDriver/VSource = 0.5
(matched divider), giving DriverOutput/VSource = 2.5 directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")

from si_qfi.schematic import loader as si_loader
from si_qfi.schematic import transfer_function as si_tf
from si_qfi.source.waveform import SourceWaveform

SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_basic.si").resolve()

_EXPECTED_PROBES = {"ViDriver", "DriverOutput", "VQubit"}
_MAG_TOL = 1e-3   # relative tolerance for the flat-magnitude checks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def si_app():
    """Raw SignalIntegrityAppHeadless with test_schematic_basic.si loaded."""
    from SignalIntegrity.App.SignalIntegrityAppHeadless import SignalIntegrityAppHeadless
    app = SignalIntegrityAppHeadless()
    opened = app.OpenProjectFile(str(SCHEMATIC_PATH))
    assert opened, f"OpenProjectFile failed for {SCHEMATIC_PATH}"
    return app


@pytest.fixture
def schematic():
    """SISchematic via loader.load_schematic(), using the default labels
    ('VQubit', 'VSource') which match this fixture file."""
    return si_loader.load_schematic(SCHEMATIC_PATH)


@pytest.fixture
def si_envelope():
    """A minimal real SignalIntegrity Waveform to use as a SourceWaveform envelope."""
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    fs = 40e9   # matches UserSampleRate in test_schematic_basic.si
    n = 256
    t = np.arange(n) / fs
    values = np.exp(-((t - t[-1] / 2) ** 2) / (2 * (t[-1] / 6) ** 2)).astype(complex)
    return Waveform(TimeDescriptor(0.0, n, fs), list(values))


@pytest.fixture
def source_waveform(si_envelope):
    return SourceWaveform(carrier_freq_ghz=5.0, envelope=si_envelope)


@pytest.fixture
def low_fs_source_waveform():
    """
    A SourceWaveform whose envelope fs (1 GSa/s) is far below the
    schematic's own 40 GSa/s -- exercises complex-baseband mode's
    fs-independent frequency-resolution path (compute_impulse_response()
    never needs the schematic's own rate for baseband mode, only the
    envelope's fs/carrier_hz).
    """
    from SignalIntegrity.Lib.TimeDomain.Waveform.Waveform import Waveform
    from SignalIntegrity.Lib.TimeDomain.Waveform.TimeDescriptor import TimeDescriptor

    fs = 1e9   # 40x below the schematic's 40 GSa/s
    n = 10     # 10 samples at 1 GSa/s -> 9ns span, a "long" waveform in dt terms
    t = np.arange(n) / fs
    values = np.exp(-((t - t[-1] / 2) ** 2) / (2 * (t[-1] / 6) ** 2)).astype(complex)
    envelope = Waveform(TimeDescriptor(0.0, n, fs), list(values))
    return SourceWaveform(carrier_freq_ghz=5.0, envelope=envelope)


# ---------------------------------------------------------------------------
# schematic/loader.py
# ---------------------------------------------------------------------------

def test_load_schematic_basic(schematic):
    """load_schematic() with default labels should match this fixture file."""
    assert set(schematic.port_names) == _EXPECTED_PROBES
    assert schematic.has_qubit_probe is True
    assert schematic.qubit_probe_label == "VQubit"
    assert schematic.source_label == "VSource"


def test_load_schematic_custom_labels():
    """qubit_probe_label / source_label are overridable per schematic."""
    result = si_loader.load_schematic(
        SCHEMATIC_PATH, qubit_probe_label="VQubit", source_label="VSource",
    )
    assert result.qubit_probe_label == "VQubit"
    assert result.source_label == "VSource"

    with pytest.raises(ValueError, match="NOT_A_REAL_PROBE"):
        si_loader.load_schematic(SCHEMATIC_PATH, qubit_probe_label="NOT_A_REAL_PROBE")

    with pytest.raises(ValueError, match="NOT_A_REAL_SOURCE"):
        si_loader.load_schematic(SCHEMATIC_PATH, source_label="NOT_A_REAL_SOURCE")


def test_extract_port_names(si_app):
    names = si_loader._extract_port_names(si_app)
    assert set(names) == _EXPECTED_PROBES


def test_check_voltage_source_present(si_app):
    """Should not raise for 'VSource' (the real source); should raise otherwise."""
    si_loader._check_voltage_source_present(si_app, "VSource")

    with pytest.raises(ValueError):
        si_loader._check_voltage_source_present(si_app, "NotASource")

    # A device that exists but isn't a source (e.g. a probe) should also raise.
    with pytest.raises(ValueError):
        si_loader._check_voltage_source_present(si_app, "VQubit")


# ---------------------------------------------------------------------------
# schematic/transfer_function.py
# ---------------------------------------------------------------------------

def test_extract_single_tf_source_to_qubit(si_app):
    """
    Full VSource -> VQubit transfer function: flat 2.5 (linear, real).
    Extraction is waveform-agnostic -- no fs/mode/carrier_hz needed at all.
    """
    tf = si_tf._extract_single_tf(si_app, "VSource", "VQubit", "VSource")
    assert tf.H.shape == tf.freqs.shape
    assert tf.h is None and tf.dt is None   # not computed until compute_impulse_response()
    mag = np.abs(tf.H)
    assert np.allclose(mag, 2.5, rtol=_MAG_TOL), f"expected flat 2.5, got {mag}"
    # H is not phase-flat: two matched 1ns lines (T1, T2) contribute a pure
    # linear delay, which rotates Re(H)/Im(H) through all four quadrants as
    # frequency increases -- that's expected and doesn't contradict "flat
    # magnitude, no reactive elements changing the gain itself".
    assert np.isclose(tf.H[0], 2.5, atol=1e-6)   # at DC, phase delay = 0


def test_extract_single_tf_driveroutput_to_qubit(si_app):
    """DriverOutput -> VQubit: flat unity (T2 is matched at both ends)."""
    tf = si_tf._extract_single_tf(si_app, "DriverOutput", "VQubit", "VSource")
    mag = np.abs(tf.H)
    assert np.allclose(mag, 1.0, rtol=_MAG_TOL), f"expected flat unity, got {mag}"


def test_extract_all_transfer_functions_no_nl_nodes(schematic):
    """Full segmentation with zero NL nodes: VSource -> VQubit directly."""
    tfs = si_tf.extract_all_transfer_functions(schematic, nl_labels=[])
    assert ("VSource", "VQubit") in tfs
    mag = np.abs(tfs[("VSource", "VQubit")].H)
    assert np.allclose(mag, 2.5, rtol=_MAG_TOL)


def test_extract_all_transfer_functions_with_nl_node(schematic):
    """
    Full segmentation with DriverOutput as an NL cut point: VSource ->
    DriverOutput -> VQubit. The two segments' magnitudes should multiply
    back to the full 2.5 (0.5 * 10 * 0.5), matching the direct-path test
    above and confirming the ratio-based segment math is self-consistent.
    """
    tfs = si_tf.extract_all_transfer_functions(schematic, nl_labels=["DriverOutput"])
    assert ("VSource", "DriverOutput") in tfs
    assert ("DriverOutput", "VQubit") in tfs

    mag_source_to_driver = np.abs(tfs[("VSource", "DriverOutput")].H)
    mag_driver_to_qubit = np.abs(tfs[("DriverOutput", "VQubit")].H)

    assert np.allclose(mag_driver_to_qubit, 1.0, rtol=_MAG_TOL)
    assert np.allclose(mag_source_to_driver * mag_driver_to_qubit, 2.5, rtol=_MAG_TOL)


def test_compute_impulse_response_baseband(si_app, source_waveform):
    """
    compute_impulse_response() is the deferred step that needs fs/mode/
    carrier_hz -- exercised here at baseband, using the envelope's own fs.
    """
    raw = si_tf._extract_single_tf(si_app, "VSource", "VQubit", "VSource")
    tf = si_tf.compute_impulse_response(
        raw, "complex_baseband", fs=source_waveform.fs, carrier_hz=source_waveform.carrier_freq_hz
    )
    assert tf.h is not None and tf.dt is not None
    assert np.isclose(tf.dt, 1.0 / source_waveform.fs)
    # freqs/H are derived from si_frequency_response (a property, not a
    # stored field -- see TransferFunction's docstring), so the meaningful
    # invariant is that the *same* underlying FrequencyResponse carries
    # through unchanged; np.array(...) allocates fresh each access, so
    # tf.H is raw.H would be False even though nothing changed.
    assert tf.si_frequency_response is raw.si_frequency_response
    assert np.array_equal(tf.freqs, raw.freqs)
    assert np.array_equal(tf.H, raw.H)
    # Discrete-convolution-kernel convention (same as real-axis mode):
    # sum(h) == H at the carrier frequency, no extra dt/fs scaling. H is
    # flat 2.5 everywhere on this schematic, so this holds regardless of
    # carrier placement. This also catches the fs-scale/ifftshift-ordering
    # class of bug (a spurious extra factor of fs previously inflated |h|
    # by ~1e10).
    assert np.isclose(np.sum(tf.h), 2.5, rtol=1e-2)
    assert np.ptp(np.abs(tf.h)) < 10.0


def test_compute_impulse_response_baseband_low_fs(si_app, low_fs_source_waveform):
    """
    Baseband mode with an envelope fs (1 GSa/s) far below the schematic's
    own 40 GSa/s. Baseband mode's impulse-response length/duration is set
    by the *schematic's* frequency resolution (df = EndFrequency /
    FrequencyPoints = 1e7 Hz here), not by fs -- so a lower fs means fewer,
    more widely-spaced samples covering the same total duration (1/df),
    not a shorter or malformed impulse response. This is the same
    fs-scale/ifftshift-ordering code path exercised by
    test_compute_impulse_response_baseband, just at a very different fs to
    make sure nothing about that fix was fs-magnitude-dependent.
    """
    raw = si_tf._extract_single_tf(si_app, "VSource", "VQubit", "VSource")
    tf = si_tf.compute_impulse_response(
        raw, "complex_baseband",
        fs=low_fs_source_waveform.fs, carrier_hz=low_fs_source_waveform.carrier_freq_hz,
    )
    assert tf.h is not None and tf.dt is not None
    assert np.isclose(tf.dt, 1.0 / low_fs_source_waveform.fs)

    # Impulse-response length is fs / df_known (df_known = schematic's own
    # frequency-sweep spacing), independent of the envelope's own n_samples.
    df_known = raw.freqs[1] - raw.freqs[0]
    expected_len = int(round(low_fs_source_waveform.fs / df_known))
    assert len(tf.h) == expected_len
    assert expected_len < len(raw.freqs)   # far fewer taps than the 40 GSa/s case

    assert tf.h.dtype == np.complex128
    # Same gain invariant as the 40 GSa/s case: sum(h) == H(f=0-equivalent)
    # == 2.5 (flat gain on this schematic), no extra dt/fs scaling -- proves
    # the low-fs path isn't silently reintroducing the fs-scale blowup bug.
    assert np.isclose(np.sum(tf.h), 2.5, rtol=1e-2)
    assert np.ptp(np.abs(tf.h)) < 10.0

    # fftconvolve(v, h) as engine.py does it (full convolution, no
    # truncation -- see engine.py's _nonlinear_pass): h is fftshifted to
    # match SI's own centered convention (see _tf_to_baseband_impulse), so
    # its peak sits near the middle of the array, not near index 0 --
    # comparing peak magnitude over the *entire* untruncated output
    # (delay/position-invariant) confirms the gain survives the convolution
    # intact regardless of where in the array that peak lands.
    from scipy.signal import fftconvolve
    v_in = low_fs_source_waveform.envelope_complex
    v_out = fftconvolve(v_in, tf.h, mode="full")
    assert np.isclose(np.max(np.abs(v_out)), 2.5 * np.max(np.abs(v_in)), rtol=1e-2)


def test_compute_impulse_response_real_axis_uses_native_si(si_app):
    """
    Real-axis mode should reuse SI's own FrequencyResponse.ImpulseResponse()
    (via tf.si_frequency_response) with no fs/carrier_hz at all -- neither is
    required (or used) for this mode. Sanity-check the result against the
    known flat 2.5 gain: the impulse response's discrete-DFT sum should
    equal H(f=0) = 2.5 for this schematic.
    """
    raw = si_tf._extract_single_tf(si_app, "VSource", "VQubit", "VSource")
    assert raw.si_frequency_response is not None

    tf = si_tf.compute_impulse_response(raw, "real_axis")
    fs_native = si_tf.native_sample_rate(raw)
    assert tf.h is not None and tf.dt is not None
    assert np.isclose(tf.dt, 1.0 / fs_native)
    assert tf.h.dtype == np.float64
    # Discrete DFT sum of the impulse response == H(f=0) = DC gain = 2.5
    # for this schematic (no dt factor -- that's the continuous-FT form).
    assert np.isclose(np.sum(tf.h), 2.5, rtol=1e-2)


def test_compute_impulse_response_real_axis_ignores_provided_fs(si_app):
    """
    Real-axis mode must ignore a caller-supplied fs entirely -- it should
    always derive its own native rate from si_frequency_response, not the
    fs argument, even if one happens to be passed (e.g. by a caller that
    unconditionally forwards fs for both modes, as engine.py does).
    """
    raw = si_tf._extract_single_tf(si_app, "VSource", "VQubit", "VSource")
    fs_native = si_tf.native_sample_rate(raw)

    tf_no_fs = si_tf.compute_impulse_response(raw, "real_axis")
    tf_wrong_fs = si_tf.compute_impulse_response(raw, "real_axis", fs=fs_native * 3.7, carrier_hz=1e9)

    assert np.isclose(tf_no_fs.dt, tf_wrong_fs.dt)
    assert np.array_equal(tf_no_fs.h, tf_wrong_fs.h)


def test_compute_impulse_response_baseband_requires_fs_and_carrier(si_app):
    """complex_baseband mode must raise if fs or carrier_hz is missing."""
    raw = si_tf._extract_single_tf(si_app, "VSource", "VQubit", "VSource")
    with pytest.raises(ValueError, match="requires both fs and carrier_hz"):
        si_tf.compute_impulse_response(raw, "complex_baseband")
    with pytest.raises(ValueError, match="requires both fs and carrier_hz"):
        si_tf.compute_impulse_response(raw, "complex_baseband", fs=1e9)
    with pytest.raises(ValueError, match="requires both fs and carrier_hz"):
        si_tf.compute_impulse_response(raw, "complex_baseband", carrier_hz=5e9)


def test_native_sample_rate(si_app):
    """
    native_sample_rate() should match test_schematic_basic.si's own
    UserSampleRate (40 GSa/s) -- EndFrequency=20GHz, so 2x that is 40 GSa/s.
    """
    raw = si_tf._extract_single_tf(si_app, "VSource", "VQubit", "VSource")
    assert np.isclose(si_tf.native_sample_rate(raw), 40e9, rtol=1e-6)


# ---------------------------------------------------------------------------
# test_schematic_lossy_T_line.si -- same topology as test_schematic_basic.si
# (VSource -> R1 -> T3 -> D1(gain=30) -> ViDriver/DriverOutput -> T1 -> R2 ->
# VQubit), but T1/T3 are DeviceTransmissionLineLossy (1 dB/Hz/s, td=1ns,
# zc=50 ohm) instead of lossless lines. Impedances are still matched at
# every junction, so the DC (f=0) gains match the basic schematic's flat
# values scaled for gain=30 instead of gain=10:
#
#     VSource -> VQubit      : 0.5 * 30.0 * 0.5 = 7.5 at DC
#     DriverOutput -> VQubit : 1.0 at DC (T1 matched at both ends)
#
# but magnitude now *rolls off* with frequency because the lines are lossy,
# unlike the flat-magnitude basic schematic -- confirmed above by probing
# |H| at DC vs. the top of the frequency sweep.
# ---------------------------------------------------------------------------

LOSSY_SCHEMATIC_PATH = (Path(__file__).parent / "test_schematic_lossy_T_line.si").resolve()


@pytest.fixture
def lossy_si_app():
    """Raw SignalIntegrityAppHeadless with test_schematic_lossy_T_line.si loaded."""
    from SignalIntegrity.App.SignalIntegrityAppHeadless import SignalIntegrityAppHeadless
    app = SignalIntegrityAppHeadless()
    opened = app.OpenProjectFile(str(LOSSY_SCHEMATIC_PATH))
    assert opened, f"OpenProjectFile failed for {LOSSY_SCHEMATIC_PATH}"
    return app


def test_compute_impulse_response_real_axis_lossy_line(lossy_si_app):
    """
    Real-axis impulse response on the lossy-line schematic: still derives
    its native rate from si_frequency_response with no fs/carrier_hz
    supplied, and the discrete-DFT sum of h should equal H(f=0) = 7.5 (the
    DC gain), even though |H| is not flat across frequency here.
    """
    raw = si_tf._extract_single_tf(lossy_si_app, "VSource", "VQubit", "VSource")
    assert np.isclose(raw.H[0], 7.5, atol=1e-6)
    mag = np.abs(raw.H)
    assert mag[-1] < mag[0]   # lossy line: high frequencies attenuated more than DC

    tf = si_tf.compute_impulse_response(raw, "real_axis")
    fs_native = si_tf.native_sample_rate(raw)
    assert np.isclose(fs_native, 40e9, rtol=1e-6)
    assert tf.h is not None and tf.dt is not None
    assert np.isclose(tf.dt, 1.0 / fs_native)
    assert tf.h.dtype == np.float64
    assert np.isclose(np.sum(tf.h), 7.5, rtol=1e-2)


def test_compute_impulse_response_baseband_lossy_line(lossy_si_app, source_waveform):
    """
    Baseband impulse response on the lossy-line schematic, using the
    envelope's own fs/carrier_hz -- mirrors
    test_compute_impulse_response_baseband but against the lossy fixture.
    """
    raw = si_tf._extract_single_tf(lossy_si_app, "VSource", "VQubit", "VSource")
    tf = si_tf.compute_impulse_response(
        raw, "complex_baseband", fs=source_waveform.fs, carrier_hz=source_waveform.carrier_freq_hz
    )
    assert tf.h is not None and tf.dt is not None
    assert np.isclose(tf.dt, 1.0 / source_waveform.fs)
    assert tf.h.dtype == np.complex128
    assert tf.si_frequency_response is raw.si_frequency_response
    # sum(h) == H(carrier_hz) (discrete-convolution-kernel convention, no
    # extra dt/fs scaling -- see test_compute_impulse_response_baseband).
    h_at_carrier = raw.H[np.argmin(np.abs(raw.freqs - source_waveform.carrier_freq_hz))]
    assert np.isclose(np.sum(tf.h), h_at_carrier, rtol=1e-2)
    assert np.ptp(np.abs(tf.h)) < 10.0


def test_compute_impulse_response_baseband_low_fs_lossy_line(lossy_si_app, low_fs_source_waveform):
    """
    Same low-fs (1 GSa/s, far below the schematic's 40 GSa/s) baseband case
    as test_compute_impulse_response_baseband_low_fs, but against the lossy
    fixture: |H| is not flat here, so unlike the lossless schematic (where
    h collapses to a single clean delta tap), the frequency-dependent loss
    across even this narrow +/-0.5GHz baseband window smears h across a few
    taps around the ~2ns line delay -- confirmed below by checking more than
    one tap is non-negligible, in addition to the same gain/length invariants.
    """
    raw = si_tf._extract_single_tf(lossy_si_app, "VSource", "VQubit", "VSource")
    tf = si_tf.compute_impulse_response(
        raw, "complex_baseband",
        fs=low_fs_source_waveform.fs, carrier_hz=low_fs_source_waveform.carrier_freq_hz,
    )
    assert tf.h is not None and tf.dt is not None
    assert np.isclose(tf.dt, 1.0 / low_fs_source_waveform.fs)

    df_known = raw.freqs[1] - raw.freqs[0]
    expected_len = int(round(low_fs_source_waveform.fs / df_known))
    assert len(tf.h) == expected_len
    assert expected_len < len(raw.freqs)

    assert tf.h.dtype == np.complex128
    h_at_carrier = raw.H[np.argmin(np.abs(raw.freqs - low_fs_source_waveform.carrier_freq_hz))]
    assert np.isclose(np.sum(tf.h), h_at_carrier, rtol=1e-2)
    assert np.ptp(np.abs(tf.h)) < 10.0

    # Unlike the lossless schematic's clean single-tap delta at low fs, the
    # lossy line's frequency-dependent attenuation smears energy across
    # neighboring taps -- confirm h is not degenerate to a single nonzero
    # sample (a real, physically-expected difference from the lossless case).
    mag = np.abs(tf.h)
    taps_above_1pct = np.sum(mag > 0.01 * mag.max())
    assert taps_above_1pct > 1

    # fftconvolve(v, h) sanity check, full/untruncated (see
    # test_compute_impulse_response_baseband_low_fs for why -- h is centered
    # via fftshift, so its peak isn't near index 0).
    from scipy.signal import fftconvolve
    v_in = low_fs_source_waveform.envelope_complex
    v_out = fftconvolve(v_in, tf.h, mode="full")
    assert np.isclose(np.max(np.abs(v_out)), np.abs(h_at_carrier) * np.max(np.abs(v_in)), rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])