"""
Tests for si_qfi.source.waveform.SourceWaveform, using a real
SignalIntegrity Waveform as the envelope (matches the sample rate used by
tests/test_schematic_basic.si so it can also stand in for the driving
waveform in schematic-hookup tests).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("SignalIntegrity")

from si_qfi.source.waveform import SourceWaveform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# source/waveform.py
# ---------------------------------------------------------------------------

def test_source_waveform_reads_si_envelope(source_waveform, si_envelope):
    """Exercises SourceWaveform.__post_init__ against a real SI Waveform object."""
    assert source_waveform.fs == 40e9
    assert source_waveform.n_samples == 256
    assert np.allclose(source_waveform.envelope_complex, np.array(si_envelope.Values()))


def test_rf_waveform_at_same_fs_is_passthrough(source_waveform):
    """rf_waveform_at() at the waveform's own fs should equal .rf_waveform exactly."""
    t, v = source_waveform.rf_waveform_at(source_waveform.fs)
    assert np.array_equal(t, source_waveform.t)
    assert np.allclose(v, source_waveform.rf_waveform)


def test_rf_waveform_at_resamples(source_waveform):
    """
    rf_waveform_at() at a different fs should resample: same physical
    duration, different sample count and spacing.
    """
    fs_new = source_waveform.fs * 2.0
    t_new, v_new = source_waveform.rf_waveform_at(fs_new)
    assert len(v_new) == len(t_new)
    assert len(v_new) == 2 * source_waveform.n_samples
    assert np.isclose(t_new[1] - t_new[0], 1.0 / fs_new)
    # Same overall duration (within one sample either way)
    orig_duration = source_waveform.t[-1] - source_waveform.t[0]
    new_duration = t_new[-1] - t_new[0]
    assert abs(new_duration - orig_duration) < 2 * max(1 / fs_new, source_waveform.dt)


def test_check_sample_rate_for_real_axis_uses_explicit_fs(source_waveform):
    """
    check_sample_rate_for_real_axis() takes fs explicitly now (the
    schematic's native rate in real usage), not source.fs implicitly.
    """
    # 40 GSa/s is enough for harmonic_order=3 at a 5 GHz carrier (needs >= 30 GSa/s).
    source_waveform.check_sample_rate_for_real_axis(40e9, harmonic_order=3)
    with pytest.raises(ValueError, match="native sample rate"):
        source_waveform.check_sample_rate_for_real_axis(10e9, harmonic_order=3)


if __name__ == "__main__":
    pytest.main([__file__])
