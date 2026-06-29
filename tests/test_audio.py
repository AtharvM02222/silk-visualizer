"""
Tests for audio/analyzer.py

These tests use a synthetically generated sine-wave audio signal so that
the tests run without needing a real audio file and without hitting disk.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from audio.analyzer import AudioAnalyzer

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050
DURATION_S = 3  # seconds — short enough to run quickly
FPS = 30


def _make_sine_wav(
    path: str, freq: float = 440.0, duration: float = DURATION_S
) -> None:
    """Write a short sine-wave WAV to *path*."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wav = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    sf.write(path, wav, SAMPLE_RATE)


@pytest.fixture(scope="module")
def sine_wav(tmp_path_factory):
    """Temporary WAV file containing a 440 Hz sine wave."""
    p = tmp_path_factory.mktemp("audio") / "sine.wav"
    _make_sine_wav(str(p))
    return str(p)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_analyzer_loads(sine_wav):
    """AudioAnalyzer should construct without error."""
    a = AudioAnalyzer(sine_wav, fps=FPS)
    assert a.n_frames > 0


def test_n_frames_reasonable(sine_wav):
    """n_frames should be close to duration × fps."""
    a = AudioAnalyzer(sine_wav, fps=FPS)
    expected = int(DURATION_S * FPS)
    # Allow ±5 frames for librosa framing differences
    assert abs(a.n_frames - expected) <= 5


def test_feature_ranges(sine_wav):
    """All per-frame feature arrays should be normalised to [0, 1]."""
    a = AudioAnalyzer(sine_wav, fps=FPS)
    for name, arr in [
        ("bass", a.bass),
        ("mids", a.mids),
        ("highs", a.highs),
        ("rms", a.rms),
        ("onset", a.onset),
    ]:
        assert arr.min() >= 0.0, f"{name} has values < 0"
        assert arr.max() <= 1.0 + 1e-6, f"{name} has values > 1"


def test_is_beat_boolean(sine_wav):
    """is_beat should be a boolean numpy array of the correct length."""
    a = AudioAnalyzer(sine_wav, fps=FPS)
    assert a.is_beat.dtype == bool
    assert len(a.is_beat) == a.n_frames


def test_get_frame_returns_all_keys(sine_wav):
    """get_frame() should return all expected keys with the right types."""
    a = AudioAnalyzer(sine_wav, fps=FPS)
    feat = a.get_frame(0)
    for key in ("rms", "bass", "mids", "highs", "onset", "is_beat"):
        assert key in feat, f"Missing key: {key}"
    assert isinstance(feat["is_beat"], bool)
    assert isinstance(feat["bass"], float)


def test_get_frame_clamping(sine_wav):
    """get_frame() should clamp out-of-range indices gracefully."""
    a = AudioAnalyzer(sine_wav, fps=FPS)
    # Should not raise
    a.get_frame(-1)
    a.get_frame(a.n_frames + 100)


def test_normalize_helper():
    """_normalize should scale values to [0, 1] and clip outliers."""
    arr = np.array([0.0, 1.0, 2.0, 100.0])  # 100 is an outlier
    result = AudioAnalyzer._normalize(arr, percentile=97)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_smooth_helper_no_alpha():
    """_smooth with alpha=0 should return the same values."""
    arr = np.array([0.1, 0.5, 0.9, 0.2], dtype=np.float32)
    result = AudioAnalyzer._smooth(arr, alpha=0.0)
    np.testing.assert_allclose(result, arr, atol=1e-6)


def test_smooth_helper_heavy():
    """_smooth with alpha=0.99 should produce a very slowly changing signal."""
    arr = np.zeros(100, dtype=np.float32)
    arr[0] = 1.0  # spike at start
    result = AudioAnalyzer._smooth(arr, alpha=0.99)
    # After 100 frames of heavy smoothing the value should still be > 0
    assert result[-1] > 0.0
    # And it should be monotonically decreasing (for this input)
    assert all(result[i] >= result[i + 1] for i in range(len(result) - 1))
