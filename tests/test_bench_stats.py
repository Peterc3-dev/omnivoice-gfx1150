"""Unit tests for bench_stats — pure logic, no heavy deps (torch/omnivoice)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench_stats import SAMPLE_RATE, audio_duration_s, summarize  # noqa: E402


def test_audio_duration_default_rate():
    # One second of audio at the native 24 kHz rate.
    assert audio_duration_s(SAMPLE_RATE) == pytest.approx(1.0)
    assert audio_duration_s(0) == 0.0
    assert audio_duration_s(48000) == pytest.approx(2.0)


def test_audio_duration_custom_rate():
    assert audio_duration_s(16000, sample_rate=16000) == pytest.approx(1.0)


def test_audio_duration_validation():
    with pytest.raises(ValueError):
        audio_duration_s(100, sample_rate=0)
    with pytest.raises(ValueError):
        audio_duration_s(-1)


def test_summarize_basic():
    # Three 10s runs over 20s of audio -> RTF 0.5, 2x realtime.
    stats = summarize([10.0, 10.0, 10.0], duration_s=20.0)
    assert stats["mean"] == pytest.approx(10.0)
    assert stats["min"] == 10.0
    assert stats["max"] == 10.0
    assert stats["duration_s"] == 20.0
    assert stats["rtf"] == pytest.approx(0.5)
    assert stats["realtime_x"] == pytest.approx(2.0)


def test_summarize_min_max_mean():
    stats = summarize([2.0, 4.0, 6.0], duration_s=4.0)
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["min"] == 2.0
    assert stats["max"] == 6.0
    assert stats["rtf"] == pytest.approx(1.0)
    assert stats["realtime_x"] == pytest.approx(1.0)


def test_summarize_rtf_reciprocal_consistency():
    stats = summarize([5.0], duration_s=10.0)
    # realtime_x must always equal 1 / rtf.
    assert stats["realtime_x"] == pytest.approx(1.0 / stats["rtf"])


def test_summarize_validation():
    with pytest.raises(ValueError):
        summarize([], duration_s=10.0)
    with pytest.raises(ValueError):
        summarize([1.0], duration_s=0.0)
