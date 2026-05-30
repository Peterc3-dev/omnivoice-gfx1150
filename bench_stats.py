"""Pure-logic helpers for the OmniVoice benchmark.

This module deliberately has NO heavy dependencies (no torch / omnivoice /
soundfile). It holds the timing-statistics math used by ``bench.py`` so the
numbers can be unit-tested without a GPU, model weights, or PyTorch installed.

Audio is generated at a fixed 24 kHz sample rate by OmniVoice.
"""

from __future__ import annotations

SAMPLE_RATE = 24000


def audio_duration_s(num_samples: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Return the duration in seconds of ``num_samples`` audio samples."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    return num_samples / sample_rate


def summarize(times: list[float], duration_s: float) -> dict[str, float]:
    """Summarize a list of per-run wall-clock times against output duration.

    Returns mean / min / max wall time, the real-time factor (RTF = wall /
    audio seconds) and the realtime multiplier (1 / RTF).

    ``times`` must be non-empty and ``duration_s`` must be > 0.
    """
    if not times:
        raise ValueError("times must be non-empty")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")

    mean = sum(times) / len(times)
    rtf = mean / duration_s
    return {
        "mean": mean,
        "min": min(times),
        "max": max(times),
        "duration_s": duration_s,
        "rtf": rtf,
        "realtime_x": 1.0 / rtf,
    }
