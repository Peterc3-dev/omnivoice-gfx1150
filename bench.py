"""OmniVoice benchmark on gfx1150-class AMD iGPUs.

Usage:
    python bench.py [device] [tag]

    device  torch device string, e.g. "cpu" or "cuda" (default "cpu")
    tag     free-form label printed alongside results (default = device)

Runs one warmup plus three timed generations for a short and a long prompt and
prints mean / min / max wall time and the real-time factor for each.
"""

import os
import sys
import time
import warnings

# Environment must be configured before torch is imported, so these imports
# intentionally sit below module-level statements (E402 is expected here).
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch  # noqa: E402
from omnivoice import OmniVoice  # noqa: E402

from bench_stats import SAMPLE_RATE, audio_duration_s, summarize  # noqa: E402

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
TAG = sys.argv[2] if len(sys.argv) > 2 else DEVICE
print(f"[bench] device={DEVICE} tag={TAG}")
print(f"[bench] torch={torch.__version__} hip={torch.version.hip} cuda_avail={torch.cuda.is_available()}")

dmap = DEVICE
dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32

t0 = time.time()
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=dmap, dtype=dtype)
print(f"[bench] load took {time.time()-t0:.1f}s")

SHORT = "Hello, this is a test of OmniVoice on a Radeon 890M."
LONG = ("The recursive routing racer is an experimental inference engine that combines "
        "hyperdimensional computing with diffusion language models, designed to run "
        "on AMD unified memory APUs without relying on MIOpen for small matmul workloads.")


def run(text, n=3, label=""):
    times = []
    audio = None
    for i in range(n + 1):  # 1 warmup
        t = time.time()
        audio = model.generate(text=text)
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t
        if i == 0:
            print(f"[bench] {label} warmup: {dt:.2f}s")
            continue
        times.append(dt)
        print(f"[bench] {label} run{i}: {dt:.2f}s  audio_s={len(audio[0])/SAMPLE_RATE:.2f}s")
    dur = audio_duration_s(len(audio[0]))
    stats = summarize(times, dur)
    print(f"[bench] {label} MEAN {stats['mean']:.2f}s MIN {stats['min']:.2f}s "
          f"MAX {stats['max']:.2f}s  RTF={stats['rtf']:.3f}  "
          f"({stats['realtime_x']:.1f}x realtime)")
    return stats["mean"], stats["min"], stats["max"], dur


if __name__ == "__main__":
    print("=== SHORT ===")
    run(SHORT, 3, "short")
    print("=== LONG ===")
    run(LONG, 3, "long")
