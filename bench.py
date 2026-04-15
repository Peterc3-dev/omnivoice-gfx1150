import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch
from omnivoice import OmniVoice
import soundfile as sf

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
TAG    = sys.argv[2] if len(sys.argv) > 2 else DEVICE
print(f"[bench] device={DEVICE} tag={TAG}")
print(f"[bench] torch={torch.__version__} hip={torch.version.hip} cuda_avail={torch.cuda.is_available()}")

dmap = DEVICE
dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32

t0 = time.time()
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=dmap, dtype=dtype)
print(f"[bench] load took {time.time()-t0:.1f}s")

SHORT = "Hello, this is a test of OmniVoice on a Radeon 890M."
LONG  = ("The recursive routing racer is an experimental inference engine that combines "
         "hyperdimensional computing with diffusion language models, designed to run "
         "on AMD unified memory APUs without relying on MIOpen for small matmul workloads.")

def run(text, n=3, label=""):
    times = []
    for i in range(n+1):  # 1 warmup
        t = time.time()
        audio = model.generate(text=text)
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t
        if i == 0:
            print(f"[bench] {label} warmup: {dt:.2f}s")
            continue
        times.append(dt)
        print(f"[bench] {label} run{i}: {dt:.2f}s  audio_s={len(audio[0])/24000:.2f}s")
    dur = len(audio[0])/24000
    mean = sum(times)/len(times)
    rtf = mean/dur
    print(f"[bench] {label} MEAN {mean:.2f}s MIN {min(times):.2f}s MAX {max(times):.2f}s  RTF={rtf:.3f}  ({1/rtf:.1f}x realtime)")
    return mean, min(times), max(times), dur

print("=== SHORT ===")
run(SHORT, 3, "short")
print("=== LONG ===")
run(LONG, 3, "long")
