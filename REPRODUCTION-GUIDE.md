# OmniVoice on Radeon 890M (gfx1150) — Reproduction Guide

**Purpose:** step-by-step method to get OmniVoice voice cloning running on AMD Radeon 890M integrated GPU (gfx1150), end-to-end from a fresh GPD Pocket 4 state.
**Audience:** future-Raz, or anyone with the same hardware trying to reproduce.
**Not a README.** The findings/performance report is at [`REPORT.md`](REPORT.md). This document is the HOW.
**Tested:** 2026-04-14 on GPD Pocket 4, AMD Ryzen AI 9 HX 370 (Strix Point), Radeon 890M iGPU, 32 GB unified memory, CachyOS, kernel 6.19.12-1-cachyos.

---

## Prerequisites

Before any of this works, you need:

1. **A working gfx1150 PyTorch build** at `~/builds/pytorch-gfx1150/` compiled against **Python 3.14** (not 3.12 — this matters). The build tree must contain `torch/_C.cpython-314-x86_64-linux-gnu.so`. No wheel in `dist/` is required — we'll load it directly from the source tree via a `.pth` file.
2. **Python 3.14** available on the system (`python3.14` command resolves). On CachyOS this is installed via the standard repos.
3. **ROCm 7.2 stack** installed system-wide — specifically `rocblas`, `miopen`, `hipblaslt`, `hip-runtime-amd`. Your PyTorch build linked against these at compile time.
4. **~6 GB free disk** for the OmniVoice model (2.9 GB) + Python dependencies (~3 GB across transformers, accelerate, librosa, etc.)
5. **Internet access** for the first-run model download from HuggingFace.

**Not required**: torchvision, CUDA, NPU drivers, ONNX Runtime, FastFlowLM, a second PyTorch venv, Docker.

---

## Step 1 — Create a clean venv

**Do not reuse `~/gaia-env-312`** or any other existing venv. Lemonade's venv is pinned to Python 3.12 + stock CUDA torch, and mixing causes the exact confusion that wasted an hour of reconnaissance earlier.

```bash
python3.14 -m venv ~/omnivoice-env
source ~/omnivoice-env/bin/activate
```

Verify the Python version is 3.14:

```bash
python --version   # Python 3.14.x
```

---

## Step 2 — Install torch's runtime Python dependencies

The gfx1150 torch build needs these but can't pull them itself because there's no wheel metadata:

```bash
pip install typing_extensions filelock sympy networkx jinja2 fsspec numpy
```

---

## Step 3 — Make the gfx1150 torch build importable

Instead of building a wheel, drop a `.pth` file into the venv's site-packages pointing at the source tree. Python will treat it as an importable `torch` package.

```bash
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "/home/raz/builds/pytorch-gfx1150" > "$SITE/pytorch-gfx1150.pth"
```

**Verify import + GPU detection:**

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('hip:', getattr(torch.version, 'hip', None))
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device='cuda')
    y = x @ x
    print('matmul ok, sum:', y.sum().item())
"
```

Expected output (approximate):
```
torch: 2.12.0a0+gitf86ea9e
hip: 7.2.26043
cuda_available: True
device: AMD Radeon 890M Graphics
matmul ok, sum: <some float>
```

If `cuda_available: False` or `hip: None`, the ROCm stack isn't being picked up. Check:
- `echo $LD_LIBRARY_PATH` — should include `/opt/rocm/lib` or equivalent
- `ls /opt/rocm/lib/libamdhip64.so*` — should exist
- `/opt/rocm/bin/rocminfo | grep gfx` — should list `gfx1150`

Do not proceed if this step fails. Everything below assumes torch sees the GPU.

---

## Step 4 — Install torchaudio (the tricky one)

The Python 3.14 wheel for `torchaudio==2.11.0` (current at time of writing) is hard-linked against `libcudart.so.13`, `libtorch_cuda.so`, and `libc10_cuda.so`. It will not load against our ROCm-only torch. Attempts to `import torchaudio` will fail at library resolution.

**Workaround**: install an older torchaudio and force-disable its C extension. OmniVoice only uses `torchaudio.functional.resample` and `torchaudio.compliance.kaldi.fbank`, both pure-Python wrappers over torch ops, so the C extension is not actually needed.

```bash
pip install --no-deps torchaudio==2.9.1
```

Then patch the extension flag. Find the file:

```bash
python -c "import torchaudio._extension; print(torchaudio._extension.__file__)"
```

Edit the `__init__.py` in that directory. Find the line:

```python
_IS_TORCHAUDIO_EXT_AVAILABLE = True
```

Change it to:

```python
_IS_TORCHAUDIO_EXT_AVAILABLE = False
```

Verify:

```bash
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__); import torchaudio.functional as F; print('resample ok')"
```

If this prints without error, torchaudio is usable for OmniVoice's needs.

---

## Step 5 — Install OmniVoice and its Python dependencies

Clone the repo:

```bash
cd ~
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
```

Install OmniVoice itself **without its dependency chain** (we install deps manually to prevent pip from pulling a torch wheel that would shadow the gfx1150 build):

```bash
pip install --no-deps -e .
```

Install the dependencies OmniVoice actually uses, by hand:

```bash
pip install \
  transformers>=5.3 \
  accelerate \
  pydub \
  tensorboardX \
  webdataset \
  soundfile \
  librosa \
  huggingface_hub \
  safetensors \
  tokenizers \
  regex \
  tqdm \
  pyyaml \
  httpx \
  hf_xet \
  psutil \
  typer \
  cffi \
  lazy_loader \
  audioread \
  decorator \
  joblib \
  msgpack \
  pooch \
  scikit-learn \
  scipy \
  numba \
  soxr
```

**Critical check:** after this, confirm your torch is still the gfx1150 build, not a replacement:

```bash
python -c "import torch; print(torch.__version__)"
```

Should still print `2.12.0a0+gitf86ea9e`. If it now prints `2.x.x+cu129` or similar, some dependency pulled a replacement — start over from Step 3 after `pip uninstall torch -y`.

---

## Step 6 — Verify OmniVoice loads

```bash
python -c "
from omnivoice.utils.audio import load_audio
import numpy as np
print('OmniVoice imports ok')
"
```

---

## Step 7 — First model download (~3 minutes, ~2.9 GB)

OmniVoice's `from_pretrained` fetches the model on first call. Trigger it explicitly so you're not waiting inside your first real run:

```bash
python -c "
from omnivoice import OmniVoice
import torch
m = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32)
print('model loaded')
"
```

Model caches under `~/.cache/huggingface/hub/`.

---

## Step 8 — Dry run on CPU (baseline)

```bash
omnivoice-infer \
  --text "Hello. This is a test." \
  --output ~/test-cpu.wav \
  --device cpu \
  --language en
```

Expect ~15 seconds for an ~11-word sentence (generates ~3.7 seconds of audio at 24 kHz mono). This proves the entire OmniVoice pipeline works before you involve the GPU.

---

## Step 9 — GPU inference

Set the HSA override so ROCm treats gfx1150 as gfx1150 rather than falling back to a generic target:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.0
```

Run:

```bash
omnivoice-infer \
  --text "Hello. This is a test." \
  --output ~/test-gpu.wav \
  --device cuda \
  --language en
```

**Expected behavior:**
- First run is slow (~26-30 seconds) due to MIOpen kernel-find on first use
- Subsequent runs ~4 seconds for the same short sentence
- Stderr will be **flooded with MIOpen warnings** like:
  ```
  MIOpen(HIP): Warning [IsEnoughWorkspace] Solver <GemmFwdRest>, workspace required: N, provided ptr: 0 size: 0
  ```
  This is expected and not a failure. It's the documented workspace=0 bug described in `../miopen-gfx1150/`. The fallback solver path completes correctly; the warnings indicate throughput overhead, not a broken run.
- Audio file should be written and sound correct on spot check

If no audio file is written or an actual Python traceback surfaces (not just MIOpen warnings), stop and read Step 11 (troubleshooting).

---

## Step 10 — Voice cloning with a reference sample

This is the real use case. You need:
1. A reference audio file (WAV preferred, 5-30 seconds, quiet room, natural speech)
2. The exact text of the reference audio (`--ref_text`)
3. The text you want spoken in the cloned voice (`--text`)

```bash
omnivoice-infer \
  --ref_audio ~/raz-voice.wav \
  --ref_text "Hey, this is Peter. This is a test of my voice clone running on a GPD Pocket Four." \
  --text "Hello. This is AgentSpyBoo speaking in my own cloned voice." \
  --output ~/raz-clone.wav \
  --device cuda \
  --language en
```

**Notes on reference audio:**
- WAV is preferred. If you have AAC/MP3/M4A from a phone recording, convert with:
  ```bash
  ffmpeg -i input.aac -ar 24000 -ac 1 output.wav
  ```
- 24 kHz mono is OmniVoice's internal rate; converting upfront avoids resampling inside the loader
- `--ref_text` must accurately transcribe the reference audio — stumbles, disfluencies, and punctuation all matter
- 5-10 seconds of reference works but produces a flatter clone. 25-35 seconds with natural speech variation produces a much more authentic result

---

## Step 11 — Troubleshooting

### `aifc.open` EOFError in traceback
You hit librosa's audioread fallback chain. Cause: the audio file is not being read by soundfile, and audioread's WAV handler is broken on Python 3.14. Fix:

```bash
pip install soundfile  # if not already
```

If soundfile is already installed, verify libsndfile is present at the system level:

```bash
pacman -Q libsndfile  # should show libsndfile 1.2+
```

If still failing, test soundfile directly on the problem file:

```bash
python -c "
import soundfile as sf
d, r = sf.read('/path/to/your/audio.wav', dtype='float32', always_2d=True)
print('OK:', d.shape, r)
"
```

If soundfile works directly but librosa still falls through, the file likely has an atypical WAV header. Re-encode via ffmpeg to a normalized format:

```bash
ffmpeg -i input.wav -ar 24000 -ac 1 -c:a pcm_s16le output.wav
```

### `cuda_is_available: False`
The ROCm stack isn't linking. Most common causes on CachyOS:
- `/opt/rocm/lib` not in `LD_LIBRARY_PATH`
- System ROCm version mismatch with what the torch build was compiled against
- User not in `video` or `render` group

Test:

```bash
rocminfo | grep -A2 "Name:.*gfx"
```

Should list your GPU. If `rocminfo` itself fails, fix ROCm before touching Python.

### `CUDA out of memory` during first GPU run
gfx1150 reports itself as having ~8-16 GB of accessible VRAM on a 32 GB unified system. The diffusion generation phase can spike. Try:

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

Or drop to CPU for the first run to confirm the model works at all.

### Inference runs but audio is silent or garbled
Warmup effect — the first GPU run can be wonky. Run it twice, second run should be clean. If still broken, try switching `dtype=torch.float16` → `torch.float32` by editing the `device_map` call in `infer.py` or passing `--dtype float32` if the CLI supports it in your version.

### Torchaudio import crashes
Confirm Step 4 was applied. Run:

```bash
python -c "
import torchaudio._extension as ext
print('ext available:', ext._IS_TORCHAUDIO_EXT_AVAILABLE)
"
```

Must print `False`. If it prints `True`, re-edit the file from Step 4.

---

## Step 12 — Benchmarking

A simple bench script lives at `~/projects/omnivoice-gfx1150-test/bench.py`. Run:

```bash
source ~/omnivoice-env/bin/activate
export HSA_OVERRIDE_GFX_VERSION=11.5.0
python ~/projects/omnivoice-gfx1150-test/bench.py cuda:0 gpu
python ~/projects/omnivoice-gfx1150-test/bench.py cpu cpu
```

See `REPORT.md` for the reference numbers from the 2026-04-14 run.

---

## Known good state summary

At the end of this guide you should have:

- `~/omnivoice-env/` — Python 3.14 venv with gfx1150 torch + OmniVoice
- `~/OmniVoice/` — cloned source tree, editable-installed
- `~/.cache/huggingface/hub/models--k2-fsa--OmniVoice/` — cached model (~2.9 GB)
- `omnivoice-infer` on `$PATH` (inside the venv)
- Ability to generate voice-cloned speech on GPU at ~3.7× speedup over CPU

Total disk footprint: ~8 GB including the torch build, venv, and cached model.
Total wall time from fresh state: ~20-30 minutes, most of it spent on the pip installs and the initial model download.
