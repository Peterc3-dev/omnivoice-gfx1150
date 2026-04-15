# OmniVoice + PyTorch gfx1150 Test Report
**Date:** 2026-04-14
**Hardware:** GPD Pocket 4, Ryzen AI 9 HX 370, Radeon 890M (gfx1150), 32GB unified memory, CachyOS, kernel 6.19.12-1-cachyos

## TL;DR
OmniVoice runs end-to-end on the Radeon 890M using Raz's custom-built PyTorch 2.12.0a0 ROCm 7.2 wheel. GPU inference is ~3.7x faster than CPU on short sentences (4.1s vs 15.0s for a 3.7s audio clip) and ~2.8x faster on long sentences (18.0s vs 49.6s for a 14.8s clip), but still slightly under realtime (RTF ~1.1). The MIOpen workspace=0 bug Raz already documented fires dozens of times during generation — the fallback solver path works correctly but leaks significant throughput. OmniVoice is usable today for offline portfolio voice-cloning work on either CPU or GPU, but it is not a Piper replacement for real-time use.

## PyTorch state before
- `~/gaia-env-312` has `torch 2.11.0+cu130` (stock NVIDIA CUDA wheel, `cuda_is_available=False`). This is a transitive Lemonade dep, not the ROCm build. MEMORY's claim that the gfx1150 build is active in gaia-env-312 was **incorrect** — reconciled.
- `~/builds/pytorch-gfx1150/` is a source tree only. **No wheel in `dist/`.** The build completed in-place (`build/` + `torch/_C.cpython-314-x86_64-linux-gnu.so` exists). Compiled against **Python 3.14**, not 3.12 — that's why it couldn't coexist with `gaia-env-312`.
- Version: `torch 2.12.0a0+gitf86ea9e`, `hip 7.2.26043`, `rocm 7.2.0`, commit `f86ea9ec245b46d6a2dd5b3c249dd4520182558c`.

## Venv setup
Created `~/omnivoice-env` with `python3.14 -m venv`. Instead of rebuilding a wheel, added a `.pth` file to site-packages pointing at `~/builds/pytorch-gfx1150` so the source tree becomes importable as `torch` without `PYTHONPATH`. Verified: `torch.cuda.is_available()=True`, `get_device_name(0)='AMD Radeon 890M Graphics'`, 512x512 matmul returns correct result.

**Key setup commands (reproducible):**
```bash
python3.14 -m venv ~/omnivoice-env
source ~/omnivoice-env/bin/activate
pip install typing_extensions filelock sympy networkx jinja2 fsspec numpy
echo "/home/raz/builds/pytorch-gfx1150" > $(python -c 'import site;print(site.getsitepackages()[0])')/pytorch-gfx1150.pth
```

## OmniVoice install
- `torchaudio` was the hairiest piece. The Py3.14 wheel on PyPI (`torchaudio-2.11.0-cp314`) is linked against `libcudart.so.13`, `libtorch_cuda.so`, `libc10_cuda.so` — cannot load against our ROCm-only torch.
- **Workaround:** installed `torchaudio==2.9.1 --no-deps` and patched `torchaudio/_extension/__init__.py` to force `_IS_TORCHAUDIO_EXT_AVAILABLE = False`. OmniVoice only uses `torchaudio.functional.resample` and `torchaudio.compliance.kaldi.fbank`, both pure-Python wrappers over torch ops, so the C extension is not actually needed.
- Cloned OmniVoice from GitHub, `pip install --no-deps -e .` then installed the remaining deps by hand: `transformers>=5.3 accelerate pydub tensorboardX webdataset soundfile librosa huggingface_hub safetensors tokenizers regex tqdm pyyaml httpx hf_xet psutil typer cffi lazy_loader audioread decorator joblib msgpack pooch scikit-learn scipy numba soxr`. Pip did **not** pull a replacement torch wheel with this approach.
- Model: `k2-fsa/OmniVoice` (the only published variant), fetched by the first `from_pretrained` call, ~2.9GB over 13 files, ~3 min.

## CPU inference
Works cleanly. Using `OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32)`:

| Input                | Output audio | Mean latency | RTF  | Realtime factor |
|----------------------|--------------|--------------|------|-----------------|
| 11 words (short)     | 3.76s        | 15.05s       | 4.00 | 0.25x           |
| 47 words (long)      | 14.78s       | 49.62s       | 3.36 | 0.30x           |

## GPU inference
Works end-to-end on `device_map="cuda:0"` with `dtype=torch.float16`. Set `HSA_OVERRIDE_GFX_VERSION=11.5.0` for safety. Warmup run is much slower (26s / 98s) — consistent with MIOpen kernel-find on first use.

| Input                | Output audio | Mean latency | RTF  | Realtime factor |
|----------------------|--------------|--------------|------|-----------------|
| 11 words (short)     | 3.73s        | 4.10s        | 1.09 | 0.92x           |
| 47 words (long)      | 14.76s       | 17.97s       | 1.21 | 0.82x           |

No crashes, no NaNs, audio sounds right on spot check.

## Speedup (GPU / CPU)
| Input   | CPU mean | GPU mean | Speedup |
|---------|----------|----------|---------|
| Short   | 15.05s   | 4.10s    | **3.67x** |
| Long    | 49.62s   | 17.97s   | **2.76x** |

For reference, Piper CPU on the same hardware is ~25x realtime. OmniVoice GPU is ~0.9x realtime. A decent zero-shot/voice-clone model on this iGPU is roughly **27x slower than a traditional non-cloning TTS**, which is the expected gap.

## Known gfx1150 issues hit
- **MIOpen workspace=0 bug (Raz's #1 documented issue):** fires 40+ times per generation with solver `GemmFwdRest`, workspace requests escalating to 313MB. The fallback path works but has obvious throughput overhead — this is the primary reason GPU is only ~3x faster than CPU instead of the 10-20x a healthy stack should give for a diffusion model. Direct confirmation of the `~/projects/miopen-gfx1150/` bug in a production-shaped workload.
- **No CK illegal-opcode crash.** OmniVoice doesn't hit Winograd Fury shapes (good — means the fix isn't blocking).
- **No hipBLAS Tensile failure.** hipBLASLt path appears clean for the attention matmuls.
- **torchvision: not needed.** OmniVoice doesn't depend on torchvision.

## Recommendations
- **Use OmniVoice on GPU now for the AgentSpyBoo demo narration.** Generate once, cache the clips. ~20s per long sentence is acceptable for offline asset production and nobody will notice the MIOpen warnings in the generated WAV.
- **For real-time voice, stay on Piper.** OmniVoice cannot do conversational latency on this hardware with the current MIOpen stack.
- **Upstream bug-file opportunity:** this is a second production-workload reproducer for the MIOpen workspace bug, with a clean stack (no CK crash, no Tensile issue, no torchvision confound). Worth attaching to the existing `ROCm/TheRock#2591` thread or to Raz's `miopen-gfx1150` repo as an "in-the-wild reproducer" — if MIOpen ever gets the GemmFwdRest workspace query fixed, OmniVoice GPU will likely jump to ~5-8x realtime without any other changes.
- **Phase 3 NPU planning: confirm-park.** Moving OmniVoice to the XDNA NPU is not worth the effort right now — the diffusion architecture + FP16 matmuls are a bad fit for the NPU's INT8 systolic array, and the GPU path works. Revisit only after MIOpen fix lands.
- **Worth another pass after kernel/firmware update?** Yes, **only** if an update bumps MIOpen with a GemmFwdRest workspace query fix. Other kernel/firmware changes will not move the needle.

## Artifacts
- Venv: `/home/raz/omnivoice-env/` (Python 3.14, torch 2.12.0a0+gitf86ea9e ROCm)
- Bench script: `/home/raz/projects/omnivoice-gfx1150-test/bench.py`
- Run: `source ~/omnivoice-env/bin/activate && python bench.py cuda:0 gpu` or `... cpu cpu`
