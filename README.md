# omnivoice-gfx1150

**Running k2-fsa/OmniVoice voice-cloning TTS on an AMD Radeon 890M integrated GPU (gfx1150 / Strix Point / Ryzen AI 9 HX 370 class).**

As far as I can find across English-language GitHub, Medium, Reddit, HackerNews, Level1Techs, Phoronix, and English-indexed Chinese sources (Zhihu, Bilibili, CSDN), this repo is the shortest known path — **found by me** as of 2026-04-15 — to running k2-fsa/OmniVoice on this specific integrated GPU. If earlier work exists on Strix Point / gfx1150 specifically, I didn't locate it. Corrections welcome — if you have an earlier run, open an issue and I'll cite it here.

This repo contains the benchmark script, the end-to-end report, and the step-by-step reproduction guide. It is not a fork of OmniVoice — it's the notes and data from getting OmniVoice to run on specific iGPU hardware.

## UPDATE 2026-04-15 — AMD staging wheels work, source build no longer necessary

As of 2026-04-15, AMD is shipping gfx1150 PyTorch nightlies at `https://rocm.nightlies.amd.com/v2-staging/gfx1150/` (`torch 2.11.0+rocm7.13.0a20260415` at time of writing — wheels are getting updated daily). I tested them in a parallel venv on the same hardware and they **install cleanly, run faster than the source build, and most of the MIOpen workspace=0 pain is fixed upstream**. See [`STAGING-WHEELS-TEST.md`](STAGING-WHEELS-TEST.md) for the full benchmark data.

**The current recommended install path** (skips the whole source build + torchaudio patch dance in the reproduction guide below):

```bash
python3.12 -m venv ~/omnivoice-env
source ~/omnivoice-env/bin/activate
pip install --upgrade pip
pip install torch torchaudio \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/ \
  --extra-index-url https://pypi.org/simple
cd ~
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
export MIOPEN_FIND_MODE=FAST      # still recommended — zeros out the remaining workspace warnings
export HSA_OVERRIDE_GFX_VERSION=11.5.0
```

That's it. No `.pth` file, no torchaudio patch, no Python 3.14 lock-in, no source compilation.

**`REPRODUCTION-GUIDE.md` in this repo documents the OLDER source-build path** — it's kept as historical record of what was needed before AMD's staging wheels worked, but is **no longer the recommended path**. Use the 4-line install above instead.

**Measured performance** on GPD Pocket 4 (Ryzen AI 9 HX 370, Radeon 890M) with the staging wheels:
- Cold start: 44s for 20s of output audio
- Warm cache: **41.5s for 20s of output audio** (faster than real-time)
- MIOpen workspace warnings: **0 per generation with FAST mode** (was 40+ on ROCm 7.2)

## Prior art I'm building on (acknowledgements)

- **[k2-fsa/OmniVoice#67](https://github.com/k2-fsa/OmniVoice/issues/67)** — OmniVoice working on AMD ROCm generally was shown by **Alexander-Ger-Reich** on 2026-04-09, with **gunmaden** confirming on 2026-04-12. They used the `rocm.nightlies.amd.com/v2/gfx110X-all/` wheel index, which targets discrete RDNA3 (gfx1100/1101/1102 — RX 7600/7700/7800/7900 class). Different hardware class from Strix Point iGPU, but this is the work I'm building on and it disproves any "first OmniVoice on AMD" framing.
- **[bkpaine — Voice Cloning on AMD Strix Halo (Chatterbox TTS)](https://medium.com/@bkpaine1/voice-cloning-on-amd-strix-halo-running-chatterbox-tts-with-native-gpu-acceleration-fa4a3db5e82c)**, March 2026 — the closest adjacent work I found for "voice cloning on AMD integrated graphics." Different TTS model (Chatterbox, not OmniVoice), different chip class (Strix Halo / gfx1151 / Radeon 8060S — ~55-120W APU, vs Strix Point / gfx1150 / Radeon 890M at ~15-54W). This is also where the `MIOPEN_FIND_MODE=FAST` workaround I use was first documented publicly for gfx1151.
- **[Peterc3-dev/miopen-gfx1150](https://github.com/Peterc3-dev/miopen-gfx1150)** — my own earlier documentation of the three MIOpen bugs on gfx1150 that this project is rate-limited by.
- **[Peterc3-dev/pytorch-gfx1150](https://github.com/Peterc3-dev/pytorch-gfx1150)** — my earlier source-built PyTorch wheel this project depends on.

## What's specifically new in this repo

- **Reproducible install recipe** for OmniVoice on Python 3.14 + source-built gfx1150 torch + the torchaudio `_IS_TORCHAUDIO_EXT_AVAILABLE=False` workaround (required because the Py3.14 torchaudio wheel is CUDA-linked and won't load against ROCm torch)
- **Measured benchmarks** on a GPD Pocket 4 (Radeon 890M / HX 370): CPU baseline, GPU without workaround, GPU with `MIOPEN_FIND_MODE=FAST`. Published as a reference point for anyone else evaluating this chip for diffusion TTS work.
- **Second production-shaped in-the-wild reproducer** for the MIOpen `GemmFwdRest` workspace=0 bug on gfx1150/1151-class hardware, contributed to the [miopen-gfx1150 repo](https://github.com/Peterc3-dev/miopen-gfx1150/blob/master/reproducers/2026-04-15-omnivoice-gfx1150.md) (first was TimLawrenz's NanoDiT gfx1151 training run)
- **3.5× measured sustained speedup** over baseline via `MIOPEN_FIND_MODE=FAST` on this specific workload

## Hardware

GPD Pocket 4 — AMD Ryzen AI 9 HX 370 (Strix Point), Radeon 890M iGPU (gfx1150), XDNA 2 NPU (50 TOPS rated), 32 GB unified memory. CachyOS, kernel 6.19.12.

## TL;DR results

| Config | Wall time (20s output) | Notes |
|---|---|---|
| CPU (torch ROCm, gfx1150, built from source) | ~50 s | RTF 4.0, 0.25× realtime |
| GPU baseline | 2m 52s | Rate-limited by MIOpen `GemmFwdRest` workspace=0 fallback |
| GPU + `MIOPEN_FIND_MODE=FAST` cold | 1m 20s | 54% faster than baseline |
| **GPU + `MIOPEN_FIND_MODE=FAST` warm** | **0m 49s** | **72% faster than baseline, 3.5× speedup** |

Output quality is indistinguishable across runs — verified by A/B listening.

## Key findings

- **OmniVoice runs end-to-end on gfx1150** using a custom-built PyTorch wheel (torch `2.12.0a0+gitf86ea9e`, HIP 7.2.26043, built against Python 3.14).
- **torchaudio is the main obstacle.** The upstream Python 3.14 wheel hard-links to `libcudart.so.13`; we work around it by installing `torchaudio==2.9.1 --no-deps` and patching `_IS_TORCHAUDIO_EXT_AVAILABLE = False`. OmniVoice only uses pure-Python torchaudio ops (`F.resample`, `kaldi.fbank`), so the C extension disable is harmless.
- **The gfx1150 MIOpen workspace=0 bug** (see [Peterc3-dev/miopen-gfx1150](https://github.com/Peterc3-dev/miopen-gfx1150)) fires 40+ times per generation with progressively smaller workspace requests: 424 MB, 313 MB, 106 MB, 42 MB, 21 MB, 5 MB, 13 MB, 27 MB, 41 MB. Every one of them falls through the naive solver path, which is the reason baseline GPU is only ~0.9× realtime instead of the 3-5× you'd expect from a modern iGPU on this workload.
- **Setting `MIOPEN_FIND_MODE=FAST`** (community-recommended workaround, see [bkpaine's Chatterbox writeup](https://medium.com/@bkpaine1/voice-cloning-on-amd-strix-halo-running-chatterbox-tts-with-native-gpu-acceleration-fa4a3db5e82c)) routes around the fallback and gives a sustained 3.5× speedup on warm runs, with no quality loss.
- **MIOpen find DB + ROCm driver state** persists across Python process exits at the kernel driver layer. First run in a fresh install pays a ~33-second driver warmup tax; subsequent runs skip it entirely.
- **No CK `illegal opcode` crash.** OmniVoice doesn't hit Winograd Fury shapes.
- **No hipBLAS Tensile failure.** The hipBLASLt path appears clean for the attention matmuls.
- **torchvision is not required.** OmniVoice doesn't depend on it.

## Files

- [`REPRODUCTION-GUIDE.md`](REPRODUCTION-GUIDE.md) — step-by-step reproduction, from venv creation through first voice clone. 12 sections covering prerequisites, torch build integration, the torchaudio patch, OmniVoice install, first run, GPU run, voice cloning with reference samples, troubleshooting, and a known-good end state summary.
- [`REPORT.md`](REPORT.md) — findings and measurements from the 2026-04-14 end-to-end test run, including the detailed performance table and MIOpen issue analysis.
- [`bench.py`](bench.py) — the benchmark script used to produce the speedup numbers. Takes device and label arguments, runs 3 warmup + 3 timed generations, reports mean / min / max.

## Status

- **Working** on the config described above
- **Not a plug-and-play install** — requires a custom-built PyTorch gfx1150 wheel, which is separate work. AMD ships experimental gfx1150 nightlies at `https://rocm.nightlies.amd.com/v2-staging/gfx1150/` which may simplify this; not yet validated for this specific use case.
- **Tested against OmniVoice commit `main` as of 2026-04-14**
- **Verified voice clone quality** through A/B listening — results clear "close family member can't distinguish" threshold on a short reference (6-10 seconds of source audio).

## Related work

- [Voice Cloning on AMD Strix Halo: Chatterbox TTS](https://medium.com/@bkpaine1/voice-cloning-on-amd-strix-halo-running-chatterbox-tts-with-native-gpu-acceleration-fa4a3db5e82c) — bkpaine, March 2026, the closest prior art (different chip, different TTS model, same MIOpen workaround).
- [Peterc3-dev/miopen-gfx1150](https://github.com/Peterc3-dev/miopen-gfx1150) — the original documentation of the `GemmFwdRest` workspace=0 bug that this project is rate-limited by.
- [Peterc3-dev/pytorch-gfx1150](https://github.com/Peterc3-dev/pytorch-gfx1150) — the source-built PyTorch wheel this project depends on.
- [ROCm/TheRock#2591](https://github.com/ROCm/TheRock/issues/2591) — upstream MIOpen issue tracker where the workspace bug is being addressed.
- [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — the TTS model itself.

## License

This documentation is released under the MIT License. OmniVoice itself is Apache 2.0. The bench script is public domain.

## Contact

Peterc3.dev@gmail.com
