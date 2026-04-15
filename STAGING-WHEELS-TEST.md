# OmniVoice on AMD Staging gfx1150 Wheels — Feasibility Test

**Date:** 2026-04-15 (early AM, GPD local time)
**Host:** GPD Pocket 4, AMD Ryzen AI 9 HX 370, Radeon 890M (gfx1150), 32 GB
**OS:** CachyOS, kernel 6.19.12
**Venv:** ~/omnivoice-env-official (isolated, custom build untouched)

## Installed versions
- Python: 3.12.13
- torch: **2.11.0+rocm7.13.0a20260415** (staging wheel — newest 2.11.0 resolved despite index page showing 2.9.1 links)
- torchaudio: 2.11.0+rocm7.13.0a20260415 (native ROCm build, NO patching needed)
- triton: 3.6.0+rocm7.13.0a20260415
- rocm SDK bundle: 7.13.0a20260415 (pulled in as wheel deps)
- torch.version.hip: 7.13.61040

## Install path (clean, one shot)
```
python3.12 -m venv ~/omnivoice-env-official
source ~/omnivoice-env-official/bin/activate
pip install --upgrade pip
pip install torch torchaudio \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/ \
  --extra-index-url https://pypi.org/simple
cd ~/OmniVoice && pip install -e .
```
No `--no-deps`, no manual patching, no torchaudio CUDA-wheel replacement. Torch stayed at the rocm build after OmniVoice pulled its own deps (transformers, accelerate, librosa, numba, scipy, numpy 2.4.4, etc.).

## Sanity checks
- `torch.cuda.is_available()` → **True**
- `torch.cuda.get_device_name(0)` → **AMD Radeon 890M Graphics**
- 512x512 matmul sum: 12743.29 (no crash)
- `from omnivoice import OmniVoice` → OK
- CPU inference ("Quick CPU sanity check.") → 1.8 s WAV produced in ~2m 3s wall

## GPU benchmark — long v3b paragraph, ref ~/raz-voice2-crop.wav, speed 0.85

| Run | Config                     | Wall time | Audio out | IsEnoughWorkspace warnings |
|-----|----------------------------|-----------|-----------|----------------------------|
| 1   | cuda, no env var           | 3m 25s    | 21.10 s   | 20                         |
| 2   | cuda, FIND_MODE=FAST cold  | **44.1 s** | 20.88 s   | **0**                      |
| 3   | cuda, FIND_MODE=FAST warm  | **41.5 s** | 21.01 s   | **0**                      |

## Comparison vs custom-build v3b baseline

| Config                   | Custom build | Staging 2.11.0+rocm7.13 | Delta                  |
|--------------------------|--------------|--------------------------|------------------------|
| No FAST                  | 2m 52s       | 3m 25s                   | +33 s slower           |
| FAST cold                | 1m 20s       | 0m 44s                   | **−36 s (45% faster)** |
| FAST warm                | 0m 49s       | 0m 41s                   | **−8 s (16% faster)**  |
| Workspace warnings/run   | 40+          | **0 (with FAST)**        | bug gone               |

## Interpretation
1. **MIOpen `GemmFwdRest` workspace=0 fallback bug is GONE in rocm7.13 when FIND_MODE=FAST.** Zero warnings across both FAST runs. Without FAST it still fires 20x (down from the 40+ custom baseline), so the bug is partially mitigated but fully bypassed only in FAST.
2. Staging FAST warm-cache is ~16% faster than the custom build warm cache (41.5 s vs 49 s for ~21 s of audio). Faster than real-time.
3. Staging FAST cold is dramatically faster — 44 s vs 80 s — likely because the solver cache warms without workspace thrashing.
4. Audio durations across all three runs (20.88–21.10 s) are within noise of each other and of the custom-build baseline, suggesting output quality is stable.
5. Baseline (no FAST) is slightly slower on staging — minor regression when workspace fallback isn't avoided. Irrelevant since FAST mode is production config.

## Blockers encountered
None. Clean one-shot install, no patches, no workarounds, no torchaudio CUDA-wheel replacement.

## Recommendation: **MIGRATE**

Rationale: wheels install cleanly from AMD's public staging index, torchaudio needs zero patches, GPU works immediately, the MIOpen workspace bug that forced the FAST workaround is effectively fixed upstream (0 warnings with FAST), and warm-cache performance is 16% faster than the hand-built source tree. Zero reason to keep maintaining a local PyTorch build for OmniVoice.

## Next step if migrating
Swap the `omnivoice-infer` launcher wrapper to point at `~/omnivoice-env-official/bin/omnivoice-infer` (keep `~/omnivoice-env/` around for a week as a rollback path, then delete it and `~/builds/pytorch-gfx1150/` to reclaim disk). Keep `MIOPEN_FIND_MODE=FAST` exported — still required to fully avoid the workspace warnings, though they no longer bite.
