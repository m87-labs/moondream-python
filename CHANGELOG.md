# Changelog

## 2.2.0

- Photon's optimized local inference now spans NVIDIA Ampere, Ada, Hopper, and
  Blackwell GPUs, including A10/A10G, A100, RTX 3090, L4, H100, B200, and RTX
  PRO 6000 Blackwell.
- Improved local response latency and throughput for Moondream and Qwen models.
- Added accelerated Whisper transcription on L4 and RTX 3090.
- Reduced transcription latency and host overhead for Qwen ASR and Parakeet,
  including one-at-a-time workloads.
- A standard `pip install moondream==2.2.0` now selects the matching optimized
  Photon runtime and GPU kernels automatically.

See the underlying engine [0.7.1](https://github.com/m87-labs/kestrel/blob/main/CHANGELOG.md#071--2026-09-09)
release notes for the complete engine changes.

## 2.1.1

- Added Gemma 4 26B-A4B base and instruction-tuned checkpoints to Photon.
- Photon transcription includes Whisper large-v3-turbo for transcription and
  English translation, and now adds Qwen3-ASR 0.6B and 1.7B and Parakeet TDT
  0.6B v3. The speech models support files and live PCM, long-form audio,
  progressive results, language and prompt controls, and timestamps or forced
  alignment where the selected model supports them.
- Reduced peak host memory while loading larger Qwen, Gemma, and Moondream 3
  checkpoints, and expanded precompiled H100 and B200 execution for the new
  model workloads.
- Pinned the Photon runtime exactly so every `moondream 2.1.1` installation
  uses the engine and bundled-kernel release validated with this client.

See the underlying engine [0.7.0](https://github.com/m87-labs/kestrel/blob/main/CHANGELOG.md#070--2026-09-01)
release notes for the complete engine changes.

## 2.1.0

- Added production NVIDIA B200 support to Photon, with expanded precompiled
  execution for Qwen 3.5, Qwen 3.6, and Gemma 4 models.
- Added Photon transcription and English translation with
  `openai/whisper-large-v3-turbo`, including long-form audio files, live PCM
  streams, and segment or word timestamps.
- Photon model clients now expose the selected model's complete capability
  surface through one model-bound bridge, including progressive and
  caller-driven streams.
- Corrected reasoning-enabled Gemma 4 queries so direct answers are returned
  as answers and reasoning channel headers do not appear in customer-visible
  output.
- Pinned the Photon runtime exactly so every `moondream 2.1.0` installation
  uses the engine and bundled-kernel release validated with this client.

See the underlying engine [0.6.1](https://github.com/m87-labs/kestrel/blob/main/CHANGELOG.md#061--2026-08-26)
and [0.6.0](https://github.com/m87-labs/kestrel/blob/main/CHANGELOG.md#060--2026-08-26)
release notes for the complete engine changes.

## 2.0.1

- Fixed Photon local inference for finetuned checkpoints whose IDs use the
  current ULID format.

## 2.0.0

- Upgraded Photon local inference to `kestrel 0.5.0`. Moondream 2 and
  Moondream 3 now automatically use bundled whole-model decode kernels on
  supported NVIDIA GPUs, with the regular optimized path retained as a
  transparent fallback.
- Added local inference support for Moondream 3.1 9B A2B.
- Photon can now run every model bundled with Kestrel 0.5, including the
  supported Qwen 3.5, Qwen 3.6, and Gemma 4 variants.
- Improved responsiveness under large request bursts and made startup and
  out-of-memory failures safer for concurrent workloads.
- Local sampling settings are now forwarded consistently, including
  temperature, top-p, output limits, and spatial object limits.
- Added multi-turn chat and spatial-reference guidance consistently across
  Cloud and Photon local inference.
- Photon clients can now release shared local engine resources deterministically
  with `close()` or a context manager.
- Pinned Kestrel exactly so every `moondream 2.0.0` installation uses the
  engine and bundled-kernel release validated with this client.

See the [Kestrel 0.5.0 changelog](https://github.com/m87-labs/kestrel/blob/main/CHANGELOG.md#050--2026-08-02)
for the complete engine release notes.

## 1.3.0

- Upgraded the Photon local inference engine to `kestrel 0.4.2`; see the
  [Kestrel changelog](https://github.com/m87-labs/kestrel/blob/main/CHANGELOG.md#042--2026-06-06).
- Updated the package Python requirement to match Kestrel's supported
  Python 3.10-3.14 range.

## 1.2.2

- Upgraded the Photon local inference engine to `kestrel 0.4.0`. On Apple
  Silicon, local installs now work across supported PyTorch 2.9-2.12 builds
  instead of being tied to one PyTorch minor version.

## 1.2.1

- Auto-detect the Photon local inference device when `device` is not specified,
  choosing CUDA when available and Apple Silicon MPS otherwise, while preserving
  explicit device overrides.
- Removed the outdated README caveat that limited Apple Silicon local inference
  to Python 3.12.

## 1.2.0

- Added `md.ft(...)`, a finetuning client for creating or resuming finetunes,
  generating rollouts, applying RL and SFT train steps, logging metrics, and
  managing checkpoints.
- Added typed finetuning request and response models under `md.types`, including
  rollout, training, metrics, checkpoint, ground-truth, RL group, and SFT group
  types.
- Added `rollout_stream(...)` for concurrent background rollout generation with
  context passthrough, so training can proceed while the next rollouts are in
  flight.
- Added `examples/train_rps_query.py`, an end-to-end rock/paper/scissors
  finetuning example.
- Hardened finetuning HTTP requests with retry/backoff handling for transient
  network and server errors.
- Changed the default finetune `train_step` learning rate to 2e-4.
- Upgraded the Photon local inference engine to v0.3.0, with local inference
  supported on NVIDIA GPUs and Apple Silicon Macs.

## 1.1.0

- Upgraded Photon engine to v0.2.1 — up to ~55% throughput improvement on H100
  and expanded GPU support (H200, GH200). See the
  [Photon changelog](https://github.com/m87-labs/photon/blob/main/CHANGELOG.md)
  for details.

## 0.2.0

- Added local GPU inference via the Photon backend (`local=True` on `md.vl()`).
