# Changelog

## 2.0.0

- Upgraded Photon local inference to `kestrel 0.5.0`. Moondream 2 and
  Moondream 3 now automatically use bundled whole-model decode kernels on
  supported NVIDIA GPUs, with the regular optimized path retained as a
  transparent fallback.
- Added local inference support for Moondream 3.1 9B A2B.
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
