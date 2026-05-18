# Changelog

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
