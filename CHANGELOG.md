# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **LoRA adapters for diffusers models (`serving.loras`)** — load one or more LoRA
  files at startup, each targeting a chosen transformer with a blend weight. Notably
  enables **step-distillation LoRAs**: a diffusion model can then generate in ~4 steps at
  guidance 1.0 instead of ~40 steps with CFG (a large speedup) by pairing the LoRA with
  matching `defaults`. Requires the optional `peft` package. See docs/config-schema.md#lora-adapters.
- **On-load `fp8` quantization for diffusers models** — `serving.quantization: fp8`
  applies torchao float8 weight-only quantization to a model's heavy weights as they load
  (per-shard, so the full bf16 is never materialized in host RAM). This lets a large
  generative model that otherwise won't fit load and run on a commodity GPU (~half the
  size/host-RAM of bf16), at quality visually near-identical to bf16. Requires the optional
  `torchao` package in the model's environment. See docs/config-schema.md#quantization.
- **API models (`backend: api`)** — models served by a cloud provider now run
  **in-process** via a provider adapter (no subprocess, venv, or weights), routed by
  `ModelRack.infer` alongside local models and returning the same `{success, data, error}`
  envelope. Ships an **Anthropic (Claude)** provider; the layer is pluggable
  (`register_provider`) for OpenAI/Google/etc. Normalized `{messages, max_tokens, system}`
  surface plus a `provider_params` passthrough for provider-native features (extended
  thinking, tools, sampling). Credentials are **references** (`api_key_env`) resolved from
  the environment — never stored. Provider SDKs are optional extras
  (`pip install 'modelrack[anthropic]'`). See docs/config-schema.md#api-models.
- Reference server implementations for every engine (diffusers image/video/edit,
  transformers VLM/omni/TTS, vLLM LLM/code) with real `load_model()` / `run_inference()`,
  so a new model is a copy-and-fill-in away. `MODELS_DIR` is bring-your-own — model
  definitions and weights live in a local, per-deployment directory (not versioned).
- **Shared venvs** — models may declare `environment.shared_venv: <name>` to reuse one
  venv (at `<MODELS_DIR>/_shared_venvs/<name>`) across several compatible models instead
  of building one venv per model. `setup` installs each model's requirements into the
  shared venv and **warns on a Python-version mismatch**; `--force` reinstalls into a
  shared venv rather than deleting it. Lets you build heavy deps (e.g. torch) once and
  share them across models with compatible dependency stacks.
- **Custom package index for `setup`** — `environment.pip_extra_index_url` (str or list)
  and the machine-wide `MODELRACK_PIP_EXTRA_INDEX_URL` env var are passed to `uv pip
  install` as `--extra-index-url`, so a model's deps (e.g. a CUDA `torch` build) can be
  pulled from a non-PyPI wheel index. See docs/config-schema.md#gpu-specific-wheels.
- A zero-dependency **CPU smoke-test model** (no torch, no weights) that returns canned
  output, so the full hub → server → envelope infer path can be validated **without a
  GPU** before touching real models.
- README + `examples/models/README.md`: a "bring your own models" guide and a
  models-directory layout reference.

### Changed
- `POST /infer/{id}` now passes the model server's `{success, data, error}` envelope through
  as-is (no double-wrapping), matching the Python `hub.infer()` shape.
- `start` now falls back to a nearby free port when a model's configured port is
  unavailable — whether already in use or **OS-reserved** (e.g. Windows/Hyper-V excluded
  port ranges) — instead of failing. The hub tracks the actual port, so routing is
  unaffected.
- The example **TTS model** now exposes the full sampling controls in its `param_schema`
  and server — guidance weight, temperature, repetition penalty, min-p, top-p and seed
  (multilingual) — with sampling args guarded by the installed library's `generate()`
  signature.

### Fixed
- CI test collection under a bare `pytest` invocation (`pythonpath = ["."]`), so
  `from tests.conftest import ...` resolves on the runners.
- The multilingual TTS example server passed an unsupported keyword argument to the
  library's `from_pretrained()`; removed so it loads on the current release
  (verified on-GPU).
- The TTS example's requirements omitted `setuptools`, which its watermarker dependency
  imports for `pkg_resources` (removed from setuptools 81+ and absent from minimal uv
  venvs); pinned `setuptools<81` so a fresh setup loads (verified on-GPU).

## [0.1.0] — 2026-07-02

### Added
- Initial release of **modelrack**.
- `ModelRegistry` — atomic CRUD over `registry.yaml`, `scan_and_sync`.
- `ModelResolver` — 3-layer config merge (base → app overrides → runtime params) with
  deep-merge and param-schema validation.
- `ConfigValidator` — config / runtime-params / weights / venv validation.
- `ProcessManager` — per-model isolated `.venv` setup via `uv`, spawn/stop/health-check,
  crash-recovering process state.
- `InferenceClient` — HTTP routing of `infer` / `unload` / `info` to model servers.
- `ConfigWatcher` — debounced hot-reload of model configs.
- Open, registry-driven model-type system (`register_type`) with 8 built-in types.
- FastAPI hub management API (`modelrack serve`) with a uniform response envelope.
- Typer + Rich CLI covering the full model + process lifecycle.
- Server templates for diffusers (image/video/edit), TTS, transformers VLM, vLLM LLM,
  and omni models; matching base requirements files.
- Comprehensive example configs for 10 models across 8 types.
- Full test suite; ruff + mypy + pytest CI on Linux & Windows.
