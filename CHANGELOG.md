# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`audio_generation` model type** — text-to-audio generation (music, sound
  effects) is now a first-class type with its own server template
  (`templates/servers/server_audio_generation.py`) and base requirements, so
  such a model plugs in the same way an image or video model does. It is
  deliberately **separate from `tts`**: the two share nothing but a waveform
  output — different inputs, different parameters, different server contract.
  The template returns a generated file **by path** rather than base64, matching
  the video template; a minute of 44.1 kHz stereo is ~10 MB, which is not
  something to put through a JSON envelope. New `tests/test_model_types.py`
  additionally asserts that *every* registered type resolves to templates that
  actually exist on disk — a missing template previously surfaced only at
  `modelrack setup` time, long after the mistake.
- **stable-audio-3 example model** — 2B latent-diffusion text-to-audio
  generator (port 7871) producing both music beds and sound effects from one
  checkpoint. `/infer` takes `prompt` + `negative_prompt`, `duration_s`,
  `steps`, `cfg_scale`, `sampler_type` and `seed`, and returns
  `{output_path, sample_rate, duration_s, channels, seed}`. The server drives
  whichever upstream inference library is installed and loads strictly from a
  pre-downloaded `weights/` directory, rewriting the checkpoint's bundled
  text-encoder reference to the local copy — otherwise loading reaches for the
  network mid-startup. Output is peak-normalised before the WAV is written
  (diffusion output is unbounded and clips loudly otherwise), and the generated
  length is set from the requested duration rather than the config's full
  window. Uses its own by-kind shared venv (`audio-cu128`) rather than joining
  the diffusers venv, so the image/video models co-tenanting it cannot inherit a
  dependency downgrade. GPU-verified: 20 s of audio in 2.3 s at 9.34 GB peak.

- **`environment.pip_index_strategy`** — sets uv's `--index-strategy` for a
  `setup` install (`first-index` | `unsafe-first-match` | `unsafe-best-match`),
  with a machine-wide `MODELRACK_PIP_INDEX_STRATEGY` default that the per-model
  value overrides. An unrecognised value is warned about and ignored rather than
  forwarded — a typo should not abort a setup.

  This closes a trap in the existing `pip_extra_index_url` support. uv defaults
  to `first-index`: once an index publishes a package *at all*, only that index
  is consulted for it — the protection against dependency confusion. But
  accelerator wheel indexes also republish ordinary packages (`packaging`,
  `setuptools`, …) at pinned older versions, so adding one could make an
  unrelated dependency unresolvable, and **uv's error names that dependency
  rather than the index**, sending you to look in entirely the wrong place.
  Without an extra index configured, nothing changes.

  `docs/config-schema.md` now also states the other half of this, which is not
  obvious and fails silently: **name the accelerator package explicitly in
  `requirements.txt`**. Pulled in only as some other package's transitive
  dependency, it resolves from PyPI — the CPU build on Windows — and the model
  then runs entirely on the processor, producing correct output at a fraction of
  the speed with nothing anywhere reporting a problem.
- **qwen3-tts-clone example model** — Qwen3-TTS-12Hz-1.7B-**Base** voice-clone
  counterpart to the CustomVoice model (port 7811, Apache-2.0). `/infer` takes
  `text` + `ref_audio_path` (absolute path on the shared filesystem, same
  convention as chatterbox/fish) + optional `ref_text`; with a transcript it
  uses high-fidelity ICL cloning and **auto-falls back to x-vector-only mode
  when the transcript is empty** (the qwen_tts package would otherwise raise).
  The reference clip is capped to the leading `serving.max_reference_seconds`
  (15 s), and `seed >= 0` gives reproducible sampling via `torch.manual_seed`.
  Shares one venv with qwen3-tts via `environment.shared_venv: qwen-tts`
  (identical dep stacks — torch built once, ~4.7 GB saved). Verified on-GPU
  (ICL clone + x-vector fallback + process-manager start/stop).
- **Streaming inference (SSE)** — `POST /infer/{id}/stream` and `ModelRack.stream_infer()`
  yield generated text token-by-token as Server-Sent Events (`data: {"text": ...}` per
  chunk, terminated by `data: [DONE]`). Local model servers opt in by implementing a
  `run_inference_stream` generator (exposed as the server's `/infer_stream`); the
  transformers VLM/LLM reference server ships it via `TextIteratorStreamer`. API models
  yield the full result once for now (provider token-streaming is a future addition).
- **Google (Gemini) API provider** — `backend: api`, `provider: google`. Supports Vertex
  AI (Application Default Credentials via `GOOGLE_APPLICATION_CREDENTIALS`; non-secret
  `project`/`location`) and AI Studio API keys. Optional extra: `modelrack[google]`.
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
- **qwen3-tts server dropped its sampling params** — `temperature`, `top_p`,
  `max_new_tokens` and `seed` were declared in `config.yaml` but never passed to
  `generate_custom_voice`, so they silently had no effect. Now forwarded
  (verified against `qwen_tts._merge_generate_kwargs`: explicit values win),
  with `seed >= 0` applied via `torch.manual_seed`.
- **qwen3-tts / qwen3-tts-clone setup installed CPU-only torch on Windows** —
  plain PyPI serves `+cpu` wheels; both models now pin `torch==2.11.0+cu128`
  with the PyTorch cu128 index via `environment.pip_extra_index_url` and a
  matching `--extra-index-url` line in their requirements files.
- **qwen3-omni port collision** — its configured port 7808 was also fish-s2's;
  moved to 7812.
- **`validate` ignored `environment.shared_venv`** — `ConfigValidator.validate_venv`
  checked only the per-model `.venv` path, so a model on a shared venv always
  warned "venv not found" despite being fully set up. Now resolves through
  `resolve_venv_exists` (shared-venv aware), like `setup`/`start` already did.
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
