# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete, working `server.py` for **all 10 example models** (previously only
  `wan-2.2-i2v` and `qwen3-tts`): `z-image-turbo`, `qwen-image`, `qwen-image-edit`,
  `chatterbox`, `qwen2.5-vl`, `qwen3-omni`, `qwen3.6`, `qwen3-coder`. Each `load_model()`/
  `run_inference()` is grounded in the model's own card (diffusers / transformers / vLLM / TTS).
- README: "Included example models" table and a "Using models in your app" guide.

### Changed
- `POST /infer/{id}` now passes the model server's `{success, data, error}` envelope through
  as-is (no double-wrapping), matching the Python `hub.infer()` shape.

### Fixed
- CI test collection under a bare `pytest` invocation (`pythonpath = ["."]`), so
  `from tests.conftest import ...` resolves on the runners.

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
