# Config schema reference

Each local model lives in `<MODELS_DIR>/<model_id>/` and is described by a `config.yaml`.
This is the **base layer** — sacred, never written by apps. Apps layer overrides on top
(see [adding-a-model](adding-a-model.md)); UIs layer runtime params on top of that.

## Top-level sections

| Section | Required | Purpose |
|---|---|---|
| `model_id` | ✅ | Unique id (matches the folder name). |
| `type` | ✅ | Model kind (open set — see [adding-a-type](adding-a-type.md)). |
| `display_name` | | Human-friendly name for UIs. |
| `backend` | | `local` (default) or `api`. |
| `version`, `tags` | | Metadata; `tags` power `modelrack list --tags`. |
| `weights` | local | Map of role → path relative to the model folder. `main` is required and validated; others are optional. For diffusers/transformers models, point `main` at the `weights/` directory. |
| `hardware` | | `min_vram_gb`, `recommended_vram_gb`, `dtype`, `device`. |
| `server` | local | `port` (unique, reserved range 7800–7899), `host`, `startup_timeout_sec`, `endpoints`. |
| `environment` | local | `python_version`, `requirements_file`, `venv_path`, optional `shared_venv` (see [Sharing a venv](#sharing-a-venv-across-models)), optional `pip_extra_index_url` (see [GPU-specific wheels](#gpu-specific-wheels-eg-pytorch-cuda)). |
| `defaults` | | Default runtime parameter values (the base of the 3-layer merge). |
| `param_schema` | | UI-renderable description of every editable parameter. |
| `load_hints` | | Framework/pipeline/class + HF repo — hints for `server.py`. |
| `serving` | | Engine-specific serving config (diffusers/transformers/vLLM). |
| `meta` | | License, source URL, notes. |

## `param_schema` entries

Each entry describes one editable parameter so apps can render a control for it:

```yaml
param_schema:
  num_inference_steps:
    type: int            # int | float | bool | str | list
    min: 1               # numeric lower bound (optional)
    max: 100             # numeric upper bound (optional)
    step: 1              # UI step (optional)
    options: [512, 768]  # enumerated choices (optional; validated)
    label: "Steps"       # UI display name
    description: "..."    # help text (optional)
```

Runtime params are validated against this schema on `resolve(...)`: type match, `min`/`max`
range, and `options` membership. Unknown params pass through (forward-compatible with app
UIs), but `validate_runtime_params` will warn about them.

## The three layers

```
base config.yaml   →   app overrides (model_overrides.yaml)   →   runtime UI params
   (defaults)              (per-app tweaks)                          (per-request)
                    later layers win · nested dicts deep-merged
```

`param_schema`, `load_hints` and `weights` come from the base config only — overrides apply
to `defaults`, `hardware`, `serving`, etc.

## Sharing a venv across models

By default each model gets its own isolated `.venv` (`environment.venv_path`, default
`.venv`, inside the model folder). To reuse **one** venv across several models with
compatible dependency stacks — building heavy deps like torch only once — set:

```yaml
environment:
  python_version: "3.11"
  requirements_file: requirements.txt
  shared_venv: torch-cuda        # -> <MODELS_DIR>/_shared_venvs/torch-cuda
```

Every model that names the same `shared_venv` reuses that one environment. `modelrack
setup <id>` creates the shared venv on first use and installs each model's requirements
into it (union of requirements); `--force` reinstalls into it rather than deleting it
(so peers aren't broken). Shared venvs live under `<MODELS_DIR>/_shared_venvs/` and are
git-ignored.

**Caveats — sharing only works when the stacks are compatible.** A shared venv holds one
version of each package, so models needing conflicting versions (e.g. different pinned
`transformers`/`diffusers`, or a forked library) must use **separate** shared venvs.
`setup` warns when an existing shared venv's Python differs from a model's
`python_version`; hard dependency conflicts surface at install time. When in doubt, group
models by dependency stack (one shared venv per compatible group) rather than forcing all
into one.

## GPU-specific wheels (e.g. PyTorch CUDA)

`modelrack setup` installs a model's `requirements.txt` from PyPI by default — which on
many platforms resolves `torch` to the **CPU** build. For a GPU that needs a specific CUDA
build, point `setup` at an extra wheel index:

- **Machine-wide** (recommended — keeps the choice out of portable configs):
  ```bash
  export MODELRACK_PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
  modelrack setup <id>
  ```
- **Per-model** (in `config.yaml`, when a model genuinely needs its own index):
  ```yaml
  environment:
    pip_extra_index_url: https://download.pytorch.org/whl/cu128   # str or list
  ```

Both are passed to `uv pip install` as `--extra-index-url`. To *pin* a CUDA build you still
need the local version in requirements (e.g. `torch==2.8.0+cu128`), which is machine-specific
— so for a shared GPU stack the cleanest recipe is: **build the CUDA `torch` once into a
[shared venv](#sharing-a-venv-across-models), then `setup` each model into it** (the extra
index/pin lives in that one build, and models install the rest of their deps on top).

## API models (`backend: api`)

Not every model is local. An **API model** is served by a cloud provider (Anthropic,
OpenAI, Google/Vertex, …) — modelrack calls it **in-process** (no subprocess, no venv,
no weights) via a provider adapter. Declare it in `registry.yaml`:

```yaml
models:
  claude-opus:
    type: language
    backend: api
    provider: anthropic          # selects the provider adapter
    api_model_id: claude-opus-4-8 # the provider's own model id
    tags: [language, claude]
```

A `config.yaml` is **optional** for API models — add one only for `defaults` and a
`param_schema` (to drive UIs); there are no `weights`, `server`, or `environment` fields.

**Credentials — references, never secrets.** A model declares the *name* of the env var
holding its key; modelrack never stores the key itself:

```yaml
# in the API model's config.yaml (optional)
api_key_env: ANTHROPIC_API_KEY   # default per provider; the value lives in the environment
```

If the env var is unset, the provider's SDK falls back to its own credential chain (env
vars, CLI login profiles, workload identity) — so the gateway holds provider keys
server-side while callers only ever hold a modelrack key.

**Normalized payload + native escape hatch.** For `language` / `vision_language` models
the normalized request is `{messages, max_tokens, system?}`; provider-native features
(e.g. extended thinking, tools, sampling params) go under `provider_params` and pass
straight through to the provider SDK. The response is normalized to
`{text, model, stop_reason, usage}`.

Install the provider SDK as an extra: `pip install 'modelrack[anthropic]'` (or
`modelrack[api]`).

## Quantization (`serving.quantization`)

Large generative models can be **quantized on load** so they fit smaller GPUs/hosts:

```yaml
serving:
  engine: diffusers
  enable_model_cpu_offload: true
  quantization: fp8      # torchao float8 weight-only
```

`fp8` stores the heavy weights in 8-bit (≈ half the size/host-RAM of bf16) and up-casts
for compute — visually near-identical to bf16 for diffusion. Quantization is applied
**per shard at load time**, so peak host memory stays low (the full bf16 is never
materialized), letting a model that otherwise won't load run on a commodity GPU. Requires
the optional `torchao` package in the model's environment (add it to `requirements.txt`).
Weights on disk stay in their original precision; the server quantizes each time it starts.

## LoRA adapters (`serving.loras`)

Diffusers models can load one or more LoRA adapters at startup:

```yaml
serving:
  loras:
    - path: loras/<name>/high_noise_model.safetensors
      target: transformer       # or transformer_2 (multi-transformer models)
      weight: 1.0
```

Each entry names a LoRA file (relative to the model folder), which transformer it
attaches to, and its blend `weight`. A common use is a **step-distillation LoRA**, which
lets a diffusion model produce a good result in ~4 steps at guidance 1.0 instead of ~40
steps with CFG — a large speedup. Pair it with matching `defaults`
(`num_inference_steps`, `guidance_scale`). Requires the optional `peft` package in the
model's environment.

## `serving` by engine

- **diffusers** (image/video/edit): `engine: diffusers`, `enable_model_cpu_offload`,
  optional `quantization` (see above).
- **transformers** (VLM/omni/TTS): `engine: transformers`, `attn_implementation`,
  `tensor_parallel_size`.
- **vLLM** (LLM/code, large VLMs): `engine: vllm`, `tensor_parallel_size`, `max_model_len`,
  `gpu_memory_utilization`, `quantization` (e.g. `fp8`), `reasoning_parser`.
