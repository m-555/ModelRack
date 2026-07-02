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
| `environment` | local | `python_version`, `requirements_file`, `venv_path`. |
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

## `serving` by engine

- **diffusers** (image/video/edit): `engine: diffusers`, `enable_model_cpu_offload`.
- **transformers** (VLM/omni/TTS): `engine: transformers`, `attn_implementation`,
  `tensor_parallel_size`.
- **vLLM** (LLM/code, large VLMs): `engine: vllm`, `tensor_parallel_size`, `max_model_len`,
  `gpu_memory_utilization`, `quantization` (e.g. `fp8`), `reasoning_parser`.
