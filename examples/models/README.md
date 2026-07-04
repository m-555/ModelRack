# Models directory (`MODELS_DIR`)

**Your models go here.** This directory is the on-disk `MODELS_DIR` that modelrack
manages. Its contents are a **local, per-deployment working area** and are **not tracked
by git** (see `.gitignore`) — only this guide is committed. Point modelrack at it:

```bash
export MODELS_DIR=/path/to/examples/models     # or set per-machine
```

## Layout

Each **local model** is a subfolder named by its `model_id`:

```
<model_id>/
  config.yaml         # model definition: type, weights, server, defaults, param_schema
  server.py           # inference server (runs in the model's own venv)
  requirements.txt    # the model's Python deps
  weights/            # model weights (downloaded; git-ignored)
  loras/              # optional LoRA adapters (git-ignored)
```

**API models** (`backend: api`, e.g. Gemini/Claude) need only an optional `config.yaml`
(for `defaults`, `param_schema`, and the `api_key_env` reference) — no weights/server/venv.

Shared venvs live under `_shared_venvs/`, the managed HF cache under `.hf_cache/`, and
generated inference outputs under each model's `outputs/` — all git-ignored.

## Registry

`registry.yaml` is **generated** — it's the index modelrack keeps of the models in this
folder. Recreate it any time from what's on disk:

```bash
modelrack scan            # scan MODELS_DIR and (re)build registry.yaml
modelrack list            # list registered models
```

## Adding a model

- **Local:** create `<model_id>/` with `config.yaml` + `server.py` + `requirements.txt`,
  add weights under `weights/`, then `modelrack setup <model_id>` and `modelrack scan`.
- **API:** add a registry entry (or `<model_id>/config.yaml`) with `backend: api`,
  `provider:` and `api_model_id:`; set the provider's credential env var.

See the docs: `docs/adding-a-model.md`, `docs/adding-a-type.md`, `docs/config-schema.md`.
