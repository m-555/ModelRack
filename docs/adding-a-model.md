# Adding a new model

A "model" is a folder inside `MODELS_DIR` containing a `config.yaml`. That's the only
requirement for it to be discovered.

## 1. Create the folder

```
$MODELS_DIR/
  my-model/
    config.yaml          # required
    requirements.txt     # pinned deps for this model's .venv
    weights/             # weight files (gitignored)
```

## 2. Write `config.yaml`

Start from the closest example in [`examples/models`](../examples/models) that matches your
model's type, then adjust. Fill in a **comprehensive `param_schema`** — this is what your
app UIs render. See the [config schema reference](config-schema.md).

Give it a unique `server.port` in the reserved range **7800–7899**.

## 3. Register it

```bash
modelrack scan            # auto-detects folders with a config.yaml
# or, explicitly:
modelrack add my-model --type image_generation --backend local
```

## 4. Set up and run

```bash
modelrack validate my-model    # config + weights + venv checks
modelrack setup my-model       # uv venv + install requirements + copy server template
modelrack start my-model
modelrack infer my-model --payload '{"prompt": "..."}'
```

## App-level overrides (layer 2)

Apps keep a `model_overrides.yaml` (see [`examples/app`](../examples/app)) to tweak defaults
without touching the base config:

```python
hub.resolve_from_app("my-model", "model_overrides.yaml",
                     runtime_params={"num_inference_steps": 40})
```

## API models

For remote API models, register with `--backend api` and set `provider` / `model_id`. No
`weights`, `server` or venv are needed; a `config.yaml` is optional (used only for
`defaults`/`param_schema`).

```bash
modelrack add claude --type language --backend api
```
