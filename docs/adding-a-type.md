# Adding a new model *type*

`modelrack` treats model types as an **open registry**, not a fixed enum. New kinds
(embeddings, rerankers, ASR, music, ...) plug in without changing the core.

## What a type provides

A type maps to two defaults:

- a **server template** (`templates/servers/server_<...>.py`) copied into a model's folder
  on `modelrack setup`,
- a **base requirements file** (`templates/requirements/<...>.txt`).

Built-in types: `video_generation`, `image_generation`, `image_edit`, `tts`,
`vision_language`, `language`, `code`, `omni`.

## Option A — register at runtime

```python
from modelrack import register_type

register_type(
    "embedding",
    template="server_embedding.py",       # must exist in templates/servers/
    requirements="embedding.txt",         # must exist in templates/requirements/
    description="Text embedding model",
)
```

Now any model with `type: embedding` resolves, validates (no "unknown type" warning), and
`modelrack setup` copies the right template.

## Option B — contribute it to the package

1. Add `src/modelrack/templates/servers/server_embedding.py` (self-contained; it runs in an
   isolated venv, so it may import any ML library). Keep the FastAPI scaffolding identical to
   the existing templates — only customize `load_model()` / `run_inference()`.
2. Add `src/modelrack/templates/requirements/embedding.txt`.
3. Register it in `src/modelrack/schemas/model_types.py` by adding an entry to
   `_TYPE_REGISTRY`.

Nothing in the registry, resolver, or process manager needs to change.

## Unknown types still work

An unregistered type is never fatal — it produces a validator *warning* and simply has no
default template/requirements. Provide a `server.py` in the model folder yourself and it
runs. This keeps experimentation friction-free.
