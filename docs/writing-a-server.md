# Writing a `server.py`

Each local model runs a tiny FastAPI server **inside its own `.venv`**. `modelrack setup`
copies a template based on the model's `type`; you customize two functions.

## The contract

Every server must expose:

| Endpoint | Returns |
|---|---|
| `POST /infer` | `{"success": true, "data": {...}}` — runs inference on the JSON `payload` |
| `GET /health` | `{"status": "ok", "model_id": "...", "loaded": bool}` |
| `POST /unload` | Frees the model from VRAM without exiting the process |
| `GET /info` | Model info + VRAM usage |

`modelrack` starts the server as:

```
<venv-python> server.py --port <port> --model-dir <model_dir>
```

and waits for `GET /health` to return `200` before considering it up.

## What you customize

Only these two functions — the FastAPI scaffolding below them is complete and should not be
modified:

```python
def load_model(model_dir: Path, config: dict):
    """Load weights into VRAM, return the model/pipeline object."""

def run_inference(model, payload: dict) -> dict:
    """Run one request. `payload` = merged config defaults + request params."""
```

The scaffolding merges `config["defaults"]` with the request payload before calling
`run_inference`, so your function receives the fully-resolved parameters.

## Tips

- **Lazy loading:** pass `--lazy` to defer weight loading until the first `/infer` (the
  scaffolding also lazy-loads if the model is still `None`).
- **Return format:** images/audio are returned as base64 (see the image/tts templates); large
  video outputs are typically saved to disk with a returned `output_path`.
- **Isolation:** this file may import `torch`, `diffusers`, `vllm`, etc. freely — it never
  runs in the hub's environment.
- **VRAM:** implement `/unload` faithfully (`model = None; gc.collect(); torch.cuda.empty_cache()`)
  so `modelrack unload <id>` can reclaim memory without a restart.
