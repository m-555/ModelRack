<h1 align="center">modelrack</h1>

<p align="center">
  <em>One source of truth for every generative-AI model you run — local or API.</em><br>
  Registry · 3-layer config pipeline · isolated-venv process manager · HTTP inference routing.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-orange">
</p>

---

`modelrack` is the control plane for your models. It knows what models exist, resolves
their configuration, and manages the per-model server processes that actually load
weights — **without ever importing `torch`, `diffusers` or `vllm` itself.**

### Why

- **Zero ML dependencies in the hub.** `modelrack` depends only on `pyyaml`, `pydantic`,
  `typer`, `fastapi`, `uvicorn`, `watchdog`, `httpx`, `python-dotenv`, `rich`. Your apps
  install `modelrack` and nothing heavy.
- **Every model is fully isolated.** Each model has its own `.venv`, its own pinned
  `requirements.txt`, its own server. Version conflicts between models are structurally
  impossible — or, when several models share a compatible dependency stack, they can opt
  into one **shared venv** (`environment.shared_venv`) to build heavy deps like torch
  once. See [config-schema](docs/config-schema.md#sharing-a-venv-across-models).
- **Config is layered.** `base config.yaml` → app overrides → runtime params. Later wins,
  deep-merged. One place to change a default; apps and UIs layer on top.
- **`param_schema` drives your UIs.** Every model publishes a schema of its editable
  parameters (type, range, options, label). Render settings panels dynamically — no
  hardcoding per model.
- **Open, extensible model types.** `video_generation`, `image_generation`, `image_edit`,
  `tts`, `vision_language`, `language`, `code`, `omni` ship built-in — and you can register
  new kinds without touching the core.
- **Local *and* API models, one interface.** Local models run as isolated subprocess
  servers; cloud API models (Anthropic, OpenAI, Google/Vertex, …) run **in-process** via
  provider adapters — same `hub.infer(id, payload)`, same `{success, data, error}`
  envelope. A normalized `messages` surface with a `provider_params` escape hatch for
  provider-native features; credentials are env-var references, never stored. See
  [config-schema](docs/config-schema.md#api-models-backend-api).

---

## Install

```bash
pip install modelrack            # from PyPI (once published)
pip install -e ".[dev]"          # from a clone, with dev tooling
```

`modelrack` shells out to [`uv`](https://astral.sh/uv) to build per-model venvs. Install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh              # Linux/macOS
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

## Quickstart (zero → first inference)

```bash
# 0. Point modelrack at your models directory (use the bundled examples to explore).
export MODELS_DIR=$(pwd)/examples/models        # PowerShell: $env:MODELS_DIR="...\examples\models"

# 1. See what's registered.
modelrack list

# 2. Inspect a model's fully-resolved config and its editable params.
modelrack show qwen-image
modelrack schema qwen-image

# 3. Create the model's isolated venv + install its pinned deps + drop in a server.py.
modelrack setup qwen-image

# 4. Run inference (auto-starts the server, routes over HTTP, returns the result).
modelrack infer qwen-image --payload '{"prompt": "a red fox in falling snow"}'
```

> **Bring your own models.** modelrack ships no weights or model definitions — your
> `MODELS_DIR` is a local, per-deployment working area. Each local model is a folder with a
> `config.yaml` + `server.py` + `requirements.txt` (+ `weights/`); API models
> (`backend: api`) need only a small `config.yaml`. See the layout guide in
> [examples/models/README.md](examples/models/README.md) and
> [docs/writing-a-server.md](docs/writing-a-server.md), then `modelrack scan` to register
> what's on disk.

### Adding models

modelrack is model-agnostic — it drives **diffusers** (image/video/edit), **transformers**
(VLM/omni/TTS), and **vLLM** (LLM/code) locals, plus in-process **API** providers
(Anthropic, Google, …). Server templates for each engine live in
[src/modelrack/templates/servers](src/modelrack/templates/servers); copy one, fill in
`load_model()` / `run_inference()`, and point `config.yaml` at your weights. Full walkthrough:
[docs/adding-a-model.md](docs/adding-a-model.md).

### Smoke-test without a GPU

The `echo` template (a zero-dependency CPU model — no torch, no weights) returns canned
output, so you can verify the full hub → server → envelope path on any machine before
touching real models: drop it into `MODELS_DIR`, then `modelrack setup echo` and
`modelrack infer echo --payload '{"text": "hi"}'`.

## Python API

```python
from modelrack import ModelRack

hub = ModelRack()                                  # reads MODELS_DIR from env
hub = ModelRack(models_dir="/data/models")         # or pass explicitly

# Resolve config (3-layer merge)
m = hub.resolve("qwen-image", runtime_params={"num_inference_steps": 30})
m = hub.resolve_from_app("qwen-image", "examples/app/model_overrides.yaml")

# Listing & schema (for building UIs)
hub.list(type="image_generation")
hub.schema("qwen-image")

# Validation
hub.validate("qwen-image")                         # -> ValidationResult(valid, errors, warnings)

# Process lifecycle
hub.setup("qwen-image"); hub.start("qwen-image")
hub.status(); hub.is_running("qwen-image")
hub.unload("qwen-image"); hub.stop("qwen-image")

# Inference (auto-starts if needed)
hub.infer("qwen-image", {"prompt": "a lighthouse at dawn"})

# Hot-reload configs
hub.watch(on_change=lambda mid, cfg: print(f"{mid} changed"))
```

## Using models in your app

There are two ways to consume modelrack from an application. Pick per app; you can mix.

| Mode | When | How |
|---|---|---|
| **In-process (Python)** | Your app is Python | `from modelrack import ModelRack` and call it directly. Servers run as child processes. |
| **REST** | Non-Python app (Node/React/Go…), or you want the hub on a separate box | Run `modelrack serve` and call the HTTP API. |

Either way the flow is the same: **resolve → (setup once) → infer**. `infer()` auto-starts the
model's server on first call and routes the request to it over HTTP.

### The inference envelope

Every `infer` result — from Python `hub.infer(...)` and from `POST /infer/{id}` — is the model
server's envelope:

```json
{ "success": true, "data": { /* model output, see table below */ }, "error": null }
```

Read your output at `result["data"]`. Errors raise `InferenceError` in Python, or return
`{"success": false, "error": "..."}` with a non-200 status over REST.

### Payloads & outputs by model type

`data` (the request payload) is merged over the model's `defaults`, so you only send what the
user changed. Outputs follow these conventions (implemented in each model's `server.py`):

| Type | Example models | Key payload fields | `data` output |
|---|---|---|---|
| `video_generation` | wan-2.2-i2v | `prompt`, `image` (path/URL/base64), `num_frames`, `fps`, `seed` | `{output_path, width, height, num_frames, fps}` |
| `tts` | qwen3-tts, chatterbox | `text`, `language`, `speaker`, `instruct` | `{audio_base64, sample_rate, encoding}` |
| `image_generation` | qwen-image, z-image-turbo | `prompt`, `negative_prompt`, `num_inference_steps`, `width`, `height` | `{image_base64}` |
| `image_edit` | qwen-image-edit | `prompt`, `images` (list of base64) | `{image_base64}` |
| `language` / `code` | qwen3.6, qwen3-coder | `messages` or `prompt`, `temperature`, `max_tokens` | `{text}` |
| `vision_language` / `omni` | qwen2.5-vl, qwen3-omni | `messages` (multimodal), sampling params | `{text}` (+ `audio_base64` for omni) |

> The full, per-model list of editable fields (types, ranges, options, labels) is the model's
> `param_schema` — fetch it with `hub.schema(id)` / `GET /models/{id}/schema` and render your
> settings UI from it (see [Dynamic settings UIs](#dynamic-settings-uis-from-param_schema)).

### Example — Qwen3-TTS (audio out)

**Python (in-process):**

```python
from modelrack import ModelRack
import base64

hub = ModelRack()
hub.setup("qwen3-tts")          # one-time: build the isolated venv + install deps

res = hub.infer("qwen3-tts", {
    "text": "Welcome back — your render is ready.",
    "language": "English",
    "speaker": "Ryan",
    "instruct": "Warm and upbeat.",
})
audio = base64.b64decode(res["data"]["audio_base64"])
open("welcome.wav", "wb").write(audio)     # sample_rate in res["data"]["sample_rate"]
```

**REST (any language):**

```bash
curl -s localhost:7777/infer/qwen3-tts -H 'content-type: application/json' -d '{
  "payload": {"text": "Welcome back!", "language": "English", "speaker": "Ryan"}
}' | jq -r '.data.audio_base64' | base64 -d > welcome.wav
```

### Example — WAN 2.2 image-to-video (video out)

```python
res = hub.infer("wan-2.2-i2v", {
    "prompt": "the camera slowly pushes in as snow starts to fall",
    "image": "https://example.com/first_frame.jpg",   # path, URL, or base64 data URI
    "num_frames": 81,
    "fps": 16,
    "seed": 42,
}, timeout=1200)                                        # video gen is slow — raise the timeout
video_path = res["data"]["output_path"]                # .mp4 written on the server host
```

> **Where outputs live:** WAN writes the `.mp4` to the model folder's `outputs/` on the *server
> host*. Because model servers run on the same machine as the hub, a same-host app reads
> `output_path` directly. To serve a remote frontend, either expose that folder statically or
> edit the model's `run_inference()` to return bytes/base64 instead of a path.

### Long-running apps

Start heavy models once and keep them warm instead of paying cold-start per request:

```python
hub.start("qwen3-tts")                  # blocks until healthy; stays up
# ... handle many requests ...
hub.infer("qwen3-tts", {...}, auto_start=False)   # fail fast if it isn't running
hub.unload("qwen3-tts")                 # free VRAM but keep the process
hub.stop("qwen3-tts")                   # shut it down
```

`hub.status()` / `GET /processes` report what's running (port, pid, uptime). Process state is
persisted, so it survives a hub restart.

### Dynamic settings UIs from `param_schema`

The whole point of `param_schema` is that your app **never hardcodes** a model's controls —
render them from the schema and send back only what changed as `runtime_params`:

```jsx
// React sketch: fetch the schema, render a control per parameter
const schema = (await fetch(`/models/${id}/schema`).then(r => r.json())).data;

return Object.entries(schema).map(([name, s]) =>
  s.options ? <Select label={s.label} options={s.options} />
  : s.type === "int" || s.type === "float"
      ? <Slider label={s.label} min={s.min} max={s.max} step={s.step} />
  : s.type === "bool" ? <Toggle label={s.label} />
  : <TextField label={s.label} placeholder={s.description} />
);
// ...then: POST /infer/{id} with { payload: { ...changedValues } }
```

## CLI reference

| Command | Purpose |
|---|---|
| `modelrack list [--type --backend --tags]` | List registered models |
| `modelrack show <id> [--runtime JSON]` | Show fully-resolved config |
| `modelrack schema <id>` | Show `param_schema` |
| `modelrack scan` | Sync registry with `MODELS_DIR` |
| `modelrack add / remove <id>` | Register / unregister (no file deletion) |
| `modelrack edit <id> [--server]` | Open `config.yaml` / `server.py` in `$EDITOR` |
| `modelrack validate <id> / --all` | Validate config + weights + venv |
| `modelrack setup <id> [--force]` | Create venv + install deps + copy server template |
| `modelrack start / stop / restart <id>` | Manage the inference server |
| `modelrack status [<id>]` | Show running servers |
| `modelrack unload <id>` | Free VRAM without stopping the server |
| `modelrack infer <id> --payload JSON` | Run inference |
| `modelrack serve [--port]` | Start the hub management REST API |

## REST API

`modelrack serve` (default port `7777`) exposes the hub to non-Python apps. Every response
uses the envelope `{"success": bool, "data": ..., "error": ...}`.

```
GET  /health · GET /system
GET  /models · GET /models/{id} · POST /models/{id}/resolve
GET  /models/{id}/schema · GET /models/{id}/validate · POST /models/scan
GET  /processes · GET /processes/{id}
POST /processes/{id}/{setup|start|stop|restart|unload}
POST /infer/{id}          body: {"payload": {...}, "auto_start": true, "timeout": 300}
POST /infer/{id}/stream   same body → Server-Sent Events (streamed text)
```

`POST /infer/{id}` passes the model server's `{success, data, error}` envelope straight
through (no extra wrapping), so its shape matches every other endpoint.
`POST /infer/{id}/stream` streams generated text as SSE — `data: {"text": "<chunk>"}`
per chunk, terminated by `data: [DONE]` (for models whose server implements streaming;
also available in Python via `hub.stream_infer(id, payload)`).

## How it fits together

```
        apps / UIs / Node frontends
                  │  (Python API or REST)
             ┌────▼─────┐
             │ modelrack │   registry · resolver · validator · process mgr · client
             └────┬─────┘   (ZERO ML deps)
      spawns &    │  routes HTTP
      health-checks│
   ┌──────────┬───┴────┬───────────┐
   ▼          ▼        ▼           ▼
 wan-2.2   qwen-image  qwen3.6   ...      each in its OWN .venv,
 :7801      :7803      :7809            loading weights, exposing
 (diffusers)(diffusers)(vLLM)          /infer /health /unload /info
```

## Docs

- [Config schema reference](docs/config-schema.md)
- [Adding a new model](docs/adding-a-model.md)
- [Adding a new model *type*](docs/adding-a-type.md)
- [Writing a `server.py`](docs/writing-a-server.md)

## Development

```bash
pip install -e ".[dev]"
ruff check .           # lint
ruff format .          # format
mypy src               # type-check
pytest                 # tests
```

## License

[MIT](LICENSE)
