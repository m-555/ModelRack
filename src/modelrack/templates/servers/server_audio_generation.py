"""Text-to-audio (music / sound-effect) inference server template.

Runs inside the model's isolated .venv. Customize ``load_model()`` and
``run_inference()``; the FastAPI scaffolding below the marker is shared by every
modelrack server template and should not be modified.

Generated audio is written to ``<model_dir>/outputs/`` and the response carries
the **path**, not the bytes. A minute of 44.1 kHz stereo is ~10 MB of WAV, so
base64 in a JSON envelope is both slow and memory-hungry; every modelrack
generator that produces large media (video, audio) returns a path instead.
Callers that need bytes over a network boundary can set ``return_base64: true``.

Request payload (merged config defaults + request params):
    prompt          (str, required)  what to generate
    negative_prompt (str)            what to steer away from
    duration_s      (float)          length of the clip in seconds
    steps           (int)            diffusion steps
    cfg_scale       (float)          prompt adherence
    seed            (int)            -1 for random
    return_base64   (bool)           also inline the WAV as base64

Response data: {"output_path", "sample_rate", "duration_s", "channels",
                "seed", "audio_base64"?}

Endpoints: POST /infer · GET /health · POST /unload · GET /info
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Customize these two functions ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load and return the audio-generation model.

    Weights should be pre-downloaded into ``<model_dir>/weights``. A first-run
    fetch inside the server races the hub's ``startup_timeout_sec`` and shows up
    as an opaque 502 from the gateway.
    """
    raise NotImplementedError("Customize load_model() for your audio model")


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Generate audio, write a WAV under ``outputs/``, and return its path."""
    raise NotImplementedError("Customize run_inference() for your audio model")


# ── FastAPI scaffolding — do not modify below this line ───────────────────────

app = FastAPI(title="modelrack model server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model: Any = None
model_dir: Path = Path(".")
config: dict[str, Any] = {}
_loaded_at: float | None = None


class InferRequest(BaseModel):
    payload: dict[str, Any]


@app.post("/infer")
def infer(request: InferRequest) -> dict[str, Any]:
    global model, _loaded_at
    if model is None:
        model = load_model(model_dir, config)
        _loaded_at = time.time()
    try:
        merged = {**config.get("defaults", {}), **request.payload}
        return {"success": True, "data": run_inference(model, merged), "error": None}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_id": config.get("model_id"), "loaded": model is not None}


@app.post("/unload")
def unload() -> dict[str, Any]:
    global model, _loaded_at
    model = None
    _loaded_at = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"success": True, "data": {"unloaded": True}, "error": None}


@app.get("/info")
def info() -> dict[str, Any]:
    vram_used = vram_total = None
    if torch.cuda.is_available():
        vram_used = round(torch.cuda.memory_allocated() / 1e9, 2)
        vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return {
        "success": True,
        "data": {
            "model_id": config.get("model_id"),
            "type": config.get("type"),
            "loaded": model is not None,
            "loaded_since": _loaded_at,
            "vram_used_gb": vram_used,
            "vram_total_gb": vram_total,
        },
        "error": None,
    }


def _main() -> None:
    global model_dir, config, model, _loaded_at
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--lazy", action="store_true", help="Load weights on first request")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8")) or {}

    if not args.lazy:
        print(f"Loading model from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
