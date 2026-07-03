"""Chatterbox (multilingual TTS) inference server — concrete implementation.

Runs inside this model's isolated .venv (created by `modelrack setup chatterbox`).
Depends on torch + the `chatterbox-tts` package, which are NOT installed in the hub env.

Only ``load_model()`` / ``run_inference()`` are model-specific; the FastAPI scaffolding
below the marker is identical to the modelrack TTS template and should not be modified.

Request payload (merged config defaults + request params):
    text               (str, required)
    language_id        (str)   default "en"   (23 languages)
    audio_prompt_path  (str)   path to a .wav reference for zero-shot voice cloning
    exaggeration       (float) default 0.5    expressiveness
    cfg_weight         (float) default 0.5    guidance weight
    temperature        (float) default 0.8    sampling temperature
    repetition_penalty (float) default 1.2
    min_p              (float) default 0.05
    top_p              (float) default 1.0
    seed               (int)   default -1     (-1 = random)

Sampling controls (temperature/min_p/top_p) are passed only when the installed
`chatterbox-tts` build's generate() accepts them, so the server stays compatible
across versions.

Response data: {"audio_base64", "sample_rate", "encoding": "wav"}
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio as ta
import uvicorn
import yaml
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Model-specific implementation ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load the Chatterbox multilingual TTS model.

    Chatterbox fetches/caches its own weights via from_pretrained(device=...); no local
    weight path is required (this model's config declares empty `weights`).
    """
    device = config.get("hardware", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Synthesize speech and return base64 WAV audio."""
    if "text" not in payload or not payload["text"]:
        raise ValueError("payload.text is required")

    import inspect

    # Pass only the tuning args that are present, to stay compatible across versions.
    kwargs: dict[str, Any] = {"language_id": payload.get("language_id", "en")}
    if payload.get("audio_prompt_path"):
        kwargs["audio_prompt_path"] = payload["audio_prompt_path"]
    if "exaggeration" in payload:
        kwargs["exaggeration"] = float(payload["exaggeration"])
    if "cfg_weight" in payload:
        kwargs["cfg_weight"] = float(payload["cfg_weight"])
    if "repetition_penalty" in payload:
        kwargs["repetition_penalty"] = float(payload["repetition_penalty"])

    # Sampling controls — only pass those the installed build's generate() accepts.
    _sig = inspect.signature(model.generate).parameters
    for _name in ("temperature", "min_p", "top_p"):
        if _name in payload and _name in _sig:
            kwargs[_name] = float(payload[_name])

    seed = int(payload.get("seed", -1))
    if seed >= 0:
        torch.manual_seed(seed)

    wav = model.generate(payload["text"], **kwargs)

    buf = io.BytesIO()
    ta.save(buf, wav, model.sr, format="wav")
    return {
        "audio_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "sample_rate": int(model.sr),
        "encoding": "wav",
    }


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
        print(f"Loading Chatterbox from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
