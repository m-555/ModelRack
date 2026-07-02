"""Qwen3-TTS 12Hz (Custom Voice) inference server — concrete implementation.

Runs inside this model's isolated .venv (created by `modelrack setup qwen3-tts`).
It depends on torch + the `qwen_tts` package, which are NOT installed in the modelrack
hub env.

Only ``load_model()`` / ``run_inference()`` are model-specific; the FastAPI scaffolding
below the marker is identical to the modelrack TTS template and should not be modified.

Request payload (merged config defaults + request params):
    text      (str, required)  text to synthesize
    language  (str)            "Auto" | "Chinese" | "English" | ...  (default "Auto")
    speaker   (str)            one of the 9 premium voices (default "Vivian")
    instruct  (str)            natural-language prosody/emotion control (optional)

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

import soundfile as sf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ── Model-specific implementation ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load the Qwen3-TTS custom-voice model from the local weights directory."""
    hardware = config.get("hardware", {})
    serving = config.get("serving", {})
    dtype = _DTYPES.get(hardware.get("dtype", "bfloat16"), torch.bfloat16)

    device = hardware.get("device", "cuda")
    device_map = "cuda:0" if device == "cuda" else device
    weights = str(model_dir / "weights")
    attn = serving.get("attn_implementation", "flash_attention_2")

    try:
        return Qwen3TTSModel.from_pretrained(
            weights, device_map=device_map, dtype=dtype, attn_implementation=attn
        )
    except (ImportError, ValueError) as exc:
        # flash-attn may be unavailable on this machine; fall back to the portable path.
        print(f"[qwen3-tts] '{attn}' unavailable ({exc}); falling back to sdpa.", flush=True)
        return Qwen3TTSModel.from_pretrained(
            weights, device_map=device_map, dtype=dtype, attn_implementation="sdpa"
        )


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Synthesize speech for a single utterance; return base64-encoded WAV audio."""
    if "text" not in payload or not payload["text"]:
        raise ValueError("payload.text is required")

    wavs, sample_rate = model.generate_custom_voice(
        text=payload["text"],
        language=payload.get("language", "Auto"),
        speaker=payload.get("speaker", "Vivian"),
        instruct=payload.get("instruct", ""),
    )
    # generate_custom_voice returns a list of waveforms (one per input utterance).
    waveform = wavs[0]

    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, format="WAV")
    return {
        "audio_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "sample_rate": int(sample_rate),
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
        print(f"Loading Qwen3-TTS from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
