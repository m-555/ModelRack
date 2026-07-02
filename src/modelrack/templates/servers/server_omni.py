"""Omni (any-to-any multimodal) inference server template.

Runs inside the model's isolated .venv. Customize ``load_model()`` and
``run_inference()`` (e.g. Qwen3-Omni-30B-A3B). Supports text/image/audio/video in and
text (+ optional speech) out.

Endpoints: POST /infer · GET /health · POST /unload · GET /info
"""

from __future__ import annotations

import argparse
import base64
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
    """Load and return (model, processor) for the omni model.

    Example (Qwen3-Omni):
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        weights = str(model_dir / "weights")
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            weights, dtype="auto", device_map="auto", attn_implementation="flash_attention_2",
        )
        # model.disable_talker()  # saves ~10GB VRAM if you only need text out
        processor = Qwen3OmniMoeProcessor.from_pretrained(weights)
        return model, processor
    """
    raise NotImplementedError("Customize load_model() for your omni model")


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run a multimodal turn; return text and (optionally) base64 WAV speech.

    Example (Qwen3-Omni):
        from qwen_omni_utils import process_mm_info
        omni, processor = model
        messages = payload["messages"]
        use_audio = payload.get("use_audio_in_video", True)
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                           return_tensors="pt", padding=True,
                           use_audio_in_video=use_audio).to(omni.device)
        text_ids, audio = omni.generate(
            **inputs, thinker_max_new_tokens=payload.get("max_tokens", 1024),
            speaker=payload.get("speaker", "Ethan"),
            return_audio=payload.get("return_audio", False),
            use_audio_in_video=use_audio,
        )
        out_text = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
        result = {"text": out_text}
        if audio is not None:
            import soundfile, io
            buf = io.BytesIO(); soundfile.write(buf, audio.cpu().numpy(), 24000, format="WAV")
            result["audio_base64"] = base64.b64encode(buf.getvalue()).decode()
        return result
    """
    raise NotImplementedError("Customize run_inference() for your omni model")


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
    global model
    if model is None:
        model = load_model(model_dir, config)
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
