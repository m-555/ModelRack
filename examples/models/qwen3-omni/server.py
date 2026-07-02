"""Qwen3-Omni (30B-A3B Instruct) any-to-any inference server — concrete implementation.

Runs inside this model's isolated .venv (created by `modelrack setup qwen3-omni`).
Depends on torch/transformers/qwen-omni-utils/soundfile, NOT installed in the hub env.

Only ``load_model()`` / ``run_inference()`` are model-specific; the FastAPI scaffolding
below the marker is identical to the modelrack omni template and should not be modified.

Request payload (merged config defaults + request params):
    messages           (list, required)  multimodal chat (text/image/audio/video content)
    max_tokens         (int)   default 1024   -> thinker_max_new_tokens
    return_audio       (bool)  default False  also generate speech (uses more VRAM)
    speaker            (str)   default "Ethan"  (Ethan | Chelsie | Aiden)
    use_audio_in_video (bool)  default True

Response data: {"text", "audio_base64"?, "sample_rate"?}
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
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

_AUDIO_SR = 24000


# ── Model-specific implementation ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load (model, processor) for Qwen3-Omni from the local weights directory."""
    weights = str(model_dir / "weights")
    serving = config.get("serving", {})
    attn = serving.get("attn_implementation")

    kwargs: dict[str, Any] = {"dtype": "auto", "device_map": "auto"}
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(weights, **kwargs)
    except (ImportError, ValueError) as exc:
        print(f"[qwen3-omni] '{attn}' unavailable ({exc}); retrying without it.", flush=True)
        kwargs.pop("attn_implementation", None)
        omni = Qwen3OmniMoeForConditionalGeneration.from_pretrained(weights, **kwargs)

    processor = Qwen3OmniMoeProcessor.from_pretrained(weights)
    return omni, processor


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one multimodal turn; return text and (optionally) base64 WAV speech."""
    if "messages" not in payload:
        raise ValueError("payload.messages is required")
    omni, processor = model
    conversation = payload["messages"]
    use_audio = bool(payload.get("use_audio_in_video", True))
    return_audio = bool(payload.get("return_audio", False))

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio,
    )
    inputs = inputs.to(omni.device).to(omni.dtype)

    text_ids, audio = omni.generate(
        **inputs,
        speaker=payload.get("speaker", "Ethan"),
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=int(payload.get("max_tokens", 1024)),
        return_audio=return_audio,
        use_audio_in_video=use_audio,
    )
    prompt_len = inputs["input_ids"].shape[1]
    out_text = processor.batch_decode(
        text_ids.sequences[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result: dict[str, Any] = {"text": out_text}
    if audio is not None:
        buf = io.BytesIO()
        sf.write(buf, audio.reshape(-1).detach().cpu().numpy(), samplerate=_AUDIO_SR, format="WAV")
        result["audio_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        result["sample_rate"] = _AUDIO_SR
    return result


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
        print(f"Loading Qwen3-Omni from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
