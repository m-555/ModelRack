"""Qwen2.5-VL (72B Instruct) vision-language inference server — concrete implementation.

Runs inside this model's isolated .venv (created by `modelrack setup qwen2.5-vl`).
Depends on torch/transformers/qwen-vl-utils, NOT installed in the modelrack hub env.

Only ``load_model()`` / ``run_inference()`` are model-specific; the FastAPI scaffolding
below the marker is identical to the modelrack VLM template and should not be modified.

Request payload (merged config defaults + request params):
    messages     (list, required)  chat messages with multimodal content (image/video/text),
                                    using the standard Qwen-VL content format (URLs / data URIs)
    max_tokens   (int)   default 512    -> max_new_tokens
    temperature  (float) default 0.7
    top_p        (float) default 0.8
    top_k        (int)   default 20
    repetition_penalty (float) default 1.05

Response data: {"text"}

NOTE: for the 72B model, vLLM is usually preferable (see the model's serving config). This
server uses transformers with device_map="auto", which works but needs multi-GPU for 72B.
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
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ── Model-specific implementation ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load (model, processor) for Qwen2.5-VL from the local weights directory."""
    weights = str(model_dir / "weights")
    serving = config.get("serving", {})
    attn = serving.get("attn_implementation")

    kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(weights, **kwargs)
    except (ImportError, ValueError) as exc:
        print(f"[qwen2.5-vl] '{attn}' unavailable ({exc}); retrying without it.", flush=True)
        kwargs.pop("attn_implementation", None)
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(weights, **kwargs)

    processor = AutoProcessor.from_pretrained(weights)
    return vlm, processor


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one multimodal chat turn and return the generated text."""
    if "messages" not in payload:
        raise ValueError("payload.messages is required")
    vlm, processor = model
    messages = payload["messages"]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm.device)

    do_sample = float(payload.get("temperature", 0.7)) > 0
    generated_ids = vlm.generate(
        **inputs,
        max_new_tokens=int(payload.get("max_tokens", 512)),
        do_sample=do_sample,
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.8)),
        top_k=int(payload.get("top_k", 20)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.05)),
    )
    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids, strict=False)]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {"text": output_text}


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
        print(f"Loading Qwen2.5-VL from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
