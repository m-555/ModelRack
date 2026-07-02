"""Vision-language inference server template (transformers / vLLM).

Runs inside the model's isolated .venv. Customize ``load_model()`` and
``run_inference()`` (e.g. Qwen2.5-VL). Images/videos are passed in ``messages`` using
the standard multimodal chat content format (urls or base64 data URIs).

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
    """Load and return (model, processor) for a VLM.

    Example (Qwen2.5-VL via transformers):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        weights = str(model_dir / "weights")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            weights, torch_dtype="auto", device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(weights)
        return model, processor
    """
    raise NotImplementedError("Customize load_model() for your VLM")


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run a multimodal chat turn and return generated text.

    Example (Qwen2.5-VL):
        from qwen_vl_utils import process_vision_info
        vlm, processor = model
        messages = payload["messages"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(vlm.device)
        gen = vlm.generate(**inputs, max_new_tokens=payload.get("max_tokens", 512),
                           temperature=payload.get("temperature", 0.7),
                           top_p=payload.get("top_p", 0.8))
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return {"text": out}
    """
    raise NotImplementedError("Customize run_inference() for your VLM")


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
