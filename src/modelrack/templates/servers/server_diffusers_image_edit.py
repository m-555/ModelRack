"""Image-editing inference server template (diffusers).

Runs inside the model's isolated .venv. Customize ``load_model()`` and
``run_inference()`` (e.g. Qwen-Image-Edit / QwenImageEditPlusPipeline). The payload
carries one or more input images as base64 PNG strings under ``images``.

Endpoints: POST /infer · GET /health · POST /unload · GET /info
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
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Customize these two functions ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load and return the image-edit pipeline.

    Example (Qwen-Image-Edit-2511):
        from diffusers import QwenImageEditPlusPipeline
        return QwenImageEditPlusPipeline.from_pretrained(
            str(model_dir / "weights"), torch_dtype=torch.bfloat16
        ).to("cuda")
    """
    raise NotImplementedError("Customize load_model() for your image-edit model")


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Edit the input image(s) per the prompt and return the result.

    Example:
        images = [_b64_png(b) for b in payload.get("images", [])]
        out = model(
            image=images,
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt", " "),
            num_inference_steps=payload.get("num_inference_steps", 40),
            true_cfg_scale=payload.get("true_cfg_scale", 4.0),
            num_images_per_prompt=payload.get("num_images_per_prompt", 1),
        ).images[0]
        return {"image_base64": _png_b64(out)}
    """
    raise NotImplementedError("Customize run_inference() for your image-edit model")


def _b64_png(data: str) -> Any:
    """Helper: decode a base64 PNG string into a PIL image."""
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _png_b64(image: Any) -> str:
    """Helper: encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
