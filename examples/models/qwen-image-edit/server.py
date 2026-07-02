"""Qwen-Image-Edit (2511) image-editing inference server — concrete implementation.

Runs inside this model's isolated .venv (created by `modelrack setup qwen-image-edit`).
Depends on torch/diffusers, which are NOT installed in the modelrack hub env.

Only ``load_model()`` / ``run_inference()`` are model-specific; the FastAPI scaffolding
below the marker is identical to the modelrack image template and should not be modified.

Request payload (merged config defaults + request params):
    prompt                (str, required)   the edit instruction
    images                (list[str], required)  one or more base64 PNG input images
    negative_prompt       (str)   default " "
    num_inference_steps   (int)   default 40
    true_cfg_scale        (float) default 4.0
    guidance_scale        (float) default 1.0
    num_images_per_prompt (int)   default 1
    seed                  (int)   -1 for random

Response data: {"image_base64"}  (base64-encoded PNG)
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
from diffusers import QwenImageEditPlusPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ── Model-specific implementation ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load the Qwen-Image-Edit pipeline from the local weights directory."""
    hardware = config.get("hardware", {})
    serving = config.get("serving", {})
    dtype = _DTYPES.get(hardware.get("dtype", "bfloat16"), torch.bfloat16)
    device = hardware.get("device", "cuda")

    pipe = QwenImageEditPlusPipeline.from_pretrained(str(model_dir / "weights"), torch_dtype=dtype)
    if serving.get("enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Edit the input image(s) per the prompt; return a base64 PNG."""
    if "prompt" not in payload:
        raise ValueError("payload.prompt is required")
    raw_images = payload.get("images") or []
    if not raw_images:
        raise ValueError("payload.images must contain at least one base64 image")
    images = [_b64_png(b) for b in raw_images]

    seed = int(payload.get("seed", -1))
    generator = torch.manual_seed(seed) if seed >= 0 else None

    inputs = {
        "image": images,
        "prompt": payload["prompt"],
        "negative_prompt": payload.get("negative_prompt", " "),
        "num_inference_steps": int(payload.get("num_inference_steps", 40)),
        "true_cfg_scale": float(payload.get("true_cfg_scale", 4.0)),
        "guidance_scale": float(payload.get("guidance_scale", 1.0)),
        "num_images_per_prompt": int(payload.get("num_images_per_prompt", 1)),
        "generator": generator,
    }
    with torch.inference_mode():
        output_image = model(**inputs).images[0]

    return {"image_base64": _png_b64(output_image)}


def _b64_png(data: str) -> Any:
    from PIL import Image

    raw = data.split(",", 1)[1] if data.startswith("data:") else data
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def _png_b64(image: Any) -> str:
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
        print(f"Loading Qwen-Image-Edit from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
