"""Image-segmentation / background-removal server template.

Runs inside the model's isolated .venv. Customize ``load_model()`` and
``run_inference()``; the FastAPI scaffolding below the marker is shared with the
other server templates and should not be modified.

Request payload:
    image_base64   (str)        a single input image, base64 PNG/JPEG
    images_base64  (list[str])  OR a batch of them
    output         (str)        "rgba" (cutout) | "mask" (8-bit matte)
    threshold      (float)      0.0 keeps the soft matte; >0 hardens the edge

Batching is first-class here on purpose. Segmentation is usually a POST-PROCESS
for another model's output, so a caller alternating generate/segment against a
single-residency gateway would evict and reload both models on every image.
Sending the whole set in one request keeps each model loaded once.

Response data:
    {"image_base64": ...}   for a single input
    {"images_base64": [..]} for a batch
    plus "width" / "height" (single) or "sizes" (batch)

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
    """Load and return the segmentation model.

    Example (a transformers-hosted matting model):
        from transformers import AutoModelForImageSegmentation
        m = AutoModelForImageSegmentation.from_pretrained(
            str(model_dir / "weights"), trust_remote_code=True
        )
        return m.to("cuda").eval()
    """
    raise NotImplementedError("Customize load_model() for your segmentation model")


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Produce a matte (or an RGBA cutout) for each input image.

    Use ``inputs_of(payload)`` to accept the single and batch forms uniformly,
    and ``pack(results, batched)`` to shape the response to match.
    """
    raise NotImplementedError("Customize run_inference() for your segmentation model")


# ── Shared helpers ────────────────────────────────────────────────────────────


def inputs_of(payload: dict[str, Any]) -> tuple[list[Any], bool]:
    """(images, was_batched) from either payload form. Raises if neither is set."""
    from PIL import Image

    raw = payload.get("images_base64")
    batched = raw is not None
    if not batched:
        one = payload.get("image_base64")
        if not one:
            raise ValueError("payload.image_base64 or payload.images_base64 is required")
        raw = [one]
    if not isinstance(raw, list) or not raw:
        raise ValueError("payload.images_base64 must be a non-empty list")

    images = [Image.open(io.BytesIO(base64.b64decode(item))).convert("RGB") for item in raw]
    return images, batched


def pack(results: list[Any], batched: bool) -> dict[str, Any]:
    """Shape the response to mirror the request form."""
    encoded = [png_b64(img) for img in results]
    if batched:
        return {
            "images_base64": encoded,
            "sizes": [{"width": im.width, "height": im.height} for im in results],
        }
    return {
        "image_base64": encoded[0],
        "width": results[0].width,
        "height": results[0].height,
    }


def png_b64(image: Any) -> str:
    """Encode a PIL image as a base64 PNG.

    Always PNG: a cutout's whole value is its alpha channel, and the lossy
    formats would either drop it or fringe the edge it exists to keep clean.
    """
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
        print(f"Loading segmentation model from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
