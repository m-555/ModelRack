"""WAN 2.2 (Image-to-Video, A14B) inference server — concrete implementation.

Runs inside this model's isolated .venv (created by `modelrack setup wan-2.2-i2v`).
It depends on torch/diffusers, which are NOT installed in the modelrack hub env.

Only ``load_model()`` / ``run_inference()`` are model-specific; the FastAPI scaffolding
below the marker is identical to the modelrack video template and should not be modified.

Request payload (merged config defaults + request params):
    prompt            (str, required)  text describing the motion/scene
    image             (str, required)  input image: local path, URL, or base64 (data URI ok)
    negative_prompt   (str)
    num_inference_steps (int)          default 40
    guidance_scale    (float)          default 3.5
    num_frames        (int)            default 81 (~5s @ 16fps)
    width, height     (int)            used as the max-area budget; final size is derived
    fps               (int)            default 16
    seed              (int)            -1 for random

Response data: {"output_path", "width", "height", "num_frames", "fps"}
"""

from __future__ import annotations

import argparse
import base64
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
import yaml
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ── Model-specific implementation ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Load the WAN 2.2 I2V pipeline from the local weights directory."""
    hardware = config.get("hardware", {})
    serving = config.get("serving", {})
    dtype = _DTYPES.get(hardware.get("dtype", "bfloat16"), torch.bfloat16)
    device = hardware.get("device", "cuda")

    pipe = WanImageToVideoPipeline.from_pretrained(str(model_dir / "weights"), torch_dtype=dtype)

    # A14B is large; CPU offload lets it run on <80GB GPUs (config: serving.enable_model_cpu_offload).
    if serving.get("enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Animate the input image into a video and save it under the model's outputs/ dir."""
    if "prompt" not in payload:
        raise ValueError("payload.prompt is required")
    if "image" not in payload:
        raise ValueError("payload.image is required (path, URL, or base64)")

    image = _load_input_image(payload["image"])

    # Derive the generation size from a max-area budget while preserving aspect ratio,
    # snapping to the pipeline's spatial/patch grid (per the model card).
    max_area = int(payload.get("width", 832)) * int(payload.get("height", 480))
    aspect_ratio = image.height / image.width
    mod_value = model.vae_scale_factor_spatial * model.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))

    generator = None
    seed = int(payload.get("seed", -1))
    if seed >= 0:
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    negative_prompt = payload.get("negative_prompt") or None
    frames = model(
        image=image,
        prompt=payload["prompt"],
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=int(payload.get("num_frames", 81)),
        guidance_scale=float(payload.get("guidance_scale", 3.5)),
        num_inference_steps=int(payload.get("num_inference_steps", 40)),
        generator=generator,
    ).frames[0]

    fps = int(payload.get("fps", 16))
    out_dir = model_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"i2v_{int(time.time())}_{seed if seed >= 0 else 'rand'}.mp4"
    export_to_video(frames, str(out_path), fps=fps)

    return {
        "output_path": str(out_path),
        "width": width,
        "height": height,
        "num_frames": len(frames),
        "fps": fps,
    }


def _load_input_image(src: str) -> Any:
    """Accept a local path, http(s) URL, or (data-URI/raw) base64 string -> PIL image."""
    from PIL import Image

    if src.startswith(("http://", "https://")) or Path(src).exists():
        return load_image(src).convert("RGB")
    # base64, optionally as a data URI (data:image/png;base64,....)
    b64 = src.split(",", 1)[1] if src.startswith("data:") else src
    import io

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


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
        print(f"Loading WAN 2.2 I2V from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
