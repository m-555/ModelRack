"""Video-generation inference server template (diffusers).

Runs inside the model's isolated .venv — it is NOT imported by modelrack and may
freely depend on torch/diffusers. Customize ``load_model()`` and ``run_inference()``
for your specific model (e.g. WAN 2.2 I2V). Do not modify the FastAPI scaffolding.

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
    """Load the model into VRAM and return the pipeline/model object.

    Example (image-to-video):
        from diffusers import WanImageToVideoPipeline
        pipe = WanImageToVideoPipeline.from_pretrained(
            str(model_dir / "weights"), torch_dtype=torch.bfloat16
        ).to("cuda")
        return pipe

    Optional fp8 (serving.quantization: fp8) to fit smaller GPUs — quantize the heavy
    transformer(s) on load with torchao (per-shard, low peak host RAM):
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig
        q = TorchAoConfig(Float8WeightOnlyConfig())
        transformer = SomeTransformer.from_pretrained(
            str(model_dir / "weights"), subfolder="transformer",
            quantization_config=q, torch_dtype=torch.bfloat16,
        )
        pipe = SomePipeline.from_pretrained(
            str(model_dir / "weights"), transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
    """
    raise NotImplementedError("Customize load_model() for your video model")


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Run inference. ``payload`` is the merged config defaults + request params.

    Example (image-to-video):
        from diffusers.utils import export_to_video, load_image
        image = load_image(payload["image"])
        frames = model(
            image=image,
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt"),
            num_inference_steps=payload.get("num_inference_steps", 40),
            guidance_scale=payload.get("guidance_scale", 3.5),
            num_frames=payload.get("num_frames", 81),
            height=payload.get("height", 480),
            width=payload.get("width", 832),
        ).frames[0]
        out = "/tmp/output.mp4"
        export_to_video(frames, out, fps=payload.get("fps", 16))
        return {"output_path": out}
    """
    raise NotImplementedError("Customize run_inference() for your video model")


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
