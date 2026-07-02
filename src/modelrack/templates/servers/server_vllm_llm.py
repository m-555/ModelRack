"""LLM / code-model inference server template (vLLM).

Runs inside the model's isolated .venv. Customize ``load_model()`` and
``run_inference()`` (e.g. Qwen3.6-35B-A3B, Qwen3-Coder). Uses vLLM offline inference;
serving params (tensor_parallel_size, max_model_len, quantization, dtype) come from the
config's ``serving`` block.

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
    """Load and return the vLLM engine.

    Example (Qwen3.6-35B-A3B):
        from vllm import LLM
        serving = config.get("serving", {})
        return LLM(
            model=str(model_dir / "weights"),
            tensor_parallel_size=serving.get("tensor_parallel_size", 1),
            max_model_len=serving.get("max_model_len", 32768),
            quantization=serving.get("quantization"),
            dtype=config.get("hardware", {}).get("dtype", "bfloat16"),
            gpu_memory_utilization=serving.get("gpu_memory_utilization", 0.9),
        )
    """
    raise NotImplementedError("Customize load_model() for your LLM")


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Generate a completion from ``messages`` (chat) or ``prompt`` (raw).

    Example:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=payload.get("temperature", 0.7),
            top_p=payload.get("top_p", 0.8),
            top_k=payload.get("top_k", 20),
            min_p=payload.get("min_p", 0.0),
            max_tokens=payload.get("max_tokens", 2048),
            presence_penalty=payload.get("presence_penalty", 0.0),
            repetition_penalty=payload.get("repetition_penalty", 1.0),
            stop=payload.get("stop"),
            seed=payload.get("seed"),
        )
        if "messages" in payload:
            outs = model.chat(payload["messages"], sp)
        else:
            outs = model.generate([payload["prompt"]], sp)
        return {"text": outs[0].outputs[0].text}
    """
    raise NotImplementedError("Customize run_inference() for your LLM")


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
