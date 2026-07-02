"""Qwen3.6 (35B-A3B) LLM inference server — concrete implementation (vLLM).

Runs inside this model's isolated .venv (created by `modelrack setup qwen3.6`).
Depends on vllm/torch, which are NOT installed in the modelrack hub env.

Only ``load_model()`` / ``run_inference()`` are model-specific; the FastAPI scaffolding
below the marker is identical to the modelrack vLLM template and should not be modified.

Request payload (merged config defaults + request params) -- provide messages OR prompt:
    messages         (list)  chat messages [{role, content}]
    prompt           (str)   raw prompt (alternative to messages)
    enable_thinking  (bool)  default True   (chain-of-thought)
    temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, max_tokens, seed

Response data: {"text"}
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
from vllm import LLM, SamplingParams

# vLLM accepts these as `dtype`; anything else (e.g. "fp8") is a quantization, not a dtype.
_VALID_DTYPES = {"auto", "half", "float16", "bfloat16", "float", "float32"}


# ── Model-specific implementation ─────────────────────────────────────────────


def load_model(model_dir: Path, config: dict[str, Any]) -> Any:
    """Construct the vLLM engine from this model's serving config."""
    serving = config.get("serving", {})
    hardware = config.get("hardware", {})
    dtype = hardware.get("dtype", "bfloat16")
    if dtype not in _VALID_DTYPES:
        dtype = "auto"

    return LLM(
        model=str(model_dir / "weights"),
        tensor_parallel_size=int(serving.get("tensor_parallel_size", 1)),
        max_model_len=serving.get("max_model_len"),
        quantization=serving.get("quantization"),
        gpu_memory_utilization=float(serving.get("gpu_memory_utilization", 0.9)),
        dtype=dtype,
        trust_remote_code=True,
    )


def _sampling_params(payload: dict[str, Any]) -> SamplingParams:
    seed = int(payload.get("seed", -1))
    kwargs: dict[str, Any] = {
        "temperature": float(payload.get("temperature", 1.0)),
        "top_p": float(payload.get("top_p", 0.95)),
        "top_k": int(payload.get("top_k", 20)),
        "min_p": float(payload.get("min_p", 0.0)),
        "max_tokens": int(payload.get("max_tokens", 32768)),
        "presence_penalty": float(payload.get("presence_penalty", 1.5)),
        "repetition_penalty": float(payload.get("repetition_penalty", 1.0)),
    }
    if payload.get("stop"):
        kwargs["stop"] = payload["stop"]
    if seed >= 0:
        kwargs["seed"] = seed
    return SamplingParams(**kwargs)


def run_inference(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Generate a completion from `messages` (chat) or `prompt` (raw)."""
    sp = _sampling_params(payload)

    if "messages" in payload:
        chat_kwargs: dict[str, Any] = {}
        if "enable_thinking" in payload:
            chat_kwargs["chat_template_kwargs"] = {
                "enable_thinking": bool(payload["enable_thinking"])
            }
        outputs = model.chat(payload["messages"], sp, **chat_kwargs)
    elif "prompt" in payload:
        outputs = model.generate([payload["prompt"]], sp)
    else:
        raise ValueError("payload must contain either 'messages' or 'prompt'")

    return {"text": outputs[0].outputs[0].text}


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
        print(f"Loading Qwen3.6 (vLLM) from {model_dir} ...", flush=True)
        model = load_model(model_dir, config)
        _loaded_at = time.time()
        print("Model loaded.", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
