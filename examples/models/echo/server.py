"""Echo server — zero-ML CPU stub for smoke-testing modelrack's infer path.

Runs inside the model's isolated .venv (only fastapi / uvicorn / pyyaml — no
torch). Returns canned data so the full hub -> server -> envelope round-trip can
be validated without a GPU or real weights.

Endpoints: POST /infer · GET /health · POST /unload · GET /info
Mirrors the contract of the real templates in modelrack/templates/servers/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="modelrack echo stub")

config: dict[str, Any] = {}
_loaded_at: float | None = None


class InferRequest(BaseModel):
    payload: dict[str, Any]


@app.post("/infer")
def infer(request: InferRequest) -> dict[str, Any]:
    try:
        merged = {**config.get("defaults", {}), **request.payload}
        text = str(merged.get("text", ""))
        prefix = str(merged.get("prefix", ""))
        return {
            "success": True,
            "data": {
                "echo": f"{prefix}{text}",
                "received": request.payload,
                "model_id": config.get("model_id"),
            },
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_id": config.get("model_id"), "loaded": True}


@app.post("/unload")
def unload() -> dict[str, Any]:
    return {"success": True, "data": {"unloaded": True}, "error": None}


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "success": True,
        "data": {
            "model_id": config.get("model_id"),
            "type": config.get("type"),
            "loaded": True,
            "loaded_since": _loaded_at,
            "vram_used_gb": None,
            "vram_total_gb": None,
        },
        "error": None,
    }


def _main() -> None:
    global config, _loaded_at
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--lazy", action="store_true", help="No-op (kept for template parity)")
    args = parser.parse_args()

    config = yaml.safe_load((Path(args.model_dir) / "config.yaml").read_text(encoding="utf-8")) or {}
    _loaded_at = time.time()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    _main()
