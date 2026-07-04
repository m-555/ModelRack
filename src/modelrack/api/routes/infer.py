"""/infer proxy endpoint."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from modelrack import ModelRack
from modelrack.api.deps import fail, get_hub
from modelrack.exceptions import ModelRackError

router = APIRouter(prefix="/infer", tags=["inference"])


class InferBody(BaseModel):
    payload: dict[str, Any]
    auto_start: bool = True
    timeout: int = 300


@router.post("/{model_id}", response_model=None)
def infer(
    model_id: str, body: InferBody, hub: ModelRack = Depends(get_hub)
) -> dict[str, Any] | JSONResponse:
    try:
        # The model server already returns a {success, data, error} envelope; pass it
        # through as-is so REST and Python callers see the same shape (no double-wrap).
        return hub.infer(model_id, body.payload, auto_start=body.auto_start, timeout=body.timeout)
    except ModelRackError as exc:
        return fail(str(exc), status_code=502)


@router.post("/{model_id}/stream", response_model=None)
def infer_stream(
    model_id: str, body: InferBody, hub: ModelRack = Depends(get_hub)
) -> StreamingResponse:
    """Stream generated text as Server-Sent Events: `data: {"text": "<chunk>"}` per
    chunk, terminated by `data: [DONE]`. Errors arrive as `data: {"error": "..."}`."""

    def _sse() -> Iterator[str]:
        try:
            for chunk in hub.stream_infer(
                model_id, body.payload, auto_start=body.auto_start, timeout=body.timeout
            ):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except ModelRackError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")
