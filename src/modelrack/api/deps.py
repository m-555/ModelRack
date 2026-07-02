"""Shared helpers for the API layer: the hub singleton and response envelope."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from modelrack import ModelRack


def get_hub(request: Request) -> ModelRack:
    """Return the ModelRack instance stored on the app state."""
    return request.app.state.hub  # type: ignore[no-any-return]


def ok(data: Any) -> dict[str, Any]:
    """Success envelope."""
    return {"success": True, "data": data, "error": None}


def fail(message: str, status_code: int = 400) -> JSONResponse:
    """Error envelope as a JSONResponse with the given status code."""
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "data": None, "error": message},
    )
