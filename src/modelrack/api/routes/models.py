"""/models endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from modelrack import ModelRack
from modelrack.api.deps import fail, get_hub, ok
from modelrack.exceptions import ModelRackError

router = APIRouter(prefix="/models", tags=["models"])


class ResolveBody(BaseModel):
    app_overrides: dict[str, Any] | None = None
    runtime_params: dict[str, Any] | None = None


@router.get("")
def list_models(
    hub: ModelRack = Depends(get_hub),
    type: str | None = Query(None),
    backend: str | None = Query(None),
    tags: list[str] | None = Query(None),
) -> dict[str, Any]:
    return ok(hub.list(type=type, backend=backend, tags=tags))


@router.post("/scan")
def scan(hub: ModelRack = Depends(get_hub)) -> dict[str, Any]:
    return ok(hub.scan())


@router.get("/{model_id}", response_model=None)
def get_model(model_id: str, hub: ModelRack = Depends(get_hub)) -> dict[str, Any] | JSONResponse:
    try:
        return ok(hub.resolve(model_id).to_dict())
    except ModelRackError as exc:
        return fail(str(exc), status_code=404)


@router.post("/{model_id}/resolve", response_model=None)
def resolve_model(
    model_id: str, body: ResolveBody, hub: ModelRack = Depends(get_hub)
) -> dict[str, Any] | JSONResponse:
    try:
        resolved = hub.resolve(
            model_id, app_overrides=body.app_overrides, runtime_params=body.runtime_params
        )
        return ok(resolved.to_dict())
    except ModelRackError as exc:
        return fail(str(exc), status_code=400)


@router.get("/{model_id}/schema", response_model=None)
def get_schema(model_id: str, hub: ModelRack = Depends(get_hub)) -> dict[str, Any] | JSONResponse:
    try:
        return ok(hub.schema(model_id))
    except ModelRackError as exc:
        return fail(str(exc), status_code=404)


@router.get("/{model_id}/validate", response_model=None)
def validate_model(
    model_id: str, hub: ModelRack = Depends(get_hub)
) -> dict[str, Any] | JSONResponse:
    try:
        return ok(hub.validate(model_id).to_dict())
    except ModelRackError as exc:
        return fail(str(exc), status_code=404)
