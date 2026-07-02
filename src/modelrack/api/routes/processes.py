"""/processes endpoints (setup / start / stop / restart / unload / status)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from modelrack import ModelRack
from modelrack.api.deps import fail, get_hub, ok
from modelrack.exceptions import ModelRackError

router = APIRouter(prefix="/processes", tags=["processes"])


class SetupBody(BaseModel):
    force: bool = False


@router.get("")
def list_processes(hub: ModelRack = Depends(get_hub)) -> dict[str, Any]:
    return ok([p.to_dict() for p in hub.status()])


@router.get("/{model_id}")
def process_status(model_id: str, hub: ModelRack = Depends(get_hub)) -> dict[str, Any]:
    procs = hub.status(model_id)
    return ok(procs[0].to_dict() if procs else None)


@router.post("/{model_id}/setup", response_model=None)
def setup(
    model_id: str, body: SetupBody, hub: ModelRack = Depends(get_hub)
) -> dict[str, Any] | JSONResponse:
    try:
        hub.setup(model_id, force=body.force)
        return ok({"model_id": model_id, "setup": "complete"})
    except ModelRackError as exc:
        return fail(str(exc), status_code=400)


@router.post("/{model_id}/start", response_model=None)
def start(model_id: str, hub: ModelRack = Depends(get_hub)) -> dict[str, Any] | JSONResponse:
    try:
        return ok(hub.start(model_id).to_dict())
    except ModelRackError as exc:
        return fail(str(exc), status_code=400)


@router.post("/{model_id}/stop")
def stop(model_id: str, hub: ModelRack = Depends(get_hub)) -> dict[str, Any]:
    hub.stop(model_id)
    return ok({"model_id": model_id, "status": "stopped"})


@router.post("/{model_id}/restart", response_model=None)
def restart(model_id: str, hub: ModelRack = Depends(get_hub)) -> dict[str, Any] | JSONResponse:
    try:
        return ok(hub.restart(model_id).to_dict())
    except ModelRackError as exc:
        return fail(str(exc), status_code=400)


@router.post("/{model_id}/unload", response_model=None)
def unload(model_id: str, hub: ModelRack = Depends(get_hub)) -> dict[str, Any] | JSONResponse:
    try:
        return ok(hub.unload(model_id))
    except ModelRackError as exc:
        return fail(str(exc), status_code=400)
