"""/health and /system endpoints."""

from __future__ import annotations

import shutil
from typing import Any

from fastapi import APIRouter, Depends

from modelrack import ModelRack, __version__
from modelrack.api.deps import get_hub, ok

router = APIRouter(tags=["system"])


@router.get("/health")
def health(hub: ModelRack = Depends(get_hub)) -> dict[str, Any]:
    return ok(
        {
            "status": "ok",
            "version": __version__,
            "models_dir": str(hub.models_dir),
            "running_servers": len(hub.status()),
        }
    )


@router.get("/system")
def system(hub: ModelRack = Depends(get_hub)) -> dict[str, Any]:
    usage = shutil.disk_usage(str(hub.models_dir))
    return ok(
        {
            "version": __version__,
            "models_dir": str(hub.models_dir),
            "total_models": len(hub.list()),
            "running_servers": len(hub.status()),
            "disk": {
                "total_gb": round(usage.total / 1e9, 1),
                "used_gb": round(usage.used / 1e9, 1),
                "free_gb": round(usage.free / 1e9, 1),
            },
        }
    )
