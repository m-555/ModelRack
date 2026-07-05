"""FastAPI application factory for the hub management API.

All responses use a uniform envelope::

    {"success": true,  "data": {...}, "error": null}
    {"success": false, "data": null, "error": "message"}

Start it with ``modelrack serve`` (default port 7777, override via ``MODELRACK_PORT``).
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from modelrack import ModelRack, __version__
from modelrack.api.routes import infer, models, processes, system

logger = logging.getLogger("modelrack")


def create_app(models_dir: str | Path | None = None) -> FastAPI:
    """Build the FastAPI app, wiring a :class:`ModelRack` onto ``app.state``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        hub = ModelRack(models_dir)
        # Fresh start = clean GPU: kill any model servers orphaned by a previous
        # hub instance (they hold VRAM). Disable with MODELRACK_CLEAN_ON_START=0.
        if os.getenv("MODELRACK_CLEAN_ON_START", "1").strip().lower() not in ("0", "false", "no", "off"):
            hub.processes.reset()
        report = hub.scan()  # auto-sync registry on startup
        logger.info("Hub API up. scan_and_sync: %s", report)
        app.state.hub = hub
        yield
        hub.stop_watching()

    app = FastAPI(title="modelrack", version=__version__, lifespan=lifespan)

    # CORS: allow any localhost origin/port for local dev UIs.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"success": False, "data": None, "error": str(exc.errors())},
        )

    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(processes.router)
    app.include_router(infer.router)
    return app
