"""HTTP client that routes inference calls to the right running model server."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from modelrack.exceptions import InferenceError, ServerNotRunningError
from modelrack.process_manager import ProcessManager

logger = logging.getLogger("modelrack")


class InferenceClient:
    """Route ``infer`` / ``unload`` / ``info`` calls over HTTP to model servers."""

    def __init__(self, process_manager: ProcessManager) -> None:
        self.pm = process_manager

    def infer(
        self,
        model_id: str,
        payload: dict[str, Any],
        auto_start: bool = True,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """POST ``payload`` to the model's ``/infer`` endpoint and return the result.

        Starts the server first if it is not running and ``auto_start`` is True.
        """
        url = self._require_url(model_id, auto_start)
        try:
            resp = httpx.post(f"{url}/infer", json={"payload": payload}, timeout=timeout)
        except httpx.HTTPError as exc:
            raise InferenceError(f"Request to '{model_id}' failed: {exc}") from exc

        if resp.status_code != 200:
            raise InferenceError(
                f"Inference on '{model_id}' failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    def stream_infer(
        self,
        model_id: str,
        payload: dict[str, Any],
        auto_start: bool = True,
        timeout: int = 300,
    ) -> Iterator[str]:
        """Yield generated text chunks from the model's ``/infer_stream`` (SSE).

        Raises :class:`InferenceError` if the model server doesn't support streaming
        (HTTP 501) or errors mid-stream."""
        url = self._require_url(model_id, auto_start)
        try:
            with httpx.stream(
                "POST", f"{url}/infer_stream", json={"payload": payload}, timeout=timeout
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise InferenceError(
                        f"Streaming on '{model_id}' failed ({resp.status_code}): {resp.text}"
                    )
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        return
                    obj = json.loads(data)
                    if "error" in obj:
                        raise InferenceError(f"Streaming on '{model_id}' errored: {obj['error']}")
                    yield obj.get("text", "")
        except httpx.HTTPError as exc:
            raise InferenceError(f"Stream request to '{model_id}' failed: {exc}") from exc

    def unload(self, model_id: str) -> dict[str, Any]:
        """Free the model from VRAM without stopping its process."""
        url = self._require_url(model_id, auto_start=False)
        resp = httpx.post(f"{url}/unload", timeout=60)
        if resp.status_code != 200:
            raise InferenceError(f"Unload on '{model_id}' failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def info(self, model_id: str) -> dict[str, Any]:
        """Return the server's model info + VRAM usage."""
        url = self._require_url(model_id, auto_start=False)
        resp = httpx.get(f"{url}/info", timeout=30)
        if resp.status_code != 200:
            raise InferenceError(f"Info on '{model_id}' failed ({resp.status_code}): {resp.text}")
        return resp.json()

    # --- Internals ------------------------------------------------------------
    def _require_url(self, model_id: str, auto_start: bool) -> str:
        url = self.pm.get_server_url(model_id)
        if url is not None:
            return url
        if not auto_start:
            raise ServerNotRunningError(
                f"Server for '{model_id}' is not running (auto_start=False). "
                f"Start it with 'modelrack start {model_id}'."
            )
        return self.pm.start(model_id).url
