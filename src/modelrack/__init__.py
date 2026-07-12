"""modelrack — centralized model registry, config pipeline & process manager.

Typical use::

    from modelrack import ModelRack

    hub = ModelRack()                       # reads MODELS_DIR from env
    hub = ModelRack(models_dir="/models")   # or pass explicitly

    model = hub.resolve("qwen-image", runtime_params={"num_inference_steps": 30})
    hub.setup("qwen-image")
    result = hub.infer("qwen-image", {"prompt": "a red fox in snow"})
"""

from __future__ import annotations

import builtins
import logging
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from modelrack import exceptions
from modelrack.config import configure_logging, load_settings
from modelrack.exceptions import ModelRackError
from modelrack.inference_client import InferenceClient
from modelrack.process_manager import ProcessManager
from modelrack.registry import ModelRegistry
from modelrack.resolver import ModelResolver
from modelrack.schemas import (
    Backend,
    ResolvedModel,
    ServerProcess,
    ValidationResult,
    register_type,
)
from modelrack.validator import ConfigValidator
from modelrack.watcher import ConfigWatcher

__version__ = "0.1.0"

logger = logging.getLogger("modelrack")

__all__ = [
    "ModelRack",
    "ModelRegistry",
    "ModelResolver",
    "ConfigValidator",
    "ProcessManager",
    "InferenceClient",
    "ConfigWatcher",
    "ResolvedModel",
    "ServerProcess",
    "ValidationResult",
    "Backend",
    "register_type",
    "exceptions",
    "ModelRackError",
    "__version__",
]


class ModelRack:
    """The single public entry point wiring registry, resolver, validator,
    process manager and inference client together."""

    def __init__(self, models_dir: str | Path | None = None) -> None:
        self.settings = load_settings(models_dir)
        configure_logging(self.settings.log_level)
        self.models_dir = self.settings.require_models_dir()

        self.registry = ModelRegistry(self.models_dir)
        self.resolver = ModelResolver(self.models_dir, self.registry)
        self.validator = ConfigValidator()
        self.processes = ProcessManager(
            self.models_dir,
            state_file=self.settings.state_file,
            resolver=self.resolver,
        )
        self.client = InferenceClient(self.processes)
        self._watcher: ConfigWatcher | None = None

        # VRAM policy: keep at most this many models resident on the GPU at once.
        # Before inferring model X, the VRAM of every other-than-the-N-most-recent
        # model is freed (its subprocess stays alive for a fast reload). Default 1
        # — a single shared GPU can't host several large models at once (OOM/BSOD).
        # Set MODELRACK_MAX_RESIDENT=0 to disable eviction (keep everything loaded).
        try:
            self._max_resident = int(os.getenv("MODELRACK_MAX_RESIDENT", "1"))
        except ValueError:
            self._max_resident = 1
        # LRU-ish recency of models actually infer'd this session.
        self._recent: list[str] = []

    # --- Config resolution ----------------------------------------------------
    def resolve(
        self,
        model_id: str,
        app_overrides: dict[str, Any] | None = None,
        runtime_params: dict[str, Any] | None = None,
    ) -> ResolvedModel:
        return self.resolver.resolve(model_id, app_overrides, runtime_params)

    def resolve_from_app(
        self,
        model_id: str,
        app_overrides_path: str | Path,
        runtime_params: dict[str, Any] | None = None,
    ) -> ResolvedModel:
        return self.resolver.resolve_from_app(model_id, app_overrides_path, runtime_params)

    # --- Listing / schema -----------------------------------------------------
    # NB: this class defines a method named ``list``, which shadows the builtin
    # inside the class body — so annotations use ``builtins.list`` explicitly.
    def list(
        self,
        type: str | None = None,
        backend: str | None = None,
        tags: builtins.list[str] | None = None,
    ) -> builtins.list[dict[str, Any]]:
        return self.registry.list_models(type=type, backend=backend, tags=tags)

    def schema(self, model_id: str) -> dict[str, Any]:
        return self.resolver.get_param_schema(model_id)

    def scan(self) -> dict[str, builtins.list[str]]:
        return self.registry.scan_and_sync()

    # --- Validation -----------------------------------------------------------
    def validate(self, model_id: str, check_venv: bool = True) -> ValidationResult:
        resolved = self.resolver.resolve(model_id)
        if check_venv:
            return self.validator.validate_all(resolved)
        result = self.validator.validate_model_config(resolved.merged_config)
        return result.merge(self.validator.validate_weight_paths(resolved))

    # --- Process management ---------------------------------------------------
    def setup(self, model_id: str, force: bool = False) -> None:
        self.processes.setup(model_id, force=force)

    def start(self, model_id: str) -> ServerProcess:
        return self.processes.start(model_id)

    def stop(self, model_id: str, graceful: bool = True) -> None:
        self.processes.stop(model_id, graceful=graceful)

    def restart(self, model_id: str) -> ServerProcess:
        return self.processes.restart(model_id)

    def status(self, model_id: str | None = None) -> builtins.list[ServerProcess]:
        return self.processes.status(model_id)

    def is_running(self, model_id: str) -> bool:
        return self.processes.is_running(model_id)

    # --- Inference ------------------------------------------------------------
    def infer(
        self,
        model_id: str,
        payload: dict[str, Any],
        auto_start: bool = True,
        timeout: int = 300,
    ) -> dict[str, Any]:
        # API models run in-process via a provider adapter; local models route over
        # HTTP to their subprocess server. Both return the {success,data,error} envelope.
        if self._is_api_model(model_id):
            return self._api_infer(model_id, payload)
        self._evict_for(model_id)
        return self.client.infer(model_id, payload, auto_start=auto_start, timeout=timeout)

    def stream_infer(
        self,
        model_id: str,
        payload: dict[str, Any],
        auto_start: bool = True,
        timeout: int = 300,
    ) -> Iterator[str]:
        """Yield generated text chunks. Local models stream token-by-token from their
        ``/infer_stream`` endpoint; API models yield the full result once (provider
        streaming is a future addition)."""
        if self._is_api_model(model_id):
            data = self._api_infer(model_id, payload).get("data") or {}
            yield (data.get("text") or "") if isinstance(data, dict) else str(data)
            return
        self._evict_for(model_id)
        yield from self.client.stream_infer(
            model_id, payload, auto_start=auto_start, timeout=timeout
        )

    def _evict_for(self, model_id: str) -> None:
        """Single-GPU VRAM guard: fully free every resident model except the
        ``_max_resident`` most-recently-used (plus the incoming ``model_id``).

        Eviction **STOPS** the other model servers (kills the subprocess) rather
        than calling their ``/unload``. A plain ``/unload`` is unreliable — e.g.
        Fish runs its LLM in a background thread that keeps the model on the GPU,
        so ``model = None`` frees nothing and its ~23 GB stays resident. Killing
        the process is the only way to guarantee the VRAM (weights + KV cache +
        CUDA context) is reclaimed on a tight card. The model respawns on its
        next infer (weights reload; torch.compile hits the on-disk inductor
        cache, so recompiles are cheap). No-op when MODELRACK_MAX_RESIDENT=0.

        Set MODELRACK_EVICT_MODE=unload to keep the old (subprocess-alive) behaviour.
        """
        # Update recency (most-recent last).
        if model_id in self._recent:
            self._recent.remove(model_id)
        self._recent.append(model_id)

        if self._max_resident <= 0:
            return

        stop_mode = os.getenv("MODELRACK_EVICT_MODE", "stop").strip().lower() != "unload"
        keep = set(self._recent[-self._max_resident:]) | {model_id}
        stopped_pids: list[int] = []
        freed_any = False
        for rec in self.processes.status():
            if rec.model_id not in keep:
                try:
                    if stop_mode:
                        pid = rec.pid
                        self.processes.stop(rec.model_id, graceful=False)
                        if pid:
                            stopped_pids.append(pid)
                    else:
                        self.client.unload(rec.model_id)
                    freed_any = True
                    logger.warning("VRAM guard: %s %s to make room for %s",
                                   "stopped" if stop_mode else "unloaded",
                                   rec.model_id, model_id)
                except Exception as exc:  # noqa: BLE001 - never block an infer on cleanup
                    logger.warning("VRAM guard: failed to free %s: %s", rec.model_id, exc)

        # Barrier: the incoming model must NOT start loading until the evicted
        # model's VRAM is actually back. Killing a process that holds a CUDA
        # context frees its VRAM ASYNCHRONOUSLY (esp. Windows/WDDM) — without
        # this wait the next model measures too little free VRAM at load time
        # and spills layers to CPU, then streams weights over PCIe every step
        # (Bus Interface ~100 %, GPU compute stalled, render crawls at 0 %).
        if freed_any:
            self._await_vram_settle(stopped_pids, incoming=model_id)

    def _await_vram_settle(self, pids: list[int], incoming: str = "") -> None:
        """Block until VRAM freed by just-evicted model servers is reclaimed by
        the driver, before the next model loads.

        Two stages: (1) wait for the killed PIDs to actually exit, then (2) poll
        free VRAM until it stops climbing (reclamation done), plus a minimum
        settle. Falls back to a plain settle sleep when ``nvidia-smi`` is not
        available (non-NVIDIA host / not on PATH), so it is always safe.

        Tunables (env): ``MODELRACK_EVICT_WAIT_S`` (max total wait, default 20),
        ``MODELRACK_EVICT_SETTLE_S`` (min settle / no-nvidia-smi fallback,
        default 2.0)."""
        from modelrack.utils.procstate import pid_alive

        max_wait = float(os.getenv("MODELRACK_EVICT_WAIT_S", "20"))
        settle = float(os.getenv("MODELRACK_EVICT_SETTLE_S", "2.0"))
        deadline = time.monotonic() + max_wait

        # 1. Wait for the killed processes to actually exit (they still hold VRAM
        #    until they do).
        while time.monotonic() < deadline and any(pid_alive(p) for p in pids):
            time.sleep(0.2)

        # 2. Wait for the driver to finish reclaiming: free VRAM rises as the
        #    context is torn down, then plateaus. Break on two stable reads.
        prev = self._gpu_free_mib()
        if prev is None:
            time.sleep(settle)
            return
        stable = 0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            cur = self._gpu_free_mib()
            if cur is None:
                break
            if abs(cur - prev) < 256:  # < 256 MiB change = reclamation settled
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev = cur
        logger.info("VRAM guard: %d MiB free before loading %s",
                    self._gpu_free_mib() or -1, incoming or "next model")
        time.sleep(settle)

    @staticmethod
    def _gpu_free_mib() -> int | None:
        """Free VRAM on GPU 0 in MiB via ``nvidia-smi``; None if unavailable."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None
            return int(out.stdout.strip().splitlines()[0].strip())
        except Exception:  # noqa: BLE001 - nvidia-smi absent/unparseable -> caller falls back
            return None

    def _is_api_model(self, model_id: str) -> bool:
        try:
            return self.registry.get_model_entry(model_id).get("backend") == "api"
        except Exception:  # noqa: BLE001 - unknown/unregistered -> treat as local
            return False

    def _api_infer(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        from modelrack.providers import get_provider

        resolved = self.resolver.resolve(model_id)
        provider = get_provider(resolved.provider)
        merged = {**resolved.merged_config.get("defaults", {}), **payload}
        data = provider.infer(resolved, merged)
        return {"success": True, "data": data, "error": None}

    def unload(self, model_id: str) -> dict[str, Any]:
        return self.client.unload(model_id)

    def info(self, model_id: str) -> dict[str, Any]:
        return self.client.info(model_id)

    # --- Hot reload -----------------------------------------------------------
    def watch(self, on_change: Callable[[str, dict[str, Any]], None]) -> ConfigWatcher:
        watcher = ConfigWatcher(self.models_dir, on_change)
        watcher.start()
        self._watcher = watcher
        return watcher

    def stop_watching(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None


logging.getLogger("modelrack").addHandler(logging.NullHandler())
