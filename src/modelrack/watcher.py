"""File watcher that hot-reloads model configs on change.

Watches every ``<models_dir>/*/config.yaml`` and invokes a callback (debounced) when
one changes, so long-running apps can pick up config edits without a restart.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("modelrack")

OnChange = Callable[[str, dict[str, Any]], None]


class _ConfigEventHandler(FileSystemEventHandler):
    def __init__(self, models_dir: Path, on_change: OnChange, debounce_ms: int) -> None:
        self.models_dir = models_dir
        self.on_change = on_change
        self.debounce = debounce_ms / 1000.0
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def on_modified(self, event: FileSystemEvent) -> None:
        self._maybe_fire(event)

    def on_created(self, event: FileSystemEvent) -> None:
        self._maybe_fire(event)

    def _maybe_fire(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.name != "config.yaml":
            return
        try:
            model_id = path.parent.relative_to(self.models_dir).parts[0]
        except (ValueError, IndexError):
            return
        self._debounced(model_id, path)

    def _debounced(self, model_id: str, path: Path) -> None:
        with self._lock:
            existing = self._timers.get(model_id)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(self.debounce, self._dispatch, args=(model_id, path))
            timer.daemon = True
            self._timers[model_id] = timer
            timer.start()

    def _dispatch(self, model_id: str, path: Path) -> None:
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Failed to reload %s config: %s", model_id, exc)
            return
        try:
            self.on_change(model_id, config)
        except Exception:  # noqa: BLE001 - never let a callback kill the watcher
            logger.exception("on_change callback raised for %s", model_id)


class ConfigWatcher:
    """Background watcher for model ``config.yaml`` files."""

    def __init__(
        self,
        models_dir: Path,
        on_change: OnChange,
        debounce_ms: int = 500,
    ) -> None:
        self.models_dir = Path(models_dir)
        self._handler = _ConfigEventHandler(self.models_dir, on_change, debounce_ms)
        self._observer: Observer | None = None  # type: ignore[valid-type]

    def start(self) -> None:
        if self._observer is not None:
            return
        observer = Observer()
        observer.schedule(self._handler, str(self.models_dir), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("Config watcher started on %s", self.models_dir)

    def stop(self) -> None:
        if self._observer is None:
            return
        self._observer.stop()
        self._observer.join(timeout=5)
        self._observer = None
        logger.info("Config watcher stopped")
