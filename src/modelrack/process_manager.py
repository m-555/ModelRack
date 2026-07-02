"""Lifecycle management for per-model inference-server subprocesses.

Each local model runs its own FastAPI server inside its own isolated ``.venv``. This
module creates those venvs (via ``uv``), spawns/stops the servers, health-checks them,
and persists their state so running servers survive a hub restart.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from datetime import datetime
from importlib import resources
from pathlib import Path

import httpx

from modelrack.exceptions import (
    PortConflictError,
    ServerStartupError,
    SetupNotCompletedError,
    UvNotFoundError,
)
from modelrack.registry import ModelRegistry
from modelrack.resolver import ModelResolver
from modelrack.schemas.model_types import template_for
from modelrack.schemas.resolved_model import ResolvedModel
from modelrack.schemas.server_process import ServerProcess
from modelrack.utils.paths import venv_dir, venv_exists, venv_python
from modelrack.utils.ports import find_free_port, is_port_free
from modelrack.utils.procstate import ProcessStateStore, pid_alive

logger = logging.getLogger("modelrack")

UV_INSTALL_HINT = (
    "uv not found on PATH. Install it with:\n"
    "  curl -LsSf https://astral.sh/uv/install.sh | sh   (Linux/macOS)\n"
    '  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   (Windows)'
)


class ProcessManager:
    """Spawn, stop and monitor per-model inference servers."""

    def __init__(
        self,
        models_dir: Path,
        state_file: str | Path = "~/.modelrack/processes.json",
        resolver: ModelResolver | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.registry = ModelRegistry(self.models_dir)
        self.resolver = resolver or ModelResolver(self.models_dir, self.registry)
        self.state = ProcessStateStore(state_file)
        self._processes: dict[str, ServerProcess] = self.state.load()
        # Popen handles for servers started in *this* session (best for termination).
        self._popen: dict[str, subprocess.Popen[bytes]] = {}

    # --- Setup ----------------------------------------------------------------
    def setup(self, model_id: str, force: bool = False) -> None:
        """Create the model's .venv, install requirements, and copy its server template."""
        resolved = self.resolver.resolve(model_id)
        model_dir = self.models_dir / model_id
        env = resolved.environment or {}
        venv_rel = env.get("venv_path", ".venv")
        python_version = str(env.get("python_version", "3.11"))
        requirements = model_dir / env.get("requirements_file", "requirements.txt")

        uv = shutil.which("uv")
        if uv is None:
            raise UvNotFoundError(UV_INSTALL_HINT)

        vdir = venv_dir(model_dir, venv_rel)
        if force and vdir.exists():
            logger.info("Removing existing venv %s", vdir)
            shutil.rmtree(vdir)

        if not venv_exists(model_dir, venv_rel):
            logger.info("Creating venv for %s (python %s)", model_id, python_version)
            self._run([uv, "venv", "--python", python_version, str(vdir)])

        if requirements.exists():
            logger.info("Installing requirements for %s", model_id)
            self._run(
                [
                    uv,
                    "pip",
                    "install",
                    "-r",
                    str(requirements),
                    "--python",
                    str(venv_python(model_dir, venv_rel)),
                ]
            )
        else:
            logger.warning("No requirements.txt for %s at %s", model_id, requirements)

        self._ensure_server_file(resolved, model_dir)

        if self.registry.exists(model_id):
            self.registry.update_model(model_id, setup_complete=True)

    def _ensure_server_file(self, resolved: ResolvedModel, model_dir: Path) -> None:
        """Copy the type's server template into the model folder if absent."""
        server_file = model_dir / "server.py"
        if server_file.exists():
            return
        template_name = template_for(resolved.type)
        if template_name is None:
            logger.warning(
                "No server template registered for type '%s'; write server.py manually.",
                resolved.type,
            )
            return
        template_src = resources.files("modelrack.templates.servers").joinpath(template_name)
        server_file.write_text(template_src.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Copied template %s -> %s", template_name, server_file)

    # --- Start / stop ---------------------------------------------------------
    def start(self, model_id: str) -> ServerProcess:
        """Start the model's inference server and block until healthy."""
        if self.is_running(model_id):
            return self._processes[model_id]

        resolved = self.resolver.resolve(model_id)
        model_dir = self.models_dir / model_id
        env = resolved.environment or {}
        venv_rel = env.get("venv_path", ".venv")

        if not venv_exists(model_dir, venv_rel):
            raise SetupNotCompletedError(
                f"'{model_id}' is not set up. Run 'modelrack setup {model_id}' first."
            )

        server_py = model_dir / "server.py"
        if not server_py.exists():
            raise SetupNotCompletedError(f"server.py missing for '{model_id}': {server_py}")

        server_cfg = resolved.merged_config.get("server", {})
        host = server_cfg.get("host", "127.0.0.1")
        port = resolved.server_port or 7800
        timeout = int(server_cfg.get("startup_timeout_sec", 120))

        if not is_port_free(port, host):
            raise PortConflictError(f"Port {port} for '{model_id}' is already in use on {host}.")

        python = venv_python(model_dir, venv_rel)
        cmd = [
            str(python),
            str(server_py),
            "--port",
            str(port),
            "--model-dir",
            str(model_dir),
        ]
        logger.info("Starting %s: %s", model_id, " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=str(model_dir))  # noqa: S603
        self._popen[model_id] = proc

        url = f"http://{host}:{port}"
        record = ServerProcess(
            model_id=model_id,
            port=port,
            pid=proc.pid,
            status="starting",
            started_at=datetime.now(),
            url=url,
        )
        self._processes[model_id] = record
        self._persist()

        if not self._health_check(url, timeout):
            self.stop(model_id, graceful=False)
            raise ServerStartupError(
                f"'{model_id}' failed health check within {timeout}s (see server logs)."
            )

        record.status = "running"
        self._persist()
        return record

    def stop(self, model_id: str, graceful: bool = True) -> None:
        """Stop a model's server (graceful terminate, then force-kill on timeout)."""
        record = self._processes.get(model_id)
        if record is None:
            return
        record.status = "stopping"
        self._terminate(model_id, record.pid, graceful=graceful)
        self._processes.pop(model_id, None)
        self._popen.pop(model_id, None)
        self._persist()

    def restart(self, model_id: str) -> ServerProcess:
        self.stop(model_id)
        return self.start(model_id)

    # --- Status ---------------------------------------------------------------
    def status(self, model_id: str | None = None) -> list[ServerProcess]:
        """Return live status, pruning any processes whose PID has died."""
        dead = [mid for mid, rec in self._processes.items() if not pid_alive(rec.pid)]
        for mid in dead:
            logger.info("Process for %s (pid gone) pruned", mid)
            self._processes.pop(mid, None)
        if dead:
            self._persist()

        if model_id is not None:
            rec = self._processes.get(model_id)
            return [rec] if rec else []
        return list(self._processes.values())

    def is_running(self, model_id: str) -> bool:
        rec = self._processes.get(model_id)
        if rec is None:
            return False
        if not pid_alive(rec.pid):
            self._processes.pop(model_id, None)
            self._persist()
            return False
        return True

    def get_server_url(self, model_id: str) -> str | None:
        return self._processes[model_id].url if self.is_running(model_id) else None

    # --- Internals ------------------------------------------------------------
    def _find_free_port(self, preferred: int) -> int:
        return find_free_port(preferred)

    def _health_check(self, url: str, timeout: int) -> bool:
        """Poll ``GET {url}/health`` every 2s until 200 or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{url}/health", timeout=5)
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(2)
        return False

    def _terminate(
        self, model_id: str, pid: int, graceful: bool = True, wait: float = 10.0
    ) -> None:
        """Terminate a server process, cross-platform.

        Prefers the tracked :class:`subprocess.Popen` handle (portable
        ``terminate()`` -> ``kill()``); falls back to PID-based signals for
        servers recovered from the state file after a hub restart.
        """
        if not pid_alive(pid):
            return
        popen = self._popen.get(model_id)
        try:
            if graceful:
                self._signal(popen, pid, force=False)
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    if not pid_alive(pid):
                        return
                    time.sleep(0.5)
            if pid_alive(pid):  # force
                self._signal(popen, pid, force=True)
        except (ProcessLookupError, OSError) as exc:
            logger.debug("Terminate %s (pid %s): %s", model_id, pid, exc)

    @staticmethod
    def _signal(popen: subprocess.Popen[bytes] | None, pid: int, force: bool) -> None:
        """Send terminate/kill via Popen when available, else via os signals."""
        import os
        import signal

        if popen is not None:
            popen.kill() if force else popen.terminate()
            return
        # No Popen handle (process recovered from state file). On Windows, os.kill
        # maps to TerminateProcess; SIGKILL only exists on POSIX.
        sig = signal.SIGTERM
        if os.name != "nt" and force:
            sig = getattr(signal, "SIGKILL", signal.SIGTERM)
        os.kill(pid, sig)

    def _run(self, cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}"
            )

    def _persist(self) -> None:
        self.state.save(self._processes)
