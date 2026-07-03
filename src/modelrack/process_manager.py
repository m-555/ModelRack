"""Lifecycle management for per-model inference-server subprocesses.

Each local model runs its own FastAPI server inside its own isolated ``.venv``. This
module creates those venvs (via ``uv``), spawns/stops the servers, health-checks them,
and persists their state so running servers survive a hub restart.
"""

from __future__ import annotations

import logging
import os
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
from modelrack.utils.paths import (
    resolve_venv_dir,
    resolve_venv_exists,
    resolve_venv_python,
    venv_python_version,
)
from modelrack.utils.ports import find_free_port, is_port_free
from modelrack.utils.procstate import ProcessStateStore, pid_alive

logger = logging.getLogger("modelrack")

UV_INSTALL_HINT = (
    "uv not found on PATH. Install it with:\n"
    "  curl -LsSf https://astral.sh/uv/install.sh | sh   (Linux/macOS)\n"
    '  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   (Windows)'
)

# Machine-wide extra package index(es) for `setup` installs — e.g. a PyTorch CUDA
# wheel index. Comma-separated. Applies to every model, so the GPU-specific choice
# stays out of the (portable) committed configs.
EXTRA_INDEX_ENV = "MODELRACK_PIP_EXTRA_INDEX_URL"


def _extra_index_urls(environment: dict) -> list[str]:
    """Collect extra package index URLs for a `uv pip install`, from the model's
    ``environment.pip_extra_index_url`` (str or list) plus the machine-wide
    ``MODELRACK_PIP_EXTRA_INDEX_URL`` env var. De-duplicated, order-preserving."""
    urls: list[str] = []
    cfg = environment.get("pip_extra_index_url")
    if isinstance(cfg, str):
        urls.append(cfg)
    elif isinstance(cfg, (list, tuple)):
        urls.extend(str(u) for u in cfg)
    env_val = os.environ.get(EXTRA_INDEX_ENV, "").strip()
    if env_val:
        urls.extend(part.strip() for part in env_val.split(",") if part.strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


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
        python_version = str(env.get("python_version", "3.11"))
        requirements = model_dir / env.get("requirements_file", "requirements.txt")
        shared_name = env.get("shared_venv")

        uv = shutil.which("uv")
        if uv is None:
            raise UvNotFoundError(UV_INSTALL_HINT)

        vdir = resolve_venv_dir(self.models_dir, model_dir, env)

        # --force rebuilds a per-model venv, but must NOT wipe a SHARED venv that other
        # models depend on — there we only reinstall this model's requirements into it.
        if force and vdir.exists():
            if shared_name:
                logger.warning(
                    "Not removing shared venv %s on --force (other models use it); "
                    "reinstalling %s's requirements into it instead.",
                    vdir, model_id,
                )
            else:
                logger.info("Removing existing venv %s", vdir)
                shutil.rmtree(vdir)

        if not resolve_venv_exists(self.models_dir, model_dir, env):
            logger.info(
                "Creating %svenv for %s (python %s) at %s",
                "shared " if shared_name else "", model_id, python_version, vdir,
            )
            self._run([uv, "venv", "--python", python_version, str(vdir)])
        else:
            self._warn_python_mismatch(vdir, python_version, model_id, bool(shared_name))

        if requirements.exists():
            logger.info(
                "Installing requirements for %s into %s",
                model_id, f"shared venv '{shared_name}'" if shared_name else "its venv",
            )
            cmd = [
                uv,
                "pip",
                "install",
                "-r",
                str(requirements),
                "--python",
                str(resolve_venv_python(self.models_dir, model_dir, env)),
            ]
            for url in _extra_index_urls(env):
                cmd += ["--extra-index-url", url]
                logger.info("Using extra package index for %s: %s", model_id, url)
            self._run(cmd)
        else:
            logger.warning("No requirements.txt for %s at %s", model_id, requirements)

        self._ensure_server_file(resolved, model_dir)

        if self.registry.exists(model_id):
            self.registry.update_model(model_id, setup_complete=True)

    def _warn_python_mismatch(
        self, vdir: Path, expected_pyver: str, model_id: str, shared: bool
    ) -> None:
        """Warn when an existing (especially shared) venv's Python differs from the
        model's ``python_version`` — the classic cause of subtle shared-venv breakage."""
        actual = venv_python_version(vdir)
        if actual is None:
            return
        exp = expected_pyver.strip()
        # Compare on the parts the config pins (e.g. "3.11" vs the venv's "3.11.9").
        if not actual.startswith(exp) and not exp.startswith(actual):
            logger.warning(
                "%svenv %s is Python %s but '%s' requests python_version=%s. Mixing "
                "Python versions in one venv causes subtle breakage; give this model its "
                "own shared_venv.",
                "Shared " if shared else "", vdir, actual, model_id, expected_pyver,
            )

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

        if not resolve_venv_exists(self.models_dir, model_dir, env):
            raise SetupNotCompletedError(
                f"'{model_id}' is not set up. Run 'modelrack setup {model_id}' first."
            )

        server_py = model_dir / "server.py"
        if not server_py.exists():
            raise SetupNotCompletedError(f"server.py missing for '{model_id}': {server_py}")

        server_cfg = resolved.merged_config.get("server", {})
        host = server_cfg.get("host", "127.0.0.1")
        configured_port = resolved.server_port or 7800
        timeout = int(server_cfg.get("startup_timeout_sec", 120))

        # Use the configured port if bindable; otherwise fall back to a nearby free
        # one (handles ports already in use OR OS-reserved, e.g. Windows/Hyper-V
        # excluded ranges). The hub tracks the actual port, so routing is unaffected.
        if is_port_free(configured_port, host):
            port = configured_port
        else:
            try:
                port = find_free_port(configured_port, host)
            except OSError as exc:
                raise PortConflictError(
                    f"Port {configured_port} for '{model_id}' is unavailable on {host} "
                    f"and no free port was found nearby."
                ) from exc
            logger.warning(
                "Port %s for '%s' is unavailable (in use or OS-reserved); "
                "using free port %s instead.",
                configured_port, model_id, port,
            )

        python = resolve_venv_python(self.models_dir, model_dir, env)
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
