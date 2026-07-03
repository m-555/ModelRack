"""Tests for ProcessManager (uv setup, start/stop, status) with mocked subprocess."""

from __future__ import annotations

import types
from datetime import datetime
from pathlib import Path

import pytest

import modelrack.process_manager as pm_mod
from modelrack.exceptions import (
    PortConflictError,
    ServerStartupError,
    SetupNotCompletedError,
    UvNotFoundError,
)
from modelrack.process_manager import ProcessManager
from modelrack.schemas.server_process import ServerProcess


def _fake_run_recorder(calls: list[list[str]]):
    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    return fake_run


def test_setup_calls_uv_with_correct_args(
    process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch, models_dir: Path
):
    calls: list[list[str]] = []
    monkeypatch.setattr(pm_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(pm_mod.subprocess, "run", _fake_run_recorder(calls))

    process_manager.setup("demo-image")

    assert calls[0][0] == "/usr/bin/uv"
    assert calls[0][1] == "venv"
    assert "--python" in calls[0] and "3.11" in calls[0]
    assert calls[1][1:4] == ["pip", "install", "-r"]


def test_extra_index_urls_helper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(pm_mod.EXTRA_INDEX_ENV, raising=False)
    assert pm_mod._extra_index_urls({}) == []
    assert pm_mod._extra_index_urls({"pip_extra_index_url": "https://a"}) == ["https://a"]
    assert pm_mod._extra_index_urls(
        {"pip_extra_index_url": ["https://a", "https://b"]}
    ) == ["https://a", "https://b"]
    # env-var entries are appended and de-duped against config (order preserved)
    monkeypatch.setenv(pm_mod.EXTRA_INDEX_ENV, "https://env1, https://a")
    assert pm_mod._extra_index_urls({"pip_extra_index_url": "https://a"}) == [
        "https://a",
        "https://env1",
    ]


def test_setup_passes_extra_index_url(
    process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch, models_dir: Path
):
    calls: list[list[str]] = []
    monkeypatch.setattr(pm_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(pm_mod.subprocess, "run", _fake_run_recorder(calls))
    monkeypatch.setenv(pm_mod.EXTRA_INDEX_ENV, "https://download.pytorch.org/whl/cu128")

    process_manager.setup("demo-image")

    install_cmd = calls[1]  # [uv, pip, install, -r, req, --python, py, --extra-index-url, URL]
    assert "--extra-index-url" in install_cmd
    idx = install_cmd.index("--extra-index-url")
    assert install_cmd[idx + 1] == "https://download.pytorch.org/whl/cu128"


def test_setup_copies_template_when_missing(
    process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch, models_dir: Path
):
    monkeypatch.setattr(pm_mod.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(pm_mod.subprocess, "run", _fake_run_recorder([]))

    server_py = models_dir / "demo-image" / "server.py"
    assert not server_py.exists()
    process_manager.setup("demo-image")
    assert server_py.exists()
    # image_generation -> diffusers image template
    assert "diffusers" in server_py.read_text(encoding="utf-8").lower()


def test_setup_without_uv_raises(process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pm_mod.shutil, "which", lambda _name: None)
    with pytest.raises(UvNotFoundError):
        process_manager.setup("demo-image")


def test_start_requires_setup(process_manager: ProcessManager):
    # No venv exists in the fixture, so start must refuse.
    with pytest.raises(SetupNotCompletedError):
        process_manager.start("demo-image")


def _raise_no_free_port(*_a, **_k):
    raise OSError("no free port")


def test_start_port_conflict(
    process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch, models_dir: Path
):
    # Configured port unavailable AND no free port nearby -> PortConflictError.
    monkeypatch.setattr(pm_mod, "resolve_venv_exists", lambda *_a, **_k: True)
    (models_dir / "demo-image" / "server.py").write_text("x", encoding="utf-8")
    monkeypatch.setattr(pm_mod, "is_port_free", lambda *_a, **_k: False)
    monkeypatch.setattr(pm_mod, "find_free_port", _raise_no_free_port)
    with pytest.raises(PortConflictError):
        process_manager.start("demo-image")


def test_start_port_fallback_uses_free_port(
    process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch, models_dir: Path
):
    # Configured port unavailable (e.g. OS-reserved) -> fall back to a free one.
    monkeypatch.setattr(pm_mod, "resolve_venv_exists", lambda *_a, **_k: True)
    (models_dir / "demo-image" / "server.py").write_text("x", encoding="utf-8")
    monkeypatch.setattr(pm_mod, "is_port_free", lambda *_a, **_k: False)
    monkeypatch.setattr(pm_mod, "find_free_port", lambda *_a, **_k: 7999)
    monkeypatch.setattr(
        pm_mod.subprocess, "Popen", lambda *_a, **_k: types.SimpleNamespace(pid=4242)
    )
    monkeypatch.setattr(pm_mod, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(ProcessManager, "_health_check", lambda *_a, **_k: True)

    rec = process_manager.start("demo-image")
    assert rec.port == 7999
    assert rec.url.endswith(":7999")


def test_start_health_timeout_raises(
    process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch, models_dir: Path
):
    monkeypatch.setattr(pm_mod, "resolve_venv_exists", lambda *_a, **_k: True)
    (models_dir / "demo-image" / "server.py").write_text("x", encoding="utf-8")
    monkeypatch.setattr(pm_mod, "is_port_free", lambda *_a, **_k: True)
    monkeypatch.setattr(
        pm_mod.subprocess, "Popen", lambda *_a, **_k: types.SimpleNamespace(pid=424242)
    )
    monkeypatch.setattr(pm_mod, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(ProcessManager, "_health_check", lambda *_a, **_k: False)

    with pytest.raises(ServerStartupError):
        process_manager.start("demo-image")
    assert not process_manager.is_running("demo-image")


def test_stop_graceful_terminates(process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch):
    class FakePopen:
        def __init__(self) -> None:
            self.pid = 4321
            self.terminated = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    fp = FakePopen()
    process_manager._processes["m"] = ServerProcess(
        "m", 7801, fp.pid, "running", datetime.now(), "http://127.0.0.1:7801"
    )
    process_manager._popen["m"] = fp  # type: ignore[assignment]
    # Alive until terminate() is called.
    monkeypatch.setattr(pm_mod, "pid_alive", lambda _pid: not fp.terminated)

    process_manager.stop("m")
    assert fp.terminated and not fp.killed
    assert "m" not in process_manager._processes


def test_stop_force_kills(process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch):
    class FakePopen:
        def __init__(self) -> None:
            self.pid = 4321
            self.killed = False

        def terminate(self) -> None:  # pragma: no cover - not used in force path
            pass

        def kill(self) -> None:
            self.killed = True

    fp = FakePopen()
    process_manager._processes["m"] = ServerProcess(
        "m", 7801, fp.pid, "running", datetime.now(), "http://127.0.0.1:7801"
    )
    process_manager._popen["m"] = fp  # type: ignore[assignment]
    monkeypatch.setattr(pm_mod, "pid_alive", lambda _pid: True)

    process_manager.stop("m", graceful=False)
    assert fp.killed


def test_status_prunes_dead_process(
    process_manager: ProcessManager, monkeypatch: pytest.MonkeyPatch
):
    process_manager._processes["ghost"] = ServerProcess(
        "ghost", 7801, 999999, "running", datetime.now(), "http://127.0.0.1:7801"
    )
    monkeypatch.setattr(pm_mod, "pid_alive", lambda _pid: False)
    assert process_manager.status() == []
    assert "ghost" not in process_manager._processes
