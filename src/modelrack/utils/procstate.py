"""Persistence + liveness helpers for tracked server processes.

State is stored as JSON at ``MODELRACK_STATE_FILE`` (default ~/.modelrack/processes.json)
so that running servers survive a hub restart. On load, each stored PID is checked for
liveness and dead entries are pruned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modelrack.schemas.server_process import ServerProcess
from modelrack.utils.paths import expand


def pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` currently exists.

    Cross-platform: uses ``os.kill(pid, 0)`` on POSIX and ``OpenProcess`` on Windows.
    """
    if pid <= 0:
        return False
    if os.name == "nt":  # Windows
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    else:  # POSIX
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but owned by another user
        return True


class ProcessStateStore:
    """Load/save the map of ``model_id -> ServerProcess`` with liveness pruning."""

    def __init__(self, state_file: str | Path) -> None:
        self.path = expand(state_file)

    def load(self) -> dict[str, ServerProcess]:
        """Load state, dropping entries whose PID is no longer alive."""
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

        result: dict[str, ServerProcess] = {}
        changed = False
        for model_id, entry in raw.items():
            try:
                proc = ServerProcess.from_dict(entry)
            except (KeyError, ValueError, TypeError):
                changed = True
                continue
            if pid_alive(proc.pid):
                result[model_id] = proc
            else:
                changed = True
        if changed:
            self.save(result)
        return result

    def save(self, processes: dict[str, ServerProcess]) -> None:
        """Atomically write the state map to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {mid: proc.to_dict() for mid, proc in processes.items()}
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
