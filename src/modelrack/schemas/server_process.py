"""Runtime record for a per-model inference server process."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ServerProcess:
    """A tracked inference-server subprocess."""

    model_id: str
    port: int
    pid: int
    status: str  # "starting" | "running" | "stopping" | "stopped" | "error"
    started_at: datetime
    url: str

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, (datetime.now() - self.started_at).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "port": self.port,
            "pid": self.pid,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "url": self.url,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServerProcess:
        return cls(
            model_id=data["model_id"],
            port=int(data["port"]),
            pid=int(data["pid"]),
            status=data.get("status", "stopped"),
            started_at=datetime.fromisoformat(data["started_at"]),
            url=data["url"],
        )
