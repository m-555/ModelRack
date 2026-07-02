"""Port helpers for the process manager."""

from __future__ import annotations

import socket


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if a TCP port can be bound on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(preferred: int, host: str = "127.0.0.1", attempts: int = 100) -> int:
    """Return ``preferred`` if free, else scan upward for the next free port.

    Raises ``OSError`` if no free port is found within ``attempts`` steps.
    """
    for candidate in range(preferred, preferred + attempts):
        if is_port_free(candidate, host):
            return candidate
    raise OSError(f"No free port found in range {preferred}-{preferred + attempts - 1} on {host}")
