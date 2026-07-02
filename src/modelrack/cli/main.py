"""Typer + Rich command-line interface for modelrack."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from modelrack import ModelRack, __version__
from modelrack.config import MODEL_PORT_RANGE
from modelrack.exceptions import ModelRackError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Centralized model registry, config pipeline & process manager.",
)
console = Console()
err_console = Console(stderr=True)


def _hub() -> ModelRack:
    try:
        return ModelRack()
    except ModelRackError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


def _die(message: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


def _parse_json(value: str | None, what: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        _die(f"Invalid JSON for {what}: {exc}")
    if not isinstance(parsed, dict):
        _die(f"{what} must be a JSON object")
    return parsed  # type: ignore[return-value]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"modelrack {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    _version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """modelrack — one source of truth for all your models."""


# ── Model management ──────────────────────────────────────────────────────────


@app.command("list")
def list_models(
    type: str = typer.Option(None, "--type", "-t", help="Filter by model type."),
    backend: str = typer.Option(None, "--backend", "-b", help="Filter by backend."),
    tags: list[str] = typer.Option(None, "--tags", help="Filter by tags (AND logic)."),
) -> None:
    """List all registered models."""
    hub = _hub()
    rows = hub.list(type=type, backend=backend, tags=tags or None)
    table = Table(title="Registered models", header_style="bold cyan")
    table.add_column("model_id", style="bold")
    table.add_column("type")
    table.add_column("backend")
    table.add_column("tags", style="dim")
    for entry in rows:
        table.add_row(
            entry["model_id"],
            entry.get("type", "-"),
            entry.get("backend", "-"),
            ", ".join(entry.get("tags") or []),
        )
    console.print(table)
    if not rows:
        console.print("[dim]No models found. Try 'modelrack scan'.[/dim]")


@app.command()
def show(
    model_id: str,
    runtime: str = typer.Option(None, "--runtime", help="Runtime params as JSON."),
) -> None:
    """Show a model's fully-resolved config."""
    hub = _hub()
    try:
        resolved = hub.resolve(model_id, runtime_params=_parse_json(runtime, "--runtime") or None)
    except ModelRackError as exc:
        _die(str(exc))
    console.print(
        Syntax(yaml.safe_dump(resolved.to_dict(), sort_keys=False), "yaml", theme="ansi_dark")
    )


@app.command()
def schema(model_id: str) -> None:
    """Show a model's param_schema (for app UI development)."""
    hub = _hub()
    try:
        data = hub.schema(model_id)
    except ModelRackError as exc:
        _die(str(exc))
    console.print(Syntax(yaml.safe_dump(data, sort_keys=False), "yaml", theme="ansi_dark"))


@app.command()
def scan() -> None:
    """Scan MODELS_DIR and sync the registry."""
    hub = _hub()
    report = hub.scan()
    for key, ids in report.items():
        if ids:
            console.print(f"[bold]{key}[/bold]: {', '.join(ids)}")
    if not any(report.values()):
        console.print("[dim]Nothing to sync.[/dim]")


@app.command()
def add(
    model_id: str,
    type: str = typer.Option(..., "--type", "-t"),
    backend: str = typer.Option("local", "--backend", "-b"),
    tags: list[str] = typer.Option(None, "--tags"),
) -> None:
    """Manually register a model in the registry."""
    hub = _hub()
    try:
        hub.registry.add_model(
            model_id,
            type=type,
            backend=backend,
            config_path=f"{model_id}/config.yaml" if backend == "local" else None,
            tags=tags or None,
        )
    except ModelRackError as exc:
        _die(str(exc))
    console.print(f"[green]Added[/green] {model_id}")


@app.command()
def remove(model_id: str) -> None:
    """Remove a model from the registry (does not delete files)."""
    hub = _hub()
    try:
        hub.registry.remove_model(model_id)
    except ModelRackError as exc:
        _die(str(exc))
    console.print(f"[yellow]Removed[/yellow] {model_id} (files left intact)")


@app.command()
def edit(
    model_id: str,
    server: bool = typer.Option(False, "--server", help="Open server.py instead of config.yaml."),
) -> None:
    """Open a model's config.yaml (or server.py) in $EDITOR."""
    hub = _hub()
    target = hub.models_dir / model_id / ("server.py" if server else "config.yaml")
    if not target.exists():
        _die(f"File not found: {target}")
    editor = os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
    subprocess.call([editor, str(target)])  # noqa: S603


@app.command()
def validate(
    model_id: str = typer.Argument(None),
    all_models: bool = typer.Option(False, "--all", help="Validate every registered model."),
) -> None:
    """Validate a model's config, weights and venv."""
    hub = _hub()
    if not all_models and model_id is None:
        _die("Provide a model_id or use --all.")
    targets = [m["model_id"] for m in hub.list()] if all_models else [model_id]

    any_errors = False
    for mid in targets:
        try:
            result = hub.validate(mid)
        except ModelRackError as exc:
            any_errors = True
            console.print(f"[red]FAIL[/red] [bold]{mid}[/bold]: {exc}")
            continue
        icon = "[green]PASS[/green]" if result.valid else "[red]FAIL[/red]"
        console.print(f"{icon} [bold]{mid}[/bold]")
        for e in result.errors:
            any_errors = True
            console.print(f"    [red]error:[/red] {e}")
        for w in result.warnings:
            console.print(f"    [yellow]warn:[/yellow] {w}")
    if any_errors:
        raise typer.Exit(code=1)


# ── Venv & setup ──────────────────────────────────────────────────────────────


@app.command()
def setup(
    model_id: str,
    force: bool = typer.Option(False, "--force", help="Wipe and recreate the venv."),
) -> None:
    """Create the model's .venv and install its requirements."""
    hub = _hub()
    try:
        with console.status(f"Setting up {model_id}..."):
            hub.setup(model_id, force=force)
    except ModelRackError as exc:
        _die(str(exc))
    console.print(f"[green]Setup complete[/green] for {model_id}")


# ── Process management ────────────────────────────────────────────────────────


@app.command()
def start(model_id: str) -> None:
    """Start a model's inference server (blocks until healthy)."""
    hub = _hub()
    try:
        with console.status(f"Starting {model_id}..."):
            proc = hub.start(model_id)
    except ModelRackError as exc:
        _die(str(exc))
    console.print(f"[green]Running[/green] {model_id} at {proc.url} (pid {proc.pid})")


@app.command()
def stop(model_id: str) -> None:
    """Stop a model's inference server."""
    _hub().stop(model_id)
    console.print(f"[yellow]Stopped[/yellow] {model_id}")


@app.command()
def restart(model_id: str) -> None:
    """Restart a model's inference server."""
    hub = _hub()
    try:
        proc = hub.restart(model_id)
    except ModelRackError as exc:
        _die(str(exc))
    console.print(f"[green]Restarted[/green] {model_id} at {proc.url}")


@app.command()
def status(model_id: str = typer.Argument(None)) -> None:
    """Show running inference servers."""
    hub = _hub()
    procs = hub.status(model_id)
    table = Table(title="Server processes", header_style="bold cyan")
    for col in ("model_id", "port", "pid", "status", "uptime"):
        table.add_column(col)
    for p in procs:
        table.add_row(p.model_id, str(p.port), str(p.pid), p.status, f"{p.uptime_seconds:.0f}s")
    console.print(table)
    if not procs:
        console.print("[dim]No running servers.[/dim]")


@app.command()
def unload(model_id: str) -> None:
    """Free a model from VRAM without stopping its server."""
    hub = _hub()
    try:
        hub.unload(model_id)
    except ModelRackError as exc:
        _die(str(exc))
    console.print(f"[green]Unloaded[/green] {model_id} (server still running)")


# ── Inference ─────────────────────────────────────────────────────────────────


@app.command()
def infer(
    model_id: str,
    payload: str = typer.Option(..., "--payload", help="Inference payload as JSON."),
    no_auto_start: bool = typer.Option(False, "--no-auto-start"),
    timeout: int = typer.Option(300, "--timeout"),
) -> None:
    """Run inference against a model (starts the server if needed)."""
    hub = _hub()
    try:
        result = hub.infer(
            model_id,
            _parse_json(payload, "--payload"),
            auto_start=not no_auto_start,
            timeout=timeout,
        )
    except ModelRackError as exc:
        _die(str(exc))
    console.print_json(data=result)


# ── Hub API server ────────────────────────────────────────────────────────────


@app.command()
def serve(
    port: int = typer.Option(None, "--port", "-p", help="Port (default: MODELRACK_PORT or 7777)."),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Start the hub management REST API."""
    import uvicorn

    from modelrack.api.server import create_app
    from modelrack.config import load_settings

    settings = load_settings()
    resolved_port = port or settings.hub_port
    console.print(
        Panel.fit(
            f"[bold]modelrack {__version__}[/bold]\n"
            f"API: http://{host}:{resolved_port}\n"
            f"Models dir: {settings.models_dir}\n"
            f"Model server port range: {MODEL_PORT_RANGE.start}-{MODEL_PORT_RANGE.stop - 1}",
            title="serve",
        )
    )
    uvicorn.run(create_app(), host=host, port=resolved_port, log_level=settings.log_level.lower())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
