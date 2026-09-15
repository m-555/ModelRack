"""Characterisation tests for the Typer CLI (``modelrack.cli.main``).

These pin down what the CLI does *today* — argument handling, exit codes and the
text it prints — so that a refactor of ``cli/main.py`` is visibly behaviour
preserving. They record current behaviour; they are not a judgement about what
that behaviour ought to be.

Nothing here starts a model subprocess, binds a port or touches the network.
Every command exercised either fails before it gets that far, or has the
``ModelRack`` method it calls patched out; ``subprocess.call`` is stubbed with a
tripwire so an accidental ``$EDITOR`` launch fails the test instead of hanging.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from modelrack import __version__
from modelrack.cli import main as cli
from modelrack.cli.main import app
from modelrack.exceptions import ModelRackError

runner = CliRunner()

# Every command the CLI exposes today.
EXPECTED_COMMANDS = [
    "list",
    "show",
    "schema",
    "scan",
    "add",
    "remove",
    "edit",
    "validate",
    "setup",
    "start",
    "stop",
    "restart",
    "status",
    "unload",
    "infer",
    "serve",
]


def _squash(text: str) -> str:
    """Collapse all whitespace — Rich word-wraps, so raw substrings are brittle."""
    return " ".join(text.split())


def _all_output(result: Any) -> str:
    """stdout + stderr of an invocation, whitespace-normalised.

    Click >= 8.2 keeps the two streams apart; older versions mix them into
    ``output``. The CLI prints errors via a stderr Console and results via a
    stdout one, so tests that only care *that* a message appeared look at both.
    """
    parts = [result.stdout]
    with contextlib.suppress(AttributeError, ValueError):  # older click mixes the streams
        parts.append(result.stderr)
    return _squash(" ".join(parts))


def _forbidden_call(*args: Any, **kwargs: Any) -> int:
    raise AssertionError(f"CLI unexpectedly launched a subprocess: {args!r}")


@pytest.fixture(autouse=True)
def wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich reads COLUMNS; a wide console keeps messages on one line."""
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def cli_env(models_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at the temp MODELS_DIR with a throwaway process-state file."""
    monkeypatch.setenv("MODELS_DIR", str(models_dir))
    monkeypatch.setenv("MODELRACK_STATE_FILE", str(tmp_path / "state" / "processes.json"))
    monkeypatch.setattr(cli.subprocess, "call", _forbidden_call)
    return models_dir


# ── _parse_json ───────────────────────────────────────────────────────────────


def test_parse_json_none_is_empty_dict():
    assert cli._parse_json(None, "--runtime") == {}


def test_parse_json_empty_string_is_empty_dict():
    # An empty --payload is silently treated as "no params", not as an error.
    assert cli._parse_json("", "--payload") == {}


def test_parse_json_parses_a_nested_object():
    parsed = cli._parse_json('{"a": 1, "b": {"c": [1, 2]}}', "--runtime")
    assert parsed == {"a": 1, "b": {"c": [1, 2]}}


def test_parse_json_empty_object_stays_empty():
    assert cli._parse_json("{}", "--runtime") == {}


def test_parse_json_malformed_exits_1(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(typer.Exit) as exc:
        cli._parse_json("{nope", "--runtime")
    assert exc.value.exit_code == 1
    assert "Error: Invalid JSON for --runtime" in _squash(capsys.readouterr().err)


@pytest.mark.parametrize("value", ["[1, 2]", '"text"', "5", "true", "null"])
def test_parse_json_rejects_non_objects(value: str, capsys: pytest.CaptureFixture[str]):
    # Valid JSON that isn't an object (arrays and scalars, "null" included) is refused.
    with pytest.raises(typer.Exit) as exc:
        cli._parse_json(value, "--payload")
    assert exc.value.exit_code == 1
    assert "Error: --payload must be a JSON object" in _squash(capsys.readouterr().err)


# ── _die ──────────────────────────────────────────────────────────────────────


def test_die_exits_1_and_writes_to_stderr(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(typer.Exit) as exc:
        cli._die("something broke")
    assert exc.value.exit_code == 1
    captured = capsys.readouterr()
    assert "Error: something broke" in _squash(captured.err)
    assert captured.out == ""


# ── _hub ──────────────────────────────────────────────────────────────────────


def test_missing_models_dir_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "nowhere"))
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 1
    assert "MODELS_DIR does not exist" in _all_output(result)


# ── Top level ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag(flag: str):
    result = runner.invoke(app, [flag])
    assert result.exit_code == 0
    assert _squash(result.stdout) == f"modelrack {__version__}"


def test_no_args_prints_help_and_exits_2():
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage:" in _all_output(result)


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _all_output(result)
    for command in EXPECTED_COMMANDS:
        assert command in out


def test_unknown_command_exits_2():
    result = runner.invoke(app, ["frobnicate"])
    assert result.exit_code == 2
    assert "No such command" in _all_output(result)


# ── list ──────────────────────────────────────────────────────────────────────


def test_list_shows_registered_models(cli_env: Path):
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "Registered models" in out
    assert "demo-image" in out
    assert "demo-llm" in out


def test_list_filters_by_type(cli_env: Path):
    result = runner.invoke(app, ["list", "--type", "language"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "demo-llm" in out
    assert "demo-image" not in out


def test_list_filters_by_backend(cli_env: Path):
    result = runner.invoke(app, ["list", "--backend", "local"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "demo-image" in out
    assert "demo-llm" in out


def test_list_filters_by_tag(cli_env: Path):
    result = runner.invoke(app, ["list", "--tags", "image"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "demo-image" in out
    assert "demo-llm" not in out


def test_list_tags_use_and_logic(cli_env: Path):
    # No single model carries both tags, so the AND filter matches nothing.
    result = runner.invoke(app, ["list", "--tags", "image", "--tags", "language"])
    assert result.exit_code == 0
    assert "No models found" in _all_output(result)


def test_list_empty_result_still_exits_0(cli_env: Path):
    result = runner.invoke(app, ["list", "--type", "no-such-type"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "No models found. Try 'modelrack scan'." in out


# ── show ──────────────────────────────────────────────────────────────────────


def test_show_prints_resolved_yaml(cli_env: Path):
    result = runner.invoke(app, ["show", "demo-image"])
    assert result.exit_code == 0
    out = result.stdout
    assert "model_id: demo-image" in out
    assert "merged_config:" in out
    assert "param_schema:" in out


def test_show_applies_runtime_overrides(cli_env: Path):
    result = runner.invoke(app, ["show", "demo-image", "--runtime", '{"num_inference_steps": 20}'])
    assert result.exit_code == 0
    assert "num_inference_steps: 20" in result.stdout


def test_show_empty_runtime_is_ignored(cli_env: Path):
    # "" parses to {} which the command turns into None — same as omitting it.
    result = runner.invoke(app, ["show", "demo-image", "--runtime", ""])
    assert result.exit_code == 0
    assert "num_inference_steps: 50" in result.stdout


def test_show_malformed_runtime_exits_1(cli_env: Path):
    result = runner.invoke(app, ["show", "demo-image", "--runtime", "{bad"])
    assert result.exit_code == 1
    assert "Invalid JSON for --runtime" in _all_output(result)
    assert result.stdout.strip() == ""


def test_show_non_object_runtime_exits_1(cli_env: Path):
    result = runner.invoke(app, ["show", "demo-image", "--runtime", "[1]"])
    assert result.exit_code == 1
    assert "--runtime must be a JSON object" in _all_output(result)


def test_show_out_of_range_runtime_exits_1(cli_env: Path):
    result = runner.invoke(
        app, ["show", "demo-image", "--runtime", '{"num_inference_steps": 9999}']
    )
    assert result.exit_code == 1
    assert "above max 100" in _all_output(result)


def test_show_unknown_model_exits_1(cli_env: Path):
    result = runner.invoke(app, ["show", "nope"])
    assert result.exit_code == 1
    assert "Model 'nope' is not registered." in _all_output(result)


def test_show_without_model_id_exits_2(cli_env: Path):
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 2
    assert "Missing argument" in _all_output(result)


# ── schema ────────────────────────────────────────────────────────────────────


def test_schema_prints_param_schema(cli_env: Path):
    result = runner.invoke(app, ["schema", "demo-image"])
    assert result.exit_code == 0
    out = result.stdout
    assert "num_inference_steps:" in out
    assert "guidance_scale:" in out
    # param_schema only — the resolved config is not included.
    assert "merged_config" not in out


def test_schema_unknown_model_exits_1(cli_env: Path):
    # `schema` reads config.yaml directly, so an unregistered id is reported as a
    # missing file — not as the "is not registered" message `show` gives.
    result = runner.invoke(app, ["schema", "nope"])
    assert result.exit_code == 1
    out = _all_output(result)
    assert "config.yaml not found for 'nope'" in out
    assert "is not registered" not in out


# ── scan ──────────────────────────────────────────────────────────────────────


def test_scan_reports_already_registered(cli_env: Path):
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "already_registered" in out
    assert "demo-image" in out
    assert "demo-llm" in out


def test_scan_empty_dir_reports_nothing_to_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    empty = tmp_path / "empty-models"
    empty.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(empty))
    monkeypatch.setenv("MODELRACK_STATE_FILE", str(tmp_path / "state" / "processes.json"))
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "Nothing to sync." in _all_output(result)


# ── add / remove ──────────────────────────────────────────────────────────────


def test_add_registers_a_model(cli_env: Path):
    result = runner.invoke(app, ["add", "new-model", "--type", "language"])
    assert result.exit_code == 0
    assert "Added new-model" in _all_output(result)

    listed = runner.invoke(app, ["list"])
    assert "new-model" in _all_output(listed)


def test_add_defaults_to_local_backend_with_tags(cli_env: Path):
    result = runner.invoke(
        app, ["add", "tagged-model", "--type", "language", "--tags", "alpha", "--tags", "beta"]
    )
    assert result.exit_code == 0
    listed = _all_output(runner.invoke(app, ["list", "--backend", "local", "--tags", "alpha"]))
    assert "tagged-model" in listed
    assert "beta" in listed


def test_add_duplicate_exits_1(cli_env: Path):
    result = runner.invoke(app, ["add", "demo-llm", "--type", "language"])
    assert result.exit_code == 1
    assert "already registered" in _all_output(result)


def test_add_without_type_exits_2(cli_env: Path):
    result = runner.invoke(app, ["add", "new-model"])
    assert result.exit_code == 2
    assert "Missing option" in _all_output(result)


def test_remove_deregisters_but_keeps_files(cli_env: Path):
    result = runner.invoke(app, ["remove", "demo-llm"])
    assert result.exit_code == 0
    assert "Removed demo-llm" in _all_output(result)
    assert (cli_env / "demo-llm" / "config.yaml").exists()
    assert "demo-llm" not in _all_output(runner.invoke(app, ["list"]))


def test_remove_unknown_model_exits_1(cli_env: Path):
    result = runner.invoke(app, ["remove", "nope"])
    assert result.exit_code == 1
    assert "Model 'nope' is not registered." in _all_output(result)


# ── validate ──────────────────────────────────────────────────────────────────


def test_validate_without_target_exits_1(cli_env: Path):
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 1
    assert "Provide a model_id or use --all." in _all_output(result)


def test_validate_passes_with_warnings(cli_env: Path):
    # The fixture models have no .venv, which is a warning rather than an error.
    result = runner.invoke(app, ["validate", "demo-image"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "PASS demo-image" in out
    assert "warn: venv not found" in out


def test_validate_all_covers_every_model(cli_env: Path):
    result = runner.invoke(app, ["validate", "--all"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "PASS demo-image" in out
    assert "PASS demo-llm" in out


def test_validate_all_ignores_a_model_id_argument(cli_env: Path):
    # --all wins: the positional model_id is silently dropped.
    result = runner.invoke(app, ["validate", "demo-llm", "--all"])
    assert result.exit_code == 0
    assert "PASS demo-image" in _all_output(result)


def test_validate_unknown_model_exits_1(cli_env: Path):
    result = runner.invoke(app, ["validate", "nope"])
    assert result.exit_code == 1
    out = _all_output(result)
    assert "FAIL nope" in out
    # The per-model failure goes to stdout, not to the error console.
    assert "FAIL nope" in _squash(result.stdout)


# ── status / stop ─────────────────────────────────────────────────────────────


def test_status_with_no_servers(cli_env: Path):
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "Server processes" in out
    assert "No running servers." in out


def test_status_for_one_model_with_no_servers(cli_env: Path):
    result = runner.invoke(app, ["status", "demo-llm"])
    assert result.exit_code == 0
    assert "No running servers." in _all_output(result)


def test_stop_unknown_model_is_a_silent_success(cli_env: Path):
    # Today `stop` reports success for any id, running or not, registered or not.
    result = runner.invoke(app, ["stop", "totally-unknown"])
    assert result.exit_code == 0
    assert "Stopped totally-unknown" in _all_output(result)


# ── unload / infer / start ────────────────────────────────────────────────────


def test_unload_when_not_running_exits_1(cli_env: Path):
    result = runner.invoke(app, ["unload", "demo-llm"])
    assert result.exit_code == 1
    assert "is not running (auto_start=False)" in _all_output(result)


def test_infer_malformed_payload_exits_1(cli_env: Path):
    result = runner.invoke(app, ["infer", "demo-llm", "--payload", "nope"])
    assert result.exit_code == 1
    assert "Invalid JSON for --payload" in _all_output(result)


def test_infer_without_payload_exits_2(cli_env: Path):
    result = runner.invoke(app, ["infer", "demo-llm"])
    assert result.exit_code == 2
    assert "Missing option" in _all_output(result)


def test_infer_no_auto_start_when_not_running_exits_1(cli_env: Path):
    result = runner.invoke(
        app, ["infer", "demo-llm", "--payload", '{"prompt": "hi"}', "--no-auto-start"]
    )
    assert result.exit_code == 1
    assert "is not running (auto_start=False)" in _all_output(result)


def test_infer_prints_json_result(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[Any, ...]] = []

    def fake_infer(self: Any, model_id: str, payload: dict, **kwargs: Any) -> dict:
        calls.append((model_id, payload, kwargs))
        return {"success": True, "data": {"text": "hello"}, "error": None}

    monkeypatch.setattr(cli.ModelRack, "infer", fake_infer)
    result = runner.invoke(app, ["infer", "demo-llm", "--payload", '{"prompt": "hi"}'])

    assert result.exit_code == 0
    assert calls == [("demo-llm", {"prompt": "hi"}, {"auto_start": True, "timeout": 300})]
    assert '"text": "hello"' in result.stdout


def test_infer_flags_reach_the_hub(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, Any]] = []

    def fake_infer(self: Any, model_id: str, payload: dict, **kwargs: Any) -> dict:
        calls.append(kwargs)
        return {"success": True}

    monkeypatch.setattr(cli.ModelRack, "infer", fake_infer)
    result = runner.invoke(
        app,
        ["infer", "demo-llm", "--payload", "{}", "--no-auto-start", "--timeout", "7"],
    )

    assert result.exit_code == 0
    assert calls == [{"auto_start": False, "timeout": 7}]


def test_start_reports_url_and_pid(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    class _Proc:
        url = "http://127.0.0.1:7899"
        pid = 4242

    monkeypatch.setattr(cli.ModelRack, "start", lambda self, model_id: _Proc())
    result = runner.invoke(app, ["start", "demo-llm"])

    assert result.exit_code == 0
    out = _all_output(result)
    assert "Running demo-llm at http://127.0.0.1:7899 (pid 4242)" in out


def test_start_failure_exits_1(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(self: Any, model_id: str) -> None:
        raise ModelRackError("venv missing")

    monkeypatch.setattr(cli.ModelRack, "start", boom)
    result = runner.invoke(app, ["start", "demo-llm"])

    assert result.exit_code == 1
    assert "Error: venv missing" in _all_output(result)


def test_setup_failure_exits_1(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(self: Any, model_id: str, force: bool = False) -> None:
        raise ModelRackError("uv not found")

    monkeypatch.setattr(cli.ModelRack, "setup", boom)
    result = runner.invoke(app, ["setup", "demo-llm"])

    assert result.exit_code == 1
    assert "Error: uv not found" in _all_output(result)


def test_setup_force_flag_reaches_the_hub(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    seen: list[bool] = []

    monkeypatch.setattr(
        cli.ModelRack, "setup", lambda self, model_id, force=False: seen.append(force)
    )
    result = runner.invoke(app, ["setup", "demo-llm", "--force"])

    assert result.exit_code == 0
    assert seen == [True]
    assert "Setup complete" in _all_output(result)


def test_restart_reports_new_url(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    class _Proc:
        url = "http://127.0.0.1:7801"
        pid = 11

    monkeypatch.setattr(cli.ModelRack, "restart", lambda self, model_id: _Proc())
    result = runner.invoke(app, ["restart", "demo-image"])

    assert result.exit_code == 0
    assert "Restarted demo-image at http://127.0.0.1:7801" in _all_output(result)


# ── edit ──────────────────────────────────────────────────────────────────────


def test_edit_missing_config_exits_1(cli_env: Path):
    # _forbidden_call would trip if this reached $EDITOR.
    result = runner.invoke(app, ["edit", "no-such-model"])
    assert result.exit_code == 1
    assert "File not found" in _all_output(result)


def test_edit_server_flag_targets_server_py(cli_env: Path):
    # The fixture models ship no server.py, so --server dies on the path.
    result = runner.invoke(app, ["edit", "demo-image", "--server"])
    assert result.exit_code == 1
    out = _all_output(result)
    assert "File not found" in out
    assert "server.py" in out


def test_edit_opens_config_in_editor(cli_env: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []
    monkeypatch.setenv("EDITOR", "fake-editor")
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd: calls.append(cmd) or 0)

    result = runner.invoke(app, ["edit", "demo-image"])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == "fake-editor"
    assert calls[0][1] == str(cli_env / "demo-image" / "config.yaml")


# ── serve ─────────────────────────────────────────────────────────────────────


def test_serve_help_does_not_start_a_server(cli_env: Path):
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    out = _all_output(result)
    assert "--port" in out
    assert "--host" in out


def test_serve_rejects_a_non_integer_port(cli_env: Path):
    # Click rejects the value during parsing, before uvicorn is ever imported.
    result = runner.invoke(app, ["serve", "--port", "abc"])
    assert result.exit_code == 2
    assert "is not a valid integer" in _all_output(result)
