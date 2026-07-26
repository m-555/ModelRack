"""Tests for `setup`'s package-index controls.

These two settings only matter together, and getting them wrong fails in ways
that point at the wrong thing:

  * an accelerator wheel index without a matching index strategy makes an
    unrelated dependency unresolvable, and uv's error names that dependency
    rather than the index;
  * no index at all silently yields a CPU build of an accelerator package, which
    produces correct output at ~100x the cost and never errors.
"""

from __future__ import annotations

import pytest

from modelrack.process_manager import (
    EXTRA_INDEX_ENV,
    INDEX_STRATEGY_ENV,
    _extra_index_urls,
    _index_strategy,
)


# ── index strategy ───────────────────────────────────────────────────────────

def test_absent_by_default_so_uv_keeps_its_safe_default():
    """Not setting it must leave uv on `first-index` — the dependency-confusion
    protection. Never opt a user in implicitly."""
    assert _index_strategy({}) is None


@pytest.mark.parametrize(
    "value", ["first-index", "unsafe-first-match", "unsafe-best-match"]
)
def test_accepts_uvs_documented_strategies(value):
    assert _index_strategy({"pip_index_strategy": value}) == value


def test_unknown_strategy_is_dropped_not_forwarded():
    """Passing a typo through to uv would abort the whole setup. Warn and use
    the default instead — a slow-but-working install beats no install."""
    assert _index_strategy({"pip_index_strategy": "best-match"}) is None
    assert _index_strategy({"pip_index_strategy": "nonsense"}) is None


def test_model_config_beats_the_machine_wide_env(monkeypatch):
    monkeypatch.setenv(INDEX_STRATEGY_ENV, "unsafe-first-match")
    assert _index_strategy({"pip_index_strategy": "unsafe-best-match"}) == (
        "unsafe-best-match"
    )


def test_env_var_applies_when_the_model_says_nothing(monkeypatch):
    monkeypatch.setenv(INDEX_STRATEGY_ENV, "unsafe-best-match")
    assert _index_strategy({}) == "unsafe-best-match"


def test_blank_values_are_treated_as_unset(monkeypatch):
    monkeypatch.setenv(INDEX_STRATEGY_ENV, "   ")
    assert _index_strategy({"pip_index_strategy": ""}) is None


# ── extra index urls ─────────────────────────────────────────────────────────

def test_single_url_string_and_list_both_work():
    assert _extra_index_urls({"pip_extra_index_url": "https://a/x"}) == ["https://a/x"]
    assert _extra_index_urls(
        {"pip_extra_index_url": ["https://a/x", "https://b/y"]}
    ) == ["https://a/x", "https://b/y"]


def test_model_and_env_urls_combine_without_duplicates(monkeypatch):
    monkeypatch.setenv(EXTRA_INDEX_ENV, "https://b/y, https://c/z")
    assert _extra_index_urls({"pip_extra_index_url": "https://a/x"}) == [
        "https://a/x", "https://b/y", "https://c/z",
    ]
    # A URL named in both places is passed once, keeping first-seen order.
    monkeypatch.setenv(EXTRA_INDEX_ENV, "https://a/x")
    assert _extra_index_urls({"pip_extra_index_url": "https://a/x"}) == ["https://a/x"]


def test_no_config_means_no_flags():
    assert _extra_index_urls({}) == []


# ── the flags actually reach the uv command ──────────────────────────────────
# Resolving correctly is only half of it; the settings are worthless if `setup`
# never passes them. This intercepts the command instead of doing a multi-GB
# install.

def _setup_cmd(tmp_path, environment: dict) -> list[str]:
    """Run ProcessManager.setup against a stub model, returning the uv install
    command it built."""
    import shutil
    import yaml
    from modelrack.process_manager import ProcessManager

    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")

    models = tmp_path / "models"
    mdir = models / "stub"
    mdir.mkdir(parents=True)
    (mdir / "requirements.txt").write_text("torch==2.7.1\n", encoding="utf-8")
    (mdir / "server.py").write_text("# stub\n", encoding="utf-8")
    (mdir / "config.yaml").write_text(yaml.safe_dump({
        "model_id": "stub", "type": "audio_generation", "backend": "local",
        "weights": {"main": "weights"},
        "server": {"port": 7899},
        "environment": {"python_version": "3.11", **environment},
    }), encoding="utf-8")
    (mdir / "weights").mkdir()
    (models / "registry.yaml").write_text(yaml.safe_dump({
        "version": "1.0",
        "models": {"stub": {
            "type": "audio_generation", "backend": "local",
            "config_path": "stub/config.yaml",
        }},
    }), encoding="utf-8")

    pm = ProcessManager(models, state_file=tmp_path / "procs.json")
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))

    pm._run = fake_run                                    # type: ignore[method-assign]
    pm.setup("stub")
    return next((c for c in calls if "install" in c), [])


def test_setup_passes_both_index_flags(tmp_path):
    cmd = _setup_cmd(tmp_path, {
        "pip_extra_index_url": "https://download.pytorch.org/whl/cu128",
        "pip_index_strategy": "unsafe-best-match",
    })
    assert "--extra-index-url" in cmd
    assert "https://download.pytorch.org/whl/cu128" in cmd
    assert "--index-strategy" in cmd
    assert cmd[cmd.index("--index-strategy") + 1] == "unsafe-best-match"


def test_setup_omits_the_flags_when_unconfigured(tmp_path):
    cmd = _setup_cmd(tmp_path, {})
    assert "--extra-index-url" not in cmd
    assert "--index-strategy" not in cmd
