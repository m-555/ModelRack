"""Tests for the deep_merge utility."""

from __future__ import annotations

from modelrack.utils.merge import deep_merge


def test_scalar_override_wins():
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_nested_dicts_are_merged_not_replaced():
    base = {"defaults": {"steps": 50, "cfg": 7.5}}
    override = {"defaults": {"steps": 30}}
    assert deep_merge(base, override) == {"defaults": {"steps": 30, "cfg": 7.5}}


def test_new_keys_are_added():
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_inputs_are_not_mutated():
    base = {"x": {"y": 1}}
    override = {"x": {"z": 2}}
    deep_merge(base, override)
    assert base == {"x": {"y": 1}}
    assert override == {"x": {"z": 2}}


def test_dict_replaces_scalar_and_vice_versa():
    assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}
    assert deep_merge({"a": {"b": 2}}, {"a": 1}) == {"a": 1}


def test_three_layer_chain():
    base = {"defaults": {"steps": 50, "cfg": 7.5, "seed": -1}}
    app = {"defaults": {"steps": 30}}
    runtime = {"defaults": {"cfg": 4.0}}
    merged = deep_merge(deep_merge(base, app), runtime)
    assert merged["defaults"] == {"steps": 30, "cfg": 4.0, "seed": -1}
