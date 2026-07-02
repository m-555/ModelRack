"""Deep-merge utility used by the 3-layer config resolver."""

from __future__ import annotations

import copy
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` and return a new dict.

    - Nested dicts are merged key-by-key (not shallow-overwritten).
    - Non-dict values in ``override`` win over ``base``.
    - Neither input is mutated; the result is a deep copy.

    >>> deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    {'a': {'x': 1, 'y': 3}}
    """
    result: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result
