"""Tests for the open model-type registry.

The registry is deliberately open — an unregistered type is a warning, never an
error — so what is worth locking is that a REGISTERED type actually resolves to
templates that exist on disk. A type whose template is missing fails at
`modelrack setup` time, long after the mistake was made.
"""

from __future__ import annotations

import copy
from pathlib import Path

import modelrack
from modelrack.schemas import model_types as mt
from modelrack.validator import ConfigValidator
from tests.conftest import DIFFUSERS_CONFIG

TEMPLATES = Path(modelrack.__file__).parent / "templates"


def test_every_registered_type_has_templates_on_disk():
    missing = []
    for name in mt.known_types():
        spec = mt.get_type_spec(name)
        if not (TEMPLATES / "servers" / spec.template).is_file():
            missing.append(f"{name}: servers/{spec.template}")
        if not (TEMPLATES / "requirements" / spec.requirements).is_file():
            missing.append(f"{name}: requirements/{spec.requirements}")
    assert not missing, f"registered types with no template: {missing}"


def test_audio_generation_is_a_first_class_type():
    """Text-to-audio (music, sound effects) is its own kind — not TTS. They
    share nothing beyond producing a waveform: different inputs, different
    parameters, different server contract."""
    assert mt.is_known_type(mt.AUDIO_GENERATION)
    spec = mt.get_type_spec(mt.AUDIO_GENERATION)
    assert spec.template == "server_audio_generation.py"
    assert spec.requirements == "audio_generation.txt"
    assert mt.AUDIO_GENERATION != mt.TTS


def test_an_audio_generation_config_validates():
    cfg = copy.deepcopy(DIFFUSERS_CONFIG)
    cfg["type"] = mt.AUDIO_GENERATION
    result = ConfigValidator().validate_model_config(cfg)
    assert result.valid, result.errors
    assert not any("Unknown model type" in w for w in result.warnings)


def test_runtime_registration_still_works():
    mt.register_type("test_kind", template="server_tts.py", requirements="tts.txt",
                     description="registered at runtime")
    try:
        assert mt.is_known_type("test_kind")
        assert mt.template_for("test_kind") == "server_tts.py"
    finally:
        mt._TYPE_REGISTRY.pop("test_kind", None)
