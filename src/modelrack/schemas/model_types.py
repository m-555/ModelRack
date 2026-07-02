"""Open, registry-driven model-type system.

modelrack deliberately avoids a *closed* enum of model types. New kinds of models
(coder LLMs, omni models, embeddings, rerankers, ...) are expected to be added over
time. A new kind plugs in by:

1. adding a server template ``templates/servers/<template>.py``,
2. adding a base requirements file ``templates/requirements/<file>.txt``,
3. registering the type below (or at runtime via :func:`register_type`).

Nothing in the registry / resolver / process-manager needs to change.

Arbitrary type strings are always *allowed* — an unregistered type simply has no
default template/requirements mapping and produces a validator *warning*, never an
error. This keeps the door open for experimentation.
"""

from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    """How a model is served."""

    LOCAL = "local"
    API = "api"


# --- Built-in type constants (string values, not a closed set) ------------------
VIDEO_GENERATION = "video_generation"
IMAGE_GENERATION = "image_generation"
IMAGE_EDIT = "image_edit"
TTS = "tts"
VISION_LANGUAGE = "vision_language"
LANGUAGE = "language"
CODE = "code"
OMNI = "omni"


class TypeSpec:
    """Defaults associated with a model type."""

    __slots__ = ("template", "requirements", "description")

    def __init__(self, template: str, requirements: str, description: str) -> None:
        self.template = template
        self.requirements = requirements
        self.description = description

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TypeSpec(template={self.template!r}, requirements={self.requirements!r})"


# type name -> defaults. Extend with register_type().
_TYPE_REGISTRY: dict[str, TypeSpec] = {
    VIDEO_GENERATION: TypeSpec(
        "server_diffusers_video.py", "video_generation.txt", "Text/image-to-video diffusion"
    ),
    IMAGE_GENERATION: TypeSpec(
        "server_diffusers_image.py", "image_generation.txt", "Text-to-image diffusion"
    ),
    IMAGE_EDIT: TypeSpec(
        "server_diffusers_image_edit.py", "image_edit.txt", "Instruction-based image editing"
    ),
    TTS: TypeSpec("server_tts.py", "tts.txt", "Text-to-speech synthesis"),
    VISION_LANGUAGE: TypeSpec(
        "server_transformers_vlm.py", "vision_language.txt", "Vision-language multimodal chat"
    ),
    LANGUAGE: TypeSpec("server_vllm_llm.py", "language.txt", "Large language model (chat)"),
    CODE: TypeSpec("server_vllm_llm.py", "code.txt", "Code-specialized language model"),
    OMNI: TypeSpec("server_omni.py", "omni.txt", "Any-to-any multimodal model"),
}


def register_type(name: str, *, template: str, requirements: str, description: str = "") -> None:
    """Register (or overwrite) defaults for a model type at runtime."""
    _TYPE_REGISTRY[name] = TypeSpec(template, requirements, description)


def known_types() -> list[str]:
    """Return the sorted list of registered (built-in + runtime) type names."""
    return sorted(_TYPE_REGISTRY)


def is_known_type(name: str) -> bool:
    return name in _TYPE_REGISTRY


def get_type_spec(name: str) -> TypeSpec | None:
    """Return the :class:`TypeSpec` for a type, or ``None`` if unregistered."""
    return _TYPE_REGISTRY.get(name)


def template_for(type_name: str) -> str | None:
    spec = _TYPE_REGISTRY.get(type_name)
    return spec.template if spec else None


def requirements_for(type_name: str) -> str | None:
    spec = _TYPE_REGISTRY.get(type_name)
    return spec.requirements if spec else None
