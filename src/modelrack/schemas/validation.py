"""Result object shared by every validator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Outcome of a validation pass.

    ``valid`` reflects the absence of *errors*; warnings never make a result invalid.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Fold another result into this one and return ``self`` for chaining."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False
        return self

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}
