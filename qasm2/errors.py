"""Error model for OpenQASM 2 parsing and analysis.

This module centralizes the error reporting primitives so that tooling built on
`qasm2` can exchange structured diagnostics. Error codes follow the scheme
outlined in the instruction manual, grouping issues by category (for example,
lexical issues in the ``E10x`` family).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import ClassVar

__all__ = [
    "QasmError",
    "QasmLexicalError",
    "QasmSyntaxError",
    "QasmSemanticError",
    "QasmResolutionError",
    "QasmBackendError",
    "QasmGraphIntegrationError",
]

_CODE_PATTERN = re.compile(r"^E\d{3}$")


@dataclass(slots=True)
class QasmError(Exception):
    """Base class for all structured OpenQASM 2 errors.

    Parameters
    ----------
    code : str
        Manual-defined error identifier (for example, ``E201``).
    message : str
        Human-readable explanation of the problem.
    line : int
        One-based line index pointing at the source location.
    col : int
        One-based column index pointing at the source location.
    """

    code: str
    message: str
    line: int
    col: int

    def __post_init__(self) -> None:
        """Validate the common error attributes."""
        if not _CODE_PATTERN.match(self.code):
            raise ValueError("Error codes must follow the `E###` pattern as defined in the instruction manual.")
        if self.line < 1 or self.col < 1:
            raise ValueError("Source locations are one-based; line and column must be positive integers.")

    def __str__(self) -> str:
        """Return a concise diagnostic string."""
        return f"{self.code} (line {self.line}, col {self.col}): {self.message}"


class _CategorisedQasmError(QasmError):
    """Utility mixin enforcing category-specific validation."""

    CATEGORY_PREFIX: ClassVar[str]
    CATEGORY_LABEL: ClassVar[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.code.startswith(self.CATEGORY_PREFIX):
            raise ValueError(f"{self.CATEGORY_LABEL} must use an error code starting with '{self.CATEGORY_PREFIX}'.")


class QasmLexicalError(_CategorisedQasmError):
    """Lexical analysis failures (``E10x`` family)."""

    CATEGORY_PREFIX: ClassVar[str] = "E10"
    CATEGORY_LABEL: ClassVar[str] = "Lexical errors"


class QasmSyntaxError(_CategorisedQasmError):
    """Grammar and syntax violations (``E20x`` family)."""

    CATEGORY_PREFIX: ClassVar[str] = "E20"
    CATEGORY_LABEL: ClassVar[str] = "Syntax errors"


class QasmSemanticError(_CategorisedQasmError):
    """Semantic consistency issues (``E30x`` family)."""

    CATEGORY_PREFIX: ClassVar[str] = "E30"
    CATEGORY_LABEL: ClassVar[str] = "Semantic errors"


class QasmResolutionError(_CategorisedQasmError):
    """Symbol resolution and scoping failures (``E40x`` family)."""

    CATEGORY_PREFIX: ClassVar[str] = "E40"
    CATEGORY_LABEL: ClassVar[str] = "Resolution errors"


class QasmBackendError(_CategorisedQasmError):
    """Backend translation and target-specific issues (``E50x`` family)."""

    CATEGORY_PREFIX: ClassVar[str] = "E50"
    CATEGORY_LABEL: ClassVar[str] = "Backend errors"


class QasmGraphIntegrationError(_CategorisedQasmError):
    """GraphQOMB integration failures (``E70x`` family)."""

    CATEGORY_PREFIX: ClassVar[str] = "E70"
    CATEGORY_LABEL: ClassVar[str] = "Graph integration errors"
