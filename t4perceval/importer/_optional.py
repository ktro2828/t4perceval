"""Turning a missing optional dependency into an actionable message.

Importers live behind extras, so the first thing a user without one sees is an import
error from a module they have never heard of. This says which extra to install instead.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ("require",)


def require(module: str, *, extra: str) -> ModuleType:
    """Import ``module``, or explain which extra provides it.

    Args:
        module: The module to import.
        extra: The extra that installs it.

    Returns:
        The imported module.

    Raises:
        ImportError: When the module is unavailable, naming the install command.
    """
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"{module!r} is required by this importer but is not installed. "
            f"Install it with:  pip install 't4perceval[{extra}]'",
        ) from error
