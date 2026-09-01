from __future__ import annotations

from typing import TYPE_CHECKING

from t4perceval.core.component import ColumnarComponent

if TYPE_CHECKING:
    from t4perceval.core.component import Component

__all__ = ("component_types", "resolve_component_type")


def _discover() -> dict[str, type[Component]]:
    import t4perceval.component as components

    found: dict[str, type[Component]] = {}
    for name in dir(components):
        candidate = getattr(components, name)
        if (
            isinstance(candidate, type)
            and issubclass(candidate, ColumnarComponent)
            and candidate is not ColumnarComponent
        ):
            found[candidate.__name__] = candidate
    return found


_CACHE: dict[str, type[Component]] | None = None


def component_types() -> dict[str, type[Component]]:
    """Return every known component class, keyed by class name."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _discover()
    return dict(_CACHE)


def resolve_component_type(name: str) -> type[Component]:
    """Return the component class named ``name``.

    Serialized data records the component class by name; this is where that name is
    turned back into a class, so no module paths end up baked into stored files.
    """
    try:
        return component_types()[name]
    except KeyError:
        known = ", ".join(sorted(component_types()))
        raise KeyError(f"Unknown component type {name!r}; known: {known}") from None
