from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

from attrs import define, field

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typing_extensions import Self

__all__ = ("EntityPath", "EntityPathLike", "as_entity_path")

_SEPARATOR = "/"


def _as_parts(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return EntityPath.parse(value).parts

    parts = tuple(str(part) for part in value)
    for part in parts:
        if part == "":
            raise ValueError("EntityPath parts must not be empty")
        if _SEPARATOR in part:
            raise ValueError(f"EntityPath part must not contain {_SEPARATOR!r}, got {part!r}")
    return parts


@define(frozen=True, slots=True)
class EntityPath:
    """A hierarchical address identifying one stream of components.

    An entity path answers "what is this data about", replacing the task enums and
    estimation/ground-truth flags that used to be encoded in types.

    Examples:
        >>> EntityPath.parse("/ground_truth/objects")
        EntityPath(parts=('ground_truth', 'objects'))
        >>> str(EntityPath.parse("/estimation") / "objects")
        '/estimation/objects'
        >>> str(EntityPath.root())
        '/'
    """

    parts: tuple[str, ...] = field(converter=_as_parts)

    @classmethod
    def root(cls) -> Self:
        """Return the root path ``/``."""
        return cls(())

    @classmethod
    def parse(cls, path: str) -> Self:
        """Parse a ``/``-separated path string.

        Leading and trailing separators are ignored; empty segments are rejected.
        """
        if not isinstance(path, str):
            raise TypeError(f"EntityPath.parse expects a str, got {type(path).__name__}")

        stripped = path.strip()
        if stripped in ("", _SEPARATOR):
            return cls(())

        body = stripped.strip(_SEPARATOR)
        if body == "":
            return cls(())

        segments = body.split(_SEPARATOR)
        if any(segment == "" for segment in segments):
            raise ValueError(f"EntityPath must not contain empty segments, got {path!r}")

        return cls(tuple(segments))

    def __str__(self) -> str:
        return _SEPARATOR + _SEPARATOR.join(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __iter__(self) -> Iterator[str]:
        return iter(self.parts)

    def __truediv__(self, other: str | EntityPath) -> Self:
        """Append a child segment or a relative path."""
        suffix = other.parts if isinstance(other, EntityPath) else EntityPath.parse(other).parts
        return type(self)(self.parts + suffix)

    @property
    def name(self) -> str | None:
        """Return the last segment, or ``None`` for the root path."""
        return self.parts[-1] if self.parts else None

    @property
    def parent(self) -> Self | None:
        """Return the parent path, or ``None`` for the root path."""
        if not self.parts:
            return None
        return type(self)(self.parts[:-1])

    def is_descendant_of(self, other: EntityPath) -> bool:
        """Return whether this path is a strict descendant of ``other``."""
        return len(self.parts) > len(other.parts) and self.parts[: len(other.parts)] == other.parts

    def starts_with(self, other: EntityPath) -> bool:
        """Return whether this path equals ``other`` or is a descendant of it."""
        return self.parts[: len(other.parts)] == other.parts


def as_entity_path(value: str | EntityPath) -> EntityPath:
    """Coerce a string or an :class:`EntityPath` into an :class:`EntityPath`."""
    return value if isinstance(value, EntityPath) else EntityPath.parse(value)


EntityPathLike: TypeAlias = Union[str, EntityPath]
