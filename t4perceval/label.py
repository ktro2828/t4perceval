"""Meaning for the numeric identifier columns.

:class:`BatchClassId` and :class:`BatchInstanceId` are plain integer columns; the
registries here are the only place that says what those integers mean. They are metadata
rather than columns -- names are not numeric, and a registry describes a whole recording
rather than a row -- so a registry travels through
:class:`~t4perceval.system.base.SystemContext` and through Arrow schema metadata, in the
role Rerun gives to a static ``AnnotationContext``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from typing_extensions import Self

    from t4perceval.typing import NDArrayI32

__all__ = ("UNKNOWN_CLASS_ID", "ClassInfo", "InstanceRegistry", "LabelRegistry")

#: Class id reserved for "no known class".
UNKNOWN_CLASS_ID = -1


@define(frozen=True, slots=True)
class ClassInfo:
    """One canonical class of a :class:`LabelRegistry`."""

    class_id: int = field(converter=int)
    name: str = field(converter=str)
    color: tuple[int, int, int] | None = field(default=None)

    def __attrs_post_init__(self) -> None:
        if self.name == "":
            raise ValueError("ClassInfo.name must not be empty")
        if self.color is not None and len(self.color) != 3:
            raise ValueError(f"ClassInfo.color must be an RGB triple, got {self.color!r}")


def _as_classes(value: Iterable[ClassInfo]) -> tuple[ClassInfo, ...]:
    classes = tuple(value)

    ids = [info.class_id for info in classes]
    if len(set(ids)) != len(ids):
        raise ValueError("LabelRegistry has duplicate class ids")

    names = [info.name for info in classes]
    if len(set(names)) != len(names):
        raise ValueError("LabelRegistry has duplicate class names")

    if UNKNOWN_CLASS_ID in ids:
        raise ValueError(f"class id {UNKNOWN_CLASS_ID} is reserved for the unknown class")

    return classes


def _as_aliases(
    value: Mapping[str, int] | Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = value.items() if hasattr(value, "items") else value
    return tuple(sorted(((str(name), int(class_id)) for name, class_id in items)))


@define(frozen=True, slots=True)
class LabelRegistry:
    """Maps class names to the integers stored in a :class:`BatchClassId` column.

    Replaces ``LabelConverter`` together with its ``label_prefix`` and
    ``merge_similar_labels`` options: merging produces a *new* registry whose aliases
    point several dataset names at one canonical class, so the merge is visible in the
    data rather than hidden in a flag read at match time.

    Examples:
        >>> registry = LabelRegistry.from_names(["car", "truck", "pedestrian"])
        >>> registry.class_id("truck")
        1
        >>> merged = registry.merged({"vehicle": ["car", "truck"]})
        >>> merged.class_id("car") == merged.class_id("truck") == merged.class_id("vehicle")
        True
    """

    classes: tuple[ClassInfo, ...] = field(converter=_as_classes)
    aliases: tuple[tuple[str, int], ...] = field(default=(), converter=_as_aliases, kw_only=True)
    prefix: str = field(default="autoware", converter=str, kw_only=True)

    def __attrs_post_init__(self) -> None:
        known = {info.class_id for info in self.classes}
        for name, class_id in self.aliases:
            if class_id not in known:
                raise ValueError(f"Alias {name!r} points at unknown class id {class_id}")

    @classmethod
    def from_names(
        cls,
        names: Sequence[str],
        *,
        prefix: str = "autoware",
        colors: Mapping[str, tuple[int, int, int]] | None = None,
    ) -> Self:
        """Build a registry assigning ids ``0..K-1`` in the given order.

        Repeated names are ignored rather than rejected, so a label list assembled from
        several configuration files does not need to be de-duplicated by the caller.
        """
        colors = colors or {}
        seen: dict[str, None] = dict.fromkeys(names)
        return cls(
            tuple(
                ClassInfo(class_id, name, colors.get(name)) for class_id, name in enumerate(seen)
            ),
            prefix=prefix,
        )

    def __len__(self) -> int:
        return len(self.classes)

    def __contains__(self, name: str) -> bool:
        return self.class_id_or(name, UNKNOWN_CLASS_ID) != UNKNOWN_CLASS_ID

    @property
    def names(self) -> tuple[str, ...]:
        """Return the canonical class names, ordered by class id."""
        return tuple(info.name for info in sorted(self.classes, key=lambda i: i.class_id))

    def class_id(self, name: str) -> int:
        """Return the class id of ``name``, following aliases."""
        class_id = self.class_id_or(name, UNKNOWN_CLASS_ID)
        if class_id == UNKNOWN_CLASS_ID:
            raise KeyError(f"Unknown class name {name!r}; known: {sorted(self.names)}")
        return class_id

    def class_id_or(self, name: str, default: int) -> int:
        """Return the class id of ``name``, or ``default`` when it is not known."""
        for info in self.classes:
            if info.name == name:
                return info.class_id
        for alias, class_id in self.aliases:
            if alias == name:
                return class_id
        return default

    def name(self, class_id: int) -> str:
        """Return the canonical name of ``class_id``."""
        for info in self.classes:
            if info.class_id == class_id:
                return info.name
        raise KeyError(f"Unknown class id {class_id}")

    def info(self, class_id: int) -> ClassInfo:
        """Return the full class info of ``class_id``."""
        for candidate in self.classes:
            if candidate.class_id == class_id:
                return candidate
        raise KeyError(f"Unknown class id {class_id}")

    def encode(self, names: Sequence[str], *, strict: bool = True) -> NDArrayI32:
        """Encode class names into a column of class ids.

        Args:
            names: The names to encode.
            strict: When set, an unknown name raises; otherwise it becomes
                :data:`UNKNOWN_CLASS_ID`.
        """
        if strict:
            return np.array([self.class_id(name) for name in names], dtype=np.int32)
        return np.array(
            [self.class_id_or(name, UNKNOWN_CLASS_ID) for name in names],
            dtype=np.int32,
        )

    def decode(self, class_ids: Sequence[int]) -> tuple[str, ...]:
        """Decode a column of class ids back into canonical names."""
        return tuple(self.name(int(class_id)) for class_id in class_ids)

    def merged(self, groups: Mapping[str, Sequence[str]]) -> Self:
        """Return a registry where each group of names collapses into one class.

        Names not mentioned in ``groups`` keep their own class. Group names are assigned
        first, in the order given, so the resulting ids are deterministic.
        """
        grouped = {name for members in groups.values() for name in members}
        unknown = sorted(grouped - set(self.names) - {alias for alias, _ in self.aliases})
        if unknown:
            raise KeyError(f"Cannot merge unknown class name(s): {unknown}")

        survivors = [name for name in self.names if name not in grouped]
        canonical = [*groups, *survivors]

        classes = tuple(
            ClassInfo(class_id, name, self._color_of(name))
            for class_id, name in enumerate(canonical)
        )
        lookup = {info.name: info.class_id for info in classes}

        aliases = {
            member: lookup[group]
            for group, members in groups.items()
            for member in members
            if member not in lookup
        }
        for alias, class_id in self.aliases:
            if alias not in lookup and alias not in aliases:
                name = self.name(class_id)
                target = next(
                    (lookup[group] for group, members in groups.items() if name in members),
                    lookup.get(name),
                )
                if target is not None:
                    aliases[alias] = target

        return type(self)(classes, aliases=aliases, prefix=self.prefix)

    def _color_of(self, name: str) -> tuple[int, int, int] | None:
        for info in self.classes:
            if info.name == name:
                return info.color
        return None

    # -- metadata round-trip ----------------------------------------------------------

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible mapping, for Arrow schema metadata."""
        return {
            "prefix": self.prefix,
            "classes": [
                {"class_id": info.class_id, "name": info.name, "color": list(info.color)}
                if info.color is not None
                else {"class_id": info.class_id, "name": info.name}
                for info in self.classes
            ],
            "aliases": {name: class_id for name, class_id in self.aliases},
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Self:
        """Rebuild a registry from :meth:`to_metadata`."""
        raw_classes = metadata.get("classes") or ()
        classes = tuple(
            ClassInfo(
                int(entry["class_id"]),
                str(entry["name"]),
                tuple(entry["color"]) if entry.get("color") is not None else None,
            )
            for entry in raw_classes  # type: ignore[union-attr]
        )
        return cls(
            classes,
            aliases=metadata.get("aliases") or {},  # type: ignore[arg-type]
            prefix=str(metadata.get("prefix", "autoware")),
        )


class InstanceRegistry:
    """Interns dataset UUID strings into the integers of a ``BatchInstanceId`` column.

    Mutable on purpose: ids are handed out as objects are first seen while a scene is
    loaded. Ids are stable for the lifetime of one registry, which is what tracking
    metrics need in order to compare identities across frames.
    """

    def __init__(self) -> None:
        self._by_uuid: dict[str, int] = {}
        self._by_id: list[str] = []

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, uuid: str) -> bool:
        return uuid in self._by_uuid

    def intern(self, uuid: str) -> int:
        """Return the id of ``uuid``, assigning the next free one if it is new."""
        known = self._by_uuid.get(uuid)
        if known is not None:
            return known
        assigned = len(self._by_id)
        self._by_uuid[uuid] = assigned
        self._by_id.append(uuid)
        return assigned

    def encode(self, uuids: Sequence[str]) -> np.ndarray:
        """Intern a sequence of UUIDs into an instance id column."""
        return np.array([self.intern(uuid) for uuid in uuids], dtype=np.int64)

    def instance_id(self, uuid: str) -> int:
        """Return the id of an already interned ``uuid``.

        Unlike :meth:`intern` this never assigns a new id, so looking up a UUID that is
        not in the recording is an error rather than a silently fresh identity. That is
        what filtering needs: a mistyped UUID must not quietly match nothing.
        """
        known = self._by_uuid.get(uuid)
        if known is None:
            raise KeyError(f"Unknown instance uuid {uuid!r}")
        return known

    def instance_id_or(self, uuid: str, default: int) -> int:
        """Return the id of ``uuid``, or ``default`` when it was never interned."""
        return self._by_uuid.get(uuid, default)

    def uuid(self, instance_id: int) -> str:
        """Return the UUID an instance id was interned from."""
        if not 0 <= instance_id < len(self._by_id):
            raise KeyError(f"Unknown instance id {instance_id}")
        return self._by_id[instance_id]

    def decode(self, instance_ids: Sequence[int]) -> tuple[str, ...]:
        """Return the UUIDs of a column of instance ids."""
        return tuple(self.uuid(int(instance_id)) for instance_id in instance_ids)
