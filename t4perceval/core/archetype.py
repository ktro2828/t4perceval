from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
from attrs import NOTHING, define, fields

from t4perceval.core.chunk import Chunk
from t4perceval.core.component import ColumnarComponent, validate_lengths
from t4perceval.core.selection import normalize_selection
from t4perceval.core.timeline import TimeColumn

if TYPE_CHECKING:
    from collections.abc import Mapping

    from attrs import Attribute
    from typing_extensions import Self

    from t4perceval.core.component import Component
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.timeline import TimePoint
    from t4perceval.typing import SelectionLike

__all__ = ("Archetype", "ArchetypeT", "COMPONENT_METADATA_KEY", "as_component")

#: Key under which a field stores its ``(descriptor, component type)`` pair.
COMPONENT_METADATA_KEY = "t4perceval.component"

ArchetypeT = TypeVar("ArchetypeT", bound="Archetype")


def as_component(value: Any, component_type: type[Component]) -> Component:
    """Coerce a value into ``component_type``.

    A component of another type is re-wrapped through its raw array, so converting for
    example a :class:`BatchVector3D` into a :class:`BatchPosition3D` stays explicit and
    cheap rather than being silently accepted.
    """
    if isinstance(value, component_type):
        return value
    if isinstance(value, ColumnarComponent):
        return component_type(value.values)
    return component_type(value)


@define(frozen=True, slots=True)
class Archetype:
    """A coherent bundle of components describing one kind of thing.

    Archetypes are composed, never subclassed: :class:`BatchTracking3D` re-declares the
    box components of :class:`BatchDetection3D` and adds an instance id, rather than
    inheriting from it. ``isinstance`` therefore stays a truthful claim, and "can this be
    treated as a detection?" is answered by :meth:`has`, which is the same question a
    system asks through its ``REQUIRES``.

    Subclasses only declare fields; length validation, row selection and the chunk
    round-trip are implemented once, here.
    """

    ARCHETYPE_NAME: ClassVar[str] = ""

    # -- class-level introspection ----------------------------------------------------

    @classmethod
    def archetype_name(cls) -> str:
        """Return the archetype's name, defaulting to the class name."""
        return cls.ARCHETYPE_NAME or cls.__name__

    @classmethod
    def component_fields(cls) -> tuple[Attribute[Any], ...]:
        """Return the attrs fields that hold components, in declaration order."""
        return tuple(
            attribute for attribute in fields(cls) if COMPONENT_METADATA_KEY in attribute.metadata
        )

    @classmethod
    def descriptors(cls) -> tuple[ComponentDescriptor, ...]:
        """Return every descriptor this archetype can carry."""
        return tuple(cls._descriptor_of(attribute) for attribute in cls.component_fields())

    @classmethod
    def required_descriptors(cls) -> tuple[ComponentDescriptor, ...]:
        """Return the descriptors that must always be present."""
        return tuple(
            cls._descriptor_of(attribute)
            for attribute in cls.component_fields()
            if attribute.default is NOTHING
        )

    @classmethod
    def optional_descriptors(cls) -> tuple[ComponentDescriptor, ...]:
        """Return the descriptors that may be absent."""
        return tuple(
            cls._descriptor_of(attribute)
            for attribute in cls.component_fields()
            if attribute.default is not NOTHING
        )

    @classmethod
    def descriptor_of(cls, field_name: str) -> ComponentDescriptor:
        """Return the descriptor declared by one field."""
        for attribute in cls.component_fields():
            if attribute.name == field_name:
                return cls._descriptor_of(attribute)
        raise KeyError(f"{cls.__name__} has no component field named {field_name!r}")

    @classmethod
    def _descriptor_of(cls, attribute: Attribute[Any]) -> ComponentDescriptor:
        descriptor, _ = attribute.metadata[COMPONENT_METADATA_KEY]
        # Annotate provenance only; `tagged` keeps `component`, so identity is unchanged
        # and the same column is addressable from every archetype that declares it.
        return descriptor.tagged(cls.archetype_name())

    @staticmethod
    def _component_type_of(attribute: Attribute[Any]) -> type[Component]:
        _, component_type = attribute.metadata[COMPONENT_METADATA_KEY]
        return component_type

    # -- validation and sizing --------------------------------------------------------

    def __attrs_post_init__(self) -> None:
        columns = {
            attribute.name: getattr(self, attribute.name) for attribute in self.component_fields()
        }
        validate_lengths(len(self), **columns)

    def __len__(self) -> int:
        for attribute in self.component_fields():
            column = getattr(self, attribute.name)
            if column is not None:
                return len(column)
        return 0

    def has(self, *descriptors: ComponentDescriptor) -> bool:
        """Return whether every given descriptor is present on this instance.

        This is the composition-friendly replacement for ``isinstance`` checks against a
        base archetype::

            tracking.has(*BatchDetection3D.required_descriptors())
        """
        present = self.as_components()
        return all(descriptor in present for descriptor in descriptors)

    # -- component mapping ------------------------------------------------------------

    def as_components(self) -> dict[ComponentDescriptor, Component]:
        """Return the present components keyed by descriptor."""
        result: dict[ComponentDescriptor, Component] = {}
        for attribute in self.component_fields():
            column = getattr(self, attribute.name)
            if column is not None:
                result[self._descriptor_of(attribute)] = column
        return result

    @classmethod
    def from_components(cls, columns: Mapping[ComponentDescriptor, Component]) -> Self:
        """Build an archetype from a descriptor-keyed mapping.

        Optional components may be absent; required ones may not.
        """
        kwargs: dict[str, Component | None] = {}
        missing: list[str] = []
        for attribute in cls.component_fields():
            descriptor = cls._descriptor_of(attribute)
            column = columns.get(descriptor)
            if column is None and attribute.default is NOTHING:
                missing.append(descriptor.component)
            kwargs[attribute.name] = column

        if missing:
            raise ValueError(
                f"{cls.__name__} is missing required component(s): {', '.join(missing)}",
            )
        return cls(**{name: column for name, column in kwargs.items() if column is not None})

    # -- row selection ----------------------------------------------------------------

    def select(self, selection: SelectionLike) -> Self:
        """Return a new archetype holding an independent copy of the selected rows.

        Implemented once for every archetype by walking the declared fields. Use
        :class:`~t4perceval.core.view.EntityView` when a lazy, non-copying view is wanted.
        """
        indices = normalize_selection(selection, length=len(self))
        changes: dict[str, Component | None] = {}
        for attribute in self.component_fields():
            column = getattr(self, attribute.name)
            changes[attribute.name] = None if column is None else column.select(indices)
        return type(self)(**changes)

    # -- chunk round-trip -------------------------------------------------------------

    def to_chunk(
        self,
        entity_path: EntityPathLike,
        *,
        at: TimePoint | None = None,
        frame_id: str | None = None,
        is_static: bool = False,
    ) -> Chunk:
        """Wrap this archetype as a single-partition chunk observed at ``at``."""
        if at is None and not is_static:
            raise ValueError("to_chunk() requires either a TimePoint or is_static=True")

        indexes = (
            ()
            if at is None
            else tuple(
                TimeColumn(timeline, np.array([at[timeline]], dtype=np.int64))
                for timeline in at.timelines
            )
        )
        return Chunk.from_columns(
            entity_path,
            self.as_components(),
            indexes=indexes,
            frame_id=frame_id,
            is_static=is_static,
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Self:
        """Build an archetype from every matching column of ``chunk``."""
        return cls.from_components(chunk.columns)
