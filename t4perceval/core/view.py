from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, define, field

from t4perceval.core.archetype import ArchetypeT
from t4perceval.core.selection import normalize_selection

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.component import Component
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.core.timeline import Timeline
    from t4perceval.typing import NDArrayI64, SelectionLike

__all__ = ("EntityView",)


def _broadcast_static(column: Component, num_rows: int) -> Component:
    """Expand a one-row static column to ``num_rows`` identical rows."""
    if len(column) == num_rows:
        return column
    if len(column) != 1:
        raise ValueError(
            f"Static column has {len(column)} rows; only a single row can be broadcast "
            f"to the {num_rows} row(s) of a view",
        )
    expanded = np.broadcast_to(column.values[0], (num_rows, *column.values.shape[1:]))
    return type(column)(expanded)


@define(frozen=True, slots=True)
class EntityView:
    """A lazy window onto the rows of one entity.

    :meth:`select` only composes row indices -- nothing is copied until
    :meth:`component`, :meth:`materialize` or :meth:`to_chunk` is called. That is the
    division of labour this package keeps throughout: ``select()`` on a component or an
    archetype produces independent data, whereas a view defers the copy.
    """

    chunk: Chunk
    indices: NDArrayI64 = field(eq=cmp_using(eq=np.array_equal))
    static: dict[ComponentDescriptor, Component] = field(factory=dict, converter=dict)

    @indices.default
    def _all_rows(self) -> NDArrayI64:
        return np.arange(self.chunk.num_rows, dtype=np.int64)

    @classmethod
    def over(
        cls,
        chunk: Chunk,
        *,
        static: Mapping[ComponentDescriptor, Component] | None = None,
    ) -> Self:
        """Return a view covering every row of ``chunk``."""
        return cls(chunk, np.arange(chunk.num_rows, dtype=np.int64), dict(static or {}))

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def entity_path(self) -> EntityPath:
        return self.chunk.entity_path

    @property
    def frame_id(self) -> str | None:
        return self.chunk.frame_id

    @property
    def descriptors(self) -> tuple[ComponentDescriptor, ...]:
        """Return every descriptor reachable through this view."""
        return tuple({**self.chunk.columns, **self.static})

    def has(self, *descriptors: ComponentDescriptor) -> bool:
        """Return whether every given descriptor is reachable through this view."""
        return all(
            descriptor in self.static or descriptor in self.chunk.columns
            for descriptor in descriptors
        )

    def select(self, selection: SelectionLike) -> Self:
        """Return a narrower view. Row indices are composed; no data is copied."""
        picked = normalize_selection(selection, length=len(self))
        return type(self)(self.chunk, self.indices[picked], self.static)

    def component(self, descriptor: ComponentDescriptor) -> Component | None:
        """Materialize one column, or return ``None`` when it is not present.

        Static data wins over temporal data for the same descriptor, and a one-row static
        column is broadcast across the view.
        """
        static = self.static.get(descriptor)
        if static is not None:
            return _broadcast_static(static, len(self))

        column = self.chunk.columns.get(descriptor)
        return None if column is None else column.select(self.indices)

    def materialize(self, archetype: type[ArchetypeT]) -> ArchetypeT:
        """Build ``archetype`` from the columns this view exposes."""
        columns: dict[ComponentDescriptor, Component] = {}
        for descriptor in archetype.descriptors():
            column = self.component(descriptor)
            if column is not None:
                columns[descriptor] = column
        return archetype.from_components(columns)

    def to_chunk(self) -> Chunk:
        """Materialize the selected rows as a chunk, keeping the partition structure."""
        return self.chunk.select(self.indices)

    def partition_ids(self) -> NDArrayI64:
        """Return the partition index of every row in this view."""
        return self.chunk.partition_ids()[self.indices]

    def times(self, timeline: Timeline) -> NDArrayI64:
        """Return the time of every row in this view on ``timeline``."""
        return self.chunk.times_for_rows(timeline)[self.indices]
