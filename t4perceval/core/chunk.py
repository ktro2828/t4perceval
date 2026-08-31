from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, define, field

from t4perceval.core.entity import EntityPath, as_entity_path
from t4perceval.core.selection import normalize_selection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

    from t4perceval.core.component import Component
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.timeline import TimeColumn, Timeline
    from t4perceval.typing import ArrayLike, NDArrayI64, SelectionLike

__all__ = ("Chunk", "concat_chunks")


def _as_offsets(value: ArrayLike) -> NDArrayI64:
    offsets = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    if offsets.ndim != 1:
        raise ValueError(f"Chunk.offsets must have shape (P + 1,), got {offsets.shape}")
    if offsets.size == 0 or offsets[0] != 0:
        raise ValueError("Chunk.offsets must start with 0")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("Chunk.offsets must be monotonically non-decreasing")
    if offsets.flags.writeable:
        offsets = offsets.copy()
        offsets.flags.writeable = False
    return offsets


def _as_columns(
    value: Mapping[ComponentDescriptor, Component],
) -> dict[ComponentDescriptor, Component]:
    return dict(value)


@define(frozen=True, slots=True)
class Chunk:
    """A column-oriented table of components for one entity path.

    One **row is one object**, and :attr:`offsets` marks the frame boundaries. This is a
    deliberate flattening: evaluation math is elementwise over objects, so keeping rows
    flat lets every system work on plain contiguous arrays instead of walking list
    offsets per frame.

    Attributes:
        entity_path: Which stream this data belongs to.
        indexes: One time column per timeline, each with one time per *partition*.
        offsets: Row boundaries of the partitions, length ``num_partitions + 1``.
        columns: The component columns, each with ``num_rows`` rows.
        frame_id: Coordinate frame that every row in this chunk is expressed in.
        is_static: When set, this data applies to every point on every timeline.
    """

    entity_path: EntityPath = field(converter=as_entity_path)
    indexes: tuple[TimeColumn, ...] = field(converter=tuple)
    offsets: NDArrayI64 = field(converter=_as_offsets, eq=cmp_using(eq=np.array_equal))
    columns: dict[ComponentDescriptor, Component] = field(converter=_as_columns)
    frame_id: str | None = field(default=None, kw_only=True)
    is_static: bool = field(default=False, kw_only=True)

    def __attrs_post_init__(self) -> None:
        names = [index.timeline.name for index in self.indexes]
        if len(set(names)) != len(names):
            raise ValueError(f"Chunk has duplicate timelines: {sorted(names)}")

        for index in self.indexes:
            if len(index) != self.num_partitions:
                raise ValueError(
                    f"Index column {index.timeline.name!r} has {len(index)} times, "
                    f"expected {self.num_partitions}",
                )

        for descriptor, column in self.columns.items():
            if len(column) != self.num_rows:
                raise ValueError(
                    f"Column {descriptor.component!r} has length {len(column)}, "
                    f"expected {self.num_rows}",
                )

        if self.is_static:
            if self.indexes:
                raise ValueError("A static chunk must not carry index columns")
            if self.num_partitions != 1:
                raise ValueError(
                    f"A static chunk must have exactly one partition, got {self.num_partitions}",
                )

    @classmethod
    def from_columns(
        cls,
        entity_path: EntityPathLike,
        columns: Mapping[ComponentDescriptor, Component],
        *,
        indexes: Sequence[TimeColumn] = (),
        frame_id: str | None = None,
        is_static: bool = False,
    ) -> Self:
        """Build a single-partition chunk holding every row of ``columns``."""
        lengths = {len(column) for column in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"Columns have mismatched lengths: {sorted(lengths)}")
        num_rows = lengths.pop() if lengths else 0
        return cls(
            entity_path,
            tuple(indexes),
            np.array([0, num_rows], dtype=np.int64),
            columns,
            frame_id=frame_id,
            is_static=is_static,
        )

    @property
    def num_rows(self) -> int:
        return int(self.offsets[-1])

    @property
    def num_partitions(self) -> int:
        return int(self.offsets.shape[0] - 1)

    @property
    def timelines(self) -> tuple[Timeline, ...]:
        return tuple(index.timeline for index in self.indexes)

    @property
    def descriptors(self) -> tuple[ComponentDescriptor, ...]:
        return tuple(self.columns)

    def __len__(self) -> int:
        return self.num_rows

    def has(self, *descriptors: ComponentDescriptor) -> bool:
        """Return whether every given descriptor has a column in this chunk."""
        return all(descriptor in self.columns for descriptor in descriptors)

    def component(self, descriptor: ComponentDescriptor) -> Component | None:
        """Return one column, or ``None`` when this chunk does not carry it."""
        return self.columns.get(descriptor)

    def index(self, timeline: Timeline) -> TimeColumn | None:
        """Return the index column for ``timeline``, or ``None`` when absent."""
        for candidate in self.indexes:
            if candidate.timeline == timeline:
                return candidate
        return None

    def partition(self, index: int) -> slice:
        """Return the row range of one partition."""
        if index < 0:
            index += self.num_partitions
        if not 0 <= index < self.num_partitions:
            raise IndexError("Chunk partition index out of range")
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))

    def partition_sizes(self) -> NDArrayI64:
        """Return the number of rows in each partition."""
        return np.diff(self.offsets)

    def partition_ids(self) -> NDArrayI64:
        """Return the partition index of every row, with shape ``(num_rows,)``."""
        return np.repeat(
            np.arange(self.num_partitions, dtype=np.int64),
            self.partition_sizes(),
        )

    def times_for_rows(self, timeline: Timeline) -> NDArrayI64:
        """Return the time of every row on ``timeline``, with shape ``(num_rows,)``."""
        index = self.index(timeline)
        if index is None:
            raise KeyError(f"Chunk has no index column for timeline {timeline.name!r}")
        return np.repeat(index.times, self.partition_sizes())

    def select(self, selection: SelectionLike) -> Self:
        """Return a chunk holding the selected rows, keeping the partition structure.

        The selection must not reorder rows across partitions -- boolean masks and
        ascending index arrays always satisfy this. Partitions may become empty; their
        index entries are kept so the time axis is preserved.
        """
        indices = normalize_selection(selection, length=self.num_rows)
        picked = self.partition_ids()[indices]
        if picked.size > 1 and np.any(np.diff(picked) < 0):
            raise ValueError(
                "Chunk.select() requires a selection that does not reorder partitions",
            )

        sizes = np.bincount(picked, minlength=self.num_partitions).astype(np.int64, copy=False)
        offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64, copy=False)

        return type(self)(
            self.entity_path,
            self.indexes,
            offsets,
            {descriptor: column.select(indices) for descriptor, column in self.columns.items()},
            frame_id=self.frame_id,
            is_static=self.is_static,
        )

    def select_partitions(self, selection: SelectionLike) -> Self:
        """Return a chunk keeping only the selected partitions, in ascending order."""
        partitions = normalize_selection(selection, length=self.num_partitions)
        if partitions.size > 1 and np.any(np.diff(partitions) < 0):
            raise ValueError("Chunk.select_partitions() requires an ascending selection")

        rows = np.concatenate(
            [np.arange(self.offsets[p], self.offsets[p + 1], dtype=np.int64) for p in partitions]
            or [np.empty(0, dtype=np.int64)],
        )
        sizes = self.partition_sizes()[partitions]
        offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64, copy=False)

        return type(self)(
            self.entity_path,
            tuple(index.select_partitions(partitions) for index in self.indexes),
            offsets,
            {descriptor: column.select(rows) for descriptor, column in self.columns.items()},
            frame_id=self.frame_id,
            is_static=self.is_static,
        )

    def with_columns(self, columns: Mapping[ComponentDescriptor, Component]) -> Self:
        """Return a chunk with additional or replaced columns."""
        return type(self)(
            self.entity_path,
            self.indexes,
            self.offsets,
            {**self.columns, **columns},
            frame_id=self.frame_id,
            is_static=self.is_static,
        )


def concat_chunks(chunks: Sequence[Chunk]) -> Chunk:
    """Concatenate chunks that share an entity path, timelines, and column set.

    Partitions are appended in the given order; no sorting by time is performed.
    """
    if not chunks:
        raise ValueError("concat_chunks() requires at least one chunk")
    if len(chunks) == 1:
        return chunks[0]

    head = chunks[0]
    for chunk in chunks[1:]:
        if chunk.entity_path != head.entity_path:
            raise ValueError(
                f"Cannot concatenate chunks of different entities: "
                f"{head.entity_path} and {chunk.entity_path}",
            )
        if chunk.timelines != head.timelines:
            raise ValueError("Cannot concatenate chunks with different timelines")
        if set(chunk.columns) != set(head.columns):
            raise ValueError("Cannot concatenate chunks with different columns")
        if chunk.is_static != head.is_static:
            raise ValueError("Cannot concatenate static and temporal chunks")
        if chunk.frame_id != head.frame_id:
            raise ValueError(
                f"Cannot concatenate chunks in different frames: "
                f"{head.frame_id!r} and {chunk.frame_id!r}",
            )

    from t4perceval.core.timeline import TimeColumn

    sizes = np.concatenate([chunk.partition_sizes() for chunk in chunks])
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64, copy=False)

    indexes = tuple(
        TimeColumn(
            timeline,
            np.concatenate([chunk.index(timeline).times for chunk in chunks]),
        )
        for timeline in head.timelines
    )

    columns = {
        descriptor: type(column)(
            np.concatenate([chunk.columns[descriptor].values for chunk in chunks]),
        )
        for descriptor, column in head.columns.items()
    }

    return Chunk(
        head.entity_path,
        indexes,
        offsets,
        columns,
        frame_id=head.frame_id,
        is_static=head.is_static,
    )
