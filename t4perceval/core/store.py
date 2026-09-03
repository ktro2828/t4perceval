from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from t4perceval.core.chunk import Chunk, concat_chunks
from t4perceval.core.entity import as_entity_path
from t4perceval.core.view import EntityView

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from t4perceval.core.archetype import Archetype
    from t4perceval.core.component import Component
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.timeline import TimePoint, TimeRange, Timeline
    from t4perceval.typing import NDArrayI64

__all__ = ("Store",)


class Store:
    """A mutable log of chunks, addressed by entity path and indexed along timelines.

    A store says *what rows exist*; it deliberately does not say what the integers in
    those rows mean. Pairing it with the registries that do is
    :class:`~t4perceval.recording.Recording`'s job.

    This replaces the ``Catalog -> Scenario -> Scene -> List[PerceptionFrameResult]``
    nesting of the original package. A single frame is
    :meth:`latest_at`; a whole scene is :meth:`range`; anything shared by every frame --
    the label registry, a trajectory time axis -- is :meth:`log_static`.

    Static data belongs to every timeline and takes precedence over temporal data
    carrying the same descriptor.
    """

    def __init__(self) -> None:
        self._temporal: dict[EntityPath, list[Chunk]] = {}
        self._static: dict[EntityPath, dict[ComponentDescriptor, Component]] = {}

    # -- writing ----------------------------------------------------------------------

    def send_chunk(self, chunk: Chunk) -> None:
        """Append a chunk. Static chunks merge into the entity's static columns."""
        if chunk.is_static:
            self._static.setdefault(chunk.entity_path, {}).update(chunk.columns)
        else:
            self._temporal.setdefault(chunk.entity_path, []).append(chunk)

    def log(
        self,
        entity_path: EntityPathLike,
        archetype: Archetype,
        *,
        at: TimePoint,
        frame_id: str | None = None,
    ) -> None:
        """Log one archetype as a single partition observed at ``at``."""
        self.send_chunk(archetype.to_chunk(entity_path, at=at, frame_id=frame_id))

    def log_static(self, entity_path: EntityPathLike, archetype: Archetype) -> None:
        """Log one archetype as static data, applying to every point on every timeline."""
        self.send_chunk(archetype.to_chunk(entity_path, is_static=True))

    def log_static_components(
        self,
        entity_path: EntityPathLike,
        columns: dict[ComponentDescriptor, Component],
    ) -> None:
        """Log loose components as static data."""
        self.send_chunk(Chunk.from_columns(entity_path, columns, is_static=True))

    # -- inspection -------------------------------------------------------------------

    def entity_paths(self) -> tuple[EntityPath, ...]:
        """Return every entity path with data, in insertion order."""
        seen = {**dict.fromkeys(self._temporal), **dict.fromkeys(self._static)}
        return tuple(seen)

    def timelines(self) -> tuple[Timeline, ...]:
        """Return every timeline any chunk is indexed along."""
        found: dict[Timeline, None] = {}
        for chunks in self._temporal.values():
            for chunk in chunks:
                for timeline in chunk.timelines:
                    found[timeline] = None
        return tuple(found)

    def static(self, entity_path: EntityPathLike) -> dict[ComponentDescriptor, Component]:
        """Return the static columns of one entity."""
        return dict(self._static.get(as_entity_path(entity_path), {}))

    def chunks(self, entity_path: EntityPathLike) -> tuple[Chunk, ...]:
        """Return the temporal chunks logged to one entity, in log order."""
        return tuple(self._temporal.get(as_entity_path(entity_path), ()))

    def times(self, entity_path: EntityPathLike, timeline: Timeline) -> NDArrayI64:
        """Return the sorted, unique partition times of one entity on ``timeline``."""
        times = [
            index.times
            for chunk in self.chunks(entity_path)
            if (index := chunk.index(timeline)) is not None
        ]
        if not times:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(times))

    # -- querying ---------------------------------------------------------------------

    def latest_at(
        self,
        entity_path: EntityPathLike,
        *,
        timeline: Timeline,
        at: int,
        components: Sequence[ComponentDescriptor] | None = None,
    ) -> EntityView:
        """Return a view of the most recent partition at or before ``at``.

        When several partitions share that time, the most recently logged one wins.
        """
        path = as_entity_path(entity_path)
        best: tuple[int, Chunk, int] | None = None

        for chunk in self.chunks(path):
            index = chunk.index(timeline)
            if index is None:
                continue
            eligible = np.flatnonzero(index.times <= at)
            if eligible.size == 0:
                continue
            partition = int(eligible[np.argmax(index.times[eligible])])
            time = int(index.times[partition])
            if best is None or time >= best[0]:
                best = (time, chunk, partition)

        static = self._restrict_static(path, components)
        if best is None:
            return EntityView.over(self._empty_chunk(path, components), static=static)

        _, chunk, partition = best
        return EntityView.over(
            self._restrict_columns(chunk.select_partitions([partition]), components),
            static=static,
        )

    def range(
        self,
        entity_path: EntityPathLike,
        *,
        timeline: Timeline,
        time_range: TimeRange,
        components: Sequence[ComponentDescriptor] | None = None,
    ) -> EntityView:
        """Return a view of every partition whose time falls inside ``time_range``.

        Partitions are ordered by time; ties keep their log order.
        """
        path = as_entity_path(entity_path)
        matches: list[tuple[int, int, int, Chunk]] = []

        for order, chunk in enumerate(self.chunks(path)):
            index = chunk.index(timeline)
            if index is None:
                continue
            for partition in np.flatnonzero(time_range.contains(index.times)):
                matches.append((int(index.times[partition]), order, int(partition), chunk))

        static = self._restrict_static(path, components)
        if not matches:
            return EntityView.over(self._empty_chunk(path, components), static=static)

        matches.sort(key=lambda match: (match[0], match[1], match[2]))
        pieces = [
            self._restrict_columns(chunk.select_partitions([partition]), components)
            for _, _, partition, chunk in matches
        ]
        return EntityView.over(concat_chunks(pieces), static=static)

    # -- helpers ----------------------------------------------------------------------

    @staticmethod
    def _restrict_columns(
        chunk: Chunk,
        components: Iterable[ComponentDescriptor] | None,
    ) -> Chunk:
        if components is None:
            return chunk
        wanted = set(components)
        return Chunk(
            chunk.entity_path,
            chunk.indexes,
            chunk.offsets,
            {
                descriptor: column
                for descriptor, column in chunk.columns.items()
                if descriptor in wanted
            },
            frame_id=chunk.frame_id,
            is_static=chunk.is_static,
        )

    def _restrict_static(
        self,
        path: EntityPath,
        components: Iterable[ComponentDescriptor] | None,
    ) -> dict[ComponentDescriptor, Component]:
        static = self._static.get(path, {})
        if components is None:
            return dict(static)
        wanted = set(components)
        return {descriptor: column for descriptor, column in static.items() if descriptor in wanted}

    @staticmethod
    def _empty_chunk(
        path: EntityPath,
        components: Iterable[ComponentDescriptor] | None,
    ) -> Chunk:
        del components  # An empty result carries no columns to restrict.
        return Chunk(path, (), np.array([0, 0], dtype=np.int64), {})
