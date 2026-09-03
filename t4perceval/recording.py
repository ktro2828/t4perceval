"""An immutable log plus the meaning of the integers inside it.

A :class:`~t4perceval.core.store.Store` says *what rows exist*. It deliberately does not
say what the integers in those rows mean: a ``class_id`` of ``3`` is only interpretable
against the :class:`~t4perceval.label.LabelRegistry` that produced it. A
:class:`Recording` binds the two together, adds provenance, and drops the write methods.

That binding is the point. Class ids are assigned in first-seen order, so two registries
built independently over the same categories are both valid and silently incompatible;
carrying a store around without the registry that encoded it is what makes that mistake
possible. An importer therefore returns a ``Recording``, never a bare ``Store``.

A recording is not runnable. :meth:`~t4perceval.system.base.Pipeline.run` writes results
back into ``ctx.store``, so evaluating one means materializing the parts you want into a
fresh, writable store first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from attrs import define, evolve, field

from t4perceval.core.store import Store
from t4perceval.label import InstanceRegistry, LabelRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.component import Component
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.timeline import Timeline, TimeRange
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayI64

__all__ = ("Recording", "RecordingMetadata", "SourceInfo")


def _as_str_pairs(
    value: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    items = value.items() if hasattr(value, "items") else value
    return tuple((str(key), str(item)) for key, item in items)


@define(frozen=True, slots=True)
class SourceInfo:
    """Where one stream of a recording came from.

    Enough to find the original data again, and no more. Reconstructing a run from this
    is explicitly not a goal -- that would mean serializing arbitrary importer
    configuration, which is a later feature if it is ever one at all.
    """

    kind: str = field(converter=str)
    """What produced it: ``"t4"``, ``"rosbag"``, ``"synthetic"``."""

    uri: str = field(converter=str)
    """Dataset root, bag path, or another locator meaningful to ``kind``."""

    version: str | None = field(default=None, kw_only=True)
    scene: str | None = field(default=None, kw_only=True)
    topic: str | None = field(default=None, kw_only=True)
    entity_path: str | None = field(default=None, kw_only=True)
    extra: tuple[tuple[str, str], ...] = field(default=(), converter=_as_str_pairs, kw_only=True)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-compatible mapping."""
        data: dict[str, object] = {"kind": self.kind, "uri": self.uri}
        for name in ("version", "scene", "topic", "entity_path"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> Self:
        """Rebuild from :meth:`to_json`."""
        return cls(
            str(data["kind"]),
            str(data["uri"]),
            version=_optional_str(data.get("version")),
            scene=_optional_str(data.get("scene")),
            topic=_optional_str(data.get("topic")),
            entity_path=_optional_str(data.get("entity_path")),
            extra=data.get("extra") or (),  # type: ignore[arg-type]
        )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


@define(frozen=True, slots=True)
class RecordingMetadata:
    """Provenance for a recording.

    Every field is JSON-compatible by construction, so persisting a recording is
    mechanical rather than a design problem of its own.
    """

    format_version: int = field(default=1, kw_only=True)
    t4perceval_version: str = field(default="", converter=str, kw_only=True)
    created_at_ns: int = field(default=0, converter=int, kw_only=True)
    sources: tuple[SourceInfo, ...] = field(default=(), converter=tuple, kw_only=True)

    labels_fingerprint: str = field(default="", converter=str, kw_only=True)
    """:meth:`~t4perceval.label.LabelRegistry.fingerprint` of the registry in force.

    Recorded so that two recordings can be checked for agreement about class ids before
    their data is evaluated together.
    """

    pipeline: tuple[str, ...] = field(default=(), converter=tuple, kw_only=True)
    """One entry per system that has run. Informational; not a reconstruction recipe."""

    frame_id: str | None = field(default=None, kw_only=True)
    tags: tuple[tuple[str, str], ...] = field(default=(), converter=_as_str_pairs, kw_only=True)
    notes: str = field(default="", converter=str, kw_only=True)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-compatible mapping."""
        return {
            "format_version": self.format_version,
            "t4perceval_version": self.t4perceval_version,
            "created_at_ns": self.created_at_ns,
            "sources": [source.to_json() for source in self.sources],
            "labels_fingerprint": self.labels_fingerprint,
            "pipeline": list(self.pipeline),
            "frame_id": self.frame_id,
            "tags": dict(self.tags),
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> Self:
        """Rebuild from :meth:`to_json`."""
        return cls(
            format_version=int(data.get("format_version", 1)),  # type: ignore[arg-type]
            t4perceval_version=str(data.get("t4perceval_version", "")),
            created_at_ns=int(data.get("created_at_ns", 0)),  # type: ignore[arg-type]
            sources=tuple(
                SourceInfo.from_json(entry)
                for entry in data.get("sources") or ()  # type: ignore[union-attr]
            ),
            labels_fingerprint=str(data.get("labels_fingerprint", "")),
            pipeline=tuple(str(entry) for entry in data.get("pipeline") or ()),  # type: ignore[union-attr]
            frame_id=_optional_str(data.get("frame_id")),
            tags=data.get("tags") or (),  # type: ignore[arg-type]
            notes=str(data.get("notes", "")),
        )


@define(frozen=True, slots=True)
class Recording:
    """An immutable log plus the registries that give its integer columns meaning.

    Read-only by omission: the query methods of :class:`~t4perceval.core.store.Store` are
    delegated, and ``send_chunk`` / ``log`` / ``log_static`` are not. The wrapped store is
    private for the same reason -- handing it out would hand out the write methods too.

    ``instances`` is exposed directly because it is mutable by design and interning
    during evaluation is legitimate. Mutating it *after* chunks have been encoded against
    it is not: the integers in the columns will not follow.
    """

    _store: Store = field(alias="store")
    labels: LabelRegistry = field()
    instances: InstanceRegistry = field(factory=InstanceRegistry)
    metadata: RecordingMetadata = field(factory=RecordingMetadata)

    @classmethod
    def of(
        cls,
        store: Store,
        *,
        labels: LabelRegistry,
        instances: InstanceRegistry | None = None,
        metadata: RecordingMetadata | None = None,
    ) -> Self:
        """Bind a store to the registries that encoded it.

        The fingerprint of ``labels`` is stamped into the metadata unless one is already
        recorded there, so the agreement check needs no separate bookkeeping.
        """
        resolved = metadata if metadata is not None else RecordingMetadata()
        if not resolved.labels_fingerprint:
            resolved = evolve(resolved, labels_fingerprint=labels.fingerprint())
        return cls(
            store=store,
            labels=labels,
            instances=instances if instances is not None else InstanceRegistry(),
            metadata=resolved,
        )

    # -- read-only delegation ----------------------------------------------------------

    def entity_paths(self) -> tuple[EntityPath, ...]:
        """Return every entity path, in insertion order."""
        return self._store.entity_paths()

    def timelines(self) -> tuple[Timeline, ...]:
        """Return every timeline any chunk is indexed on."""
        return self._store.timelines()

    def chunks(self, entity_path: EntityPathLike) -> tuple[Chunk, ...]:
        """Return an entity's chunks in log order.

        Order is load-bearing: ``latest_at`` breaks ties by preferring the most recently
        logged chunk, so anything that copies these elsewhere must preserve it.
        """
        return self._store.chunks(entity_path)

    def static(self, entity_path: EntityPathLike) -> dict[ComponentDescriptor, Component]:
        """Return an entity's static columns.

        Static data lives in the store rather than in a chunk, so it does not travel with
        one; moving an entity between stores means moving this too.
        """
        return self._store.static(entity_path)

    def times(self, entity_path: EntityPathLike, timeline: Timeline) -> NDArrayI64:
        """Return the sorted, unique times an entity was observed at."""
        return self._store.times(entity_path, timeline)

    def latest_at(
        self,
        entity_path: EntityPathLike,
        *,
        timeline: Timeline,
        at: int,
        components: Sequence[ComponentDescriptor] | None = None,
    ) -> EntityView:
        """Return the most recent partition at or before ``at``."""
        return self._store.latest_at(entity_path, timeline=timeline, at=at, components=components)

    def range(
        self,
        entity_path: EntityPathLike,
        *,
        timeline: Timeline,
        time_range: TimeRange,
        components: Sequence[ComponentDescriptor] | None = None,
    ) -> EntityView:
        """Return every partition inside ``time_range``, ordered by time."""
        return self._store.range(
            entity_path,
            timeline=timeline,
            time_range=time_range,
            components=components,
        )

    # -- derivation --------------------------------------------------------------------

    def with_metadata(self, **changes: Any) -> Self:
        """Return a copy whose metadata has ``changes`` applied."""
        return evolve(self, metadata=evolve(self.metadata, **changes))

    def agrees_with(self, other: Recording) -> bool:
        """Return whether ``other`` encodes class ids the same way this recording does."""
        return self.labels.fingerprint() == other.labels.fingerprint()
