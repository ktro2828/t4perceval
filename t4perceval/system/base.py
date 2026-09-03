"""The system layer: the "S" of ECS.

A system declares the components it needs and the components it produces, and runs
against any entity that carries them. That is what replaces the ``EvaluationTask`` enum
and the single ``evaluation_config_dict`` of the original package: an evaluation task is
no longer a value to branch on, it is the set of components present plus the pipeline you
compose. Every system writes its result back as a :class:`~t4perceval.core.chunk.Chunk`,
so intermediate products -- which rows a filter dropped, what a matcher scored -- stay in
the store instead of being discarded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from attrs import define, field

from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import FRAME, TimeRange

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.core.timeline import Timeline
    from t4perceval.core.view import EntityView
    from t4perceval.label import InstanceRegistry, LabelRegistry

__all__ = (
    "EntitySystem",
    "Pipeline",
    "System",
    "SystemContext",
    "require",
    "require_same_frame",
    "resolve_frame",
    "resolve_times",
)


@define(frozen=True, slots=True)
class SystemContext:
    """Everything a system reads that is not one of its own parameters.

    The two registries give meaning to the integer identifier columns, so a system can be
    configured with class names and UUIDs instead of the raw ids stored in the columns.
    """

    store: Store
    timeline: Timeline = field(default=FRAME)
    labels: LabelRegistry | None = field(default=None, kw_only=True)
    instances: InstanceRegistry | None = field(default=None, kw_only=True)


@runtime_checkable
class System(Protocol):
    """A transformation from components to components.

    Attributes:
        REQUIRES: Descriptors that must be present on each source entity.
        PROVIDES: Descriptors the system writes to its target entity.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]]
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]]

    @property
    def sources(self) -> tuple[EntityPath, ...]:
        """Entity paths this system reads."""
        ...

    @property
    def target(self) -> EntityPath:
        """Entity path this system writes."""
        ...

    @property
    def targets(self) -> tuple[EntityPath, ...]:
        """Every entity path this system writes.

        Usually just :attr:`target`. A system that produces several results from one
        shared computation -- MOTA, MOTP and ID switches all come out of the same identity
        tracking -- writes one entity per result and lists them all here, so
        :class:`Pipeline` can still tell who produces what.
        """
        ...

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]: ...


def resolve_frame(*views: EntityView) -> str | None:
    """Return the coordinate frame these views are in, or ``None`` when none states one.

    Row count is deliberately not consulted. A frame with no objects is still recorded in
    a frame -- the importer logs an empty frame with its ``frame_id`` intact -- and
    dropping that would make a system's output frame flicker between the stated value and
    ``None`` across a scene, which ``concat_chunks`` then refuses to join.
    """
    for view in views:
        if view.frame_id is not None:
            return view.frame_id
    return None


def require_same_frame(*views: EntityView) -> str | None:
    """Return the frame every view agrees on, raising when two of them disagree.

    Geometry across two frames is meaningless, and nothing downstream notices: subtracting
    a ``map`` position from a ``base_link`` one produces a number, not an error, and the
    metric built on it looks plausible. This is the check that turns that into a failure.

    An **unstated** frame is not a disagreement. A view carries ``None`` when the entity
    had nothing in range at all, when the data was logged without a frame, or when it
    holds a metric rather than geometry -- none of which can be shown to conflict with
    anything. Only two *different, stated* frames raise.

    A view with no rows is still consulted: an empty frame is recorded in a frame like any
    other, and ignoring that would make the caller's output frame flicker across a scene.

    Returns:
        The single stated frame, or ``None`` when no view states one.

    Raises:
        ValueError: When two views state different frames.
    """
    stated: dict[str, EntityView] = {}
    for view in views:
        if view.frame_id is not None:
            stated.setdefault(view.frame_id, view)

    if len(stated) > 1:
        detail = ", ".join(
            f"{view.entity_path} in {frame!r}" for frame, view in sorted(stated.items())
        )
        raise ValueError(
            f"Cannot compare geometry across coordinate frames: {detail}. "
            f"Bring the inputs into one frame first.",
        )
    return next(iter(stated), None)


def require(view: EntityView, *descriptors: ComponentDescriptor) -> None:
    """Raise when ``view`` is missing any of ``descriptors``."""
    missing = [descriptor.component for descriptor in descriptors if not view.has(descriptor)]
    if missing:
        raise ValueError(
            f"{view.entity_path} is missing required component(s): {', '.join(missing)}",
        )


@define(slots=True)
class EntitySystem:
    """Base class holding the source and target wiring shared by every system."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = ()
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = ()

    _sources: tuple[EntityPath, ...] = field(
        converter=lambda paths: tuple(as_entity_path(path) for path in paths),
    )
    _target: EntityPath = field(converter=as_entity_path)

    @property
    def sources(self) -> tuple[EntityPath, ...]:
        return self._sources

    @property
    def target(self) -> EntityPath:
        return self._target

    @property
    def targets(self) -> tuple[EntityPath, ...]:
        return (self._target,)

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        raise NotImplementedError


class Pipeline:
    """An ordered list of systems whose wiring is checked once, up front.

    The check is about *order*: if a system reads an entity that a later system writes,
    that is a bug in the pipeline and is reported at construction time rather than as an
    empty result at run time. Components that are expected to come from the store instead
    of from another system are checked when the pipeline runs, by :func:`require`.
    """

    def __init__(self, systems: Sequence[System]) -> None:
        self._systems = tuple(systems)
        self._validate()

    def _validate(self) -> None:
        produced: dict[EntityPath, set[ComponentDescriptor]] = {}

        for position, system in enumerate(self._systems):
            later_targets = {
                target for other in self._systems[position + 1 :] for target in other.targets
            }

            for source in system.sources:
                if source in produced:
                    missing = set(system.REQUIRES) - produced[source]
                    if missing:
                        names = ", ".join(sorted(m.component for m in missing))
                        raise ValueError(
                            f"{type(system).__name__} reads {source} for component(s) "
                            f"{names}, which no earlier system provides there",
                        )
                elif source in later_targets:
                    raise ValueError(
                        f"{type(system).__name__} reads {source} before a later system "
                        f"writes it; reorder the pipeline",
                    )

            for target in system.targets:
                produced.setdefault(target, set()).update(system.PROVIDES)

    def __len__(self) -> int:
        return len(self._systems)

    def __iter__(self) -> Iterator[System]:
        return iter(self._systems)

    @property
    def systems(self) -> tuple[System, ...]:
        return self._systems

    def run(self, ctx: SystemContext, at: int | TimeRange) -> tuple[Chunk, ...]:
        """Run every system in order, sending each result into the store.

        Returns:
            Every chunk produced, in production order.
        """
        produced: list[Chunk] = []
        for system in self._systems:
            for chunk in system(ctx, at):
                ctx.store.send_chunk(chunk)
                produced.append(chunk)
        return tuple(produced)


def resolve_times(
    ctx: SystemContext, entity_path: EntityPathLike, at: int | TimeRange
) -> list[int]:
    """Return the times of ``entity_path`` that ``at`` selects, in ascending order."""
    from t4perceval.core.timeline import TimeRange as _TimeRange

    times = ctx.store.times(entity_path, ctx.timeline)
    if isinstance(at, _TimeRange):
        return [int(time) for time in times[at.contains(times)]]
    return [int(at)] if int(at) in set(times.tolist()) else []
