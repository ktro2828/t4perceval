"""The shared shape and joined-input plumbing of a metric system.

Scalar metrics write a :class:`~t4perceval.archetype.MetricValues`: the same four columns,
whatever the metric measures, with the metric's *name* carried by the entity path.
Structured metrics may override result serialization while reusing the source wiring,
class discovery and reporting-time rules.

A subclass declares which components it needs from each of its three sources and
implements :meth:`compute`. The base builds the :class:`~t4perceval.system.join.MatchJoin`,
validates the sources and wraps the result as a chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define

from t4perceval.archetype.metric import MetricValues
from t4perceval.component import ALL_CLASSES, BACKGROUND_CLASS_ID
from t4perceval.core.chunk import Chunk
from t4perceval.core.timeline import TimePoint, TimeRange
from t4perceval.descriptors import CLASS_ID, EST_INDEX, GT_INDEX, MATCH_STATUS
from t4perceval.system.base import EntitySystem, SystemContext, require
from t4perceval.system.join import MatchJoin

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.timeline import Timeline
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayF64, NDArrayI32

__all__ = (
    "MetricRow",
    "MetricSystem",
    "latest_time",
    "nan_mean",
    "registered_classes",
    "registry_classes",
    "reporting_time",
)

#: One output row: ``(class_id, threshold, value, support)``.
MetricRow = tuple[int, float, float, int]


def nan_mean(values: Sequence[float] | NDArrayF64) -> float:
    """Return the mean of the values that are defined, or ``NaN`` when none are.

    Averaging metrics has to skip the undefined ones rather than poison the result, which
    is what the original package's ``_mean`` did.
    """
    array = np.asarray(values, dtype=np.float64)
    defined = array[~np.isnan(array)]
    return float(defined.mean()) if defined.size else float("nan")


def latest_time(views: Iterable[EntityView], timeline: Timeline) -> int | None:
    """Return the last time any of ``views`` observed on ``timeline``, or ``None``."""
    latest: int | None = None
    for view in views:
        if not len(view):
            continue
        times = view.times(timeline)
        if times.size:
            latest = int(times[-1]) if latest is None else max(latest, int(times[-1]))
    return latest


def reporting_time(latest: int | None, time_range: TimeRange) -> int:
    """Return the time a scene-level metric is logged at.

    A metric summarizes a whole range, so it is logged at the last frame that range
    actually covered -- ``latest``, as :func:`latest_time` finds it. ``latest_at`` then
    returns the score that is current, and asking for a single frame gives that frame's own
    score. When nothing was observed anywhere, the start of the window is used rather than
    the sentinel end of an unbounded range.
    """
    if latest is not None:
        return latest
    return int(max(time_range.start, 0))


def registry_classes(ctx: SystemContext) -> NDArrayI32 | None:
    """Return the registry's class ids in ascending order, or ``None`` without a registry."""
    if ctx.labels is not None and len(ctx.labels):
        return np.asarray(sorted(info.class_id for info in ctx.labels.classes), dtype=np.int32)
    return None


def registered_classes(ctx: SystemContext, *views: EntityView) -> NDArrayI32:
    """Return the classes to report on, in ascending order.

    The registry decides when there is one, so a class with no objects still gets a row
    saying so instead of silently vanishing from the report. Without a registry, the
    classes present in ``views`` are used, minus the sentinels.
    """
    known = registry_classes(ctx)
    if known is not None:
        return known

    seen: set[int] = set()
    for view in views:
        if not len(view):
            continue
        column = view.component(CLASS_ID)
        if column is not None:
            seen.update(int(value) for value in column.values)
    return np.asarray(sorted(seen - {ALL_CLASSES, BACKGROUND_CLASS_ID}), dtype=np.int32)


@define(slots=True)
class MetricSystem(EntitySystem):
    """Base for a system that turns match verdicts into metric results.

    By default subclasses return scalar :class:`MetricValues` rows from :meth:`compute`.
    Sources are ``(matching, estimation, ground_truth)``. The three carry different
    components, so each is validated against its own declaration rather than one shared
    ``REQUIRES``: ``REQUIRES`` is what the *matching* source must have, and
    :attr:`REQUIRES_ESTIMATION` / :attr:`REQUIRES_GROUND_TRUTH` the other two.
    :meth:`requires_for` hands the right one to
    :class:`~t4perceval.system.base.Pipeline`, and :meth:`__call__` checks the same three
    at run time.
    """

    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = MetricValues.required_descriptors()

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCH_STATUS,
    )

    #: Components the estimation entity must carry.
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = ()

    #: Components the ground-truth entity must carry.
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = ()

    #: Default last segment of the target path, ``/metrics/<METRIC_NAME>``.
    METRIC_NAME: ClassVar[str] = "metric"

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 3:
            raise ValueError(
                f"{type(self).__name__} needs exactly three sources "
                f"(matching, estimation, ground truth), got {len(self.sources)}",
            )

    def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]:
        """Matching, estimation and ground truth each have their own requirements."""
        return (self.REQUIRES, self.REQUIRES_ESTIMATION, self.REQUIRES_GROUND_TRUTH)[index]

    @classmethod
    def on(
        cls,
        matching: EntityPathLike,
        estimation: EntityPathLike,
        ground_truth: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        **params: Any,
    ) -> Self:
        """Build a metric over one matching result and the entities it matched.

        The target defaults to ``/metrics/<METRIC_NAME>``. ``params`` are the subclass's
        own fields.
        """
        return cls(
            (matching, estimation, ground_truth),
            target if target is not None else f"/metrics/{cls.METRIC_NAME}",
            **params,
        )

    # -- subclass contract ------------------------------------------------------------

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        """Return the rows to write, keyed by the entity each metric belongs to."""
        raise NotImplementedError

    # -- plumbing ----------------------------------------------------------------------

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        matching, estimation, ground_truth = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        join = MatchJoin.of(
            ctx.store,
            matching,
            estimation,
            ground_truth,
            timeline=ctx.timeline,
            time_range=time_range,
        )

        if len(join.matches):
            require(join.matches, *self.REQUIRES)
        if len(join.estimation):
            require(join.estimation, *self.REQUIRES_ESTIMATION)
        if len(join.ground_truth):
            require(join.ground_truth, *self.REQUIRES_GROUND_TRUTH)

        at_time = self._reporting_time(join, time_range)
        return tuple(
            MetricValues.from_rows(rows).to_chunk(path, at=TimePoint(((ctx.timeline, at_time),)))
            for path, rows in self.compute(join, ctx).items()
        )

    @staticmethod
    def _reporting_time(join: MatchJoin, time_range: TimeRange) -> int:
        """Return the time a scene-level metric is logged at; see :func:`reporting_time`."""
        views = (join.matches, join.estimation, join.ground_truth)
        return reporting_time(latest_time(views, join.timeline), time_range)

    def classes(self, ctx: SystemContext, join: MatchJoin) -> NDArrayI32:
        """Return the classes to report on; see :func:`registered_classes`."""
        return registered_classes(ctx, join.estimation, join.ground_truth)
