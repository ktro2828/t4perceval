"""The summary of a sensing evaluation: the share of boxes the sensor saw, per class.

The original package stopped at the per-frame verdicts of ``SensingFrameResult``. This
metric reads those verdicts, as :class:`~t4perceval.system.sensing.SensingDetectionSystem`
writes them, joins them to the boxes they judge and reports a detection rate the way the
other tasks report a score. See :mod:`t4perceval.system.sensing` for the task as a whole.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define

from t4perceval.archetype.metric import MetricValues
from t4perceval.component import ALL_CLASSES, SensingStatus
from t4perceval.core.chunk import Chunk
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimePoint, TimeRange
from t4perceval.descriptors import (
    CLASS_ID,
    SENSING_STATUS,
)
from t4perceval.system.base import EntitySystem, SystemContext, require
from t4perceval.system.metric.base import (
    MetricRow,
    in_class,
    latest_time,
    registered_classes,
    reporting_time,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike

__all__ = ("SensingDetectionRateSystem",)


@define(slots=True)
class SensingDetectionRateSystem(EntitySystem):
    """The share of ground-truth boxes the sensor saw, per class.

    Sources are ``(sensing, objects)``: a :class:`SensingDetectionSystem` result and the
    objects entity it was computed over, joined by row. One
    :class:`~t4perceval.archetype.MetricValues` row per registered class and one for
    :data:`~t4perceval.component.ALL_CLASSES`: ``value`` is
    ``detected / (detected + failed)``, ``support`` that denominator. Warnings are left
    out of both, as they are verdicts about the annotation rather than the sensor. The
    whole range is summarized in one chunk, logged at its last frame.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (SENSING_STATUS,)
    REQUIRES_OBJECTS: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = MetricValues.required_descriptors()
    METRIC_NAME: ClassVar[str] = "sensing/detection_rate"

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources (sensing, objects), "
                f"got {len(self.sources)}",
            )

    def requires_for(self, index: int) -> tuple[ComponentDescriptor, ...]:
        return self.REQUIRES if index == 0 else self.REQUIRES_OBJECTS

    @classmethod
    def between(
        cls,
        sensing: EntityPathLike,
        objects: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
    ) -> Self:
        """Summarize the verdicts at ``sensing`` by the classes of ``objects``.

        The target defaults to ``/metrics/sensing/detection_rate``.
        """
        return cls(
            (sensing, objects),
            target if target is not None else as_entity_path("/metrics") / cls.METRIC_NAME,
        )

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        sensing, objects = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)
        results = ctx.store.range(sensing, timeline=ctx.timeline, time_range=time_range)
        boxes = ctx.store.range(objects, timeline=ctx.timeline, time_range=time_range)
        if len(results) != len(boxes):
            raise ValueError(
                f"{sensing} holds {len(results)} verdict(s) but {objects} has {len(boxes)} "
                "object(s) in range; the two must be aligned row for row",
            )
        if len(results) and not np.array_equal(
            results.times(ctx.timeline), boxes.times(ctx.timeline)
        ):
            # The same row count can hide verdicts filed at other frames than the boxes,
            # which a join by row would pair with the wrong objects.
            raise ValueError(
                f"{sensing} and {objects} hold the same number of rows but at different "
                f"{ctx.timeline.name!r} times; the verdicts must be those of these objects",
            )
        if len(results):
            require(results, *self.REQUIRES)
            require(boxes, *self.REQUIRES_OBJECTS)
            status = results.component(SENSING_STATUS).values  # type: ignore[union-attr]
            class_ids = boxes.component(CLASS_ID).values  # type: ignore[union-attr]
        else:
            status = np.empty(0, dtype=np.int8)
            class_ids = np.empty(0, dtype=np.int32)

        detected = status == int(SensingStatus.DETECTED)
        judged = detected | (status == int(SensingStatus.FAILED))
        rows: list[MetricRow] = []
        for class_id in [*registered_classes(ctx, boxes).tolist(), ALL_CLASSES]:
            selected = in_class(class_ids, class_id) & judged
            support = int(np.count_nonzero(selected))
            value = (
                float(np.count_nonzero(detected & selected)) / support if support else float("nan")
            )
            rows.append((int(class_id), float("nan"), value, support))

        at_time = reporting_time(latest_time((results,), ctx.timeline), time_range)
        return (
            MetricValues.from_rows(rows).to_chunk(
                self.target, at=TimePoint(((ctx.timeline, at_time),))
            ),
        )
