"""Segmentation metrics: an element-wise comparison with no matching stage.

Segmentation labels a pixel or a point, not an object, so estimation and ground truth are
already aligned row by row and there is nothing to pair. The systems here take two sources
instead of a matching result plus two entities, and depend on ``CLASS_ID`` alone -- so one
implementation serves :class:`~t4perceval.archetype.SemanticSegmentation2D` and
:class:`~t4perceval.archetype.SemanticSegmentation3D` alike. Everything is derived from one
count matrix over ``(ground-truth class, estimated class)``.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.archetype.metric import ConfusionMatrix, MetricValues
from t4perceval.component import ALL_CLASSES, BACKGROUND_CLASS_ID
from t4perceval.core.entity import as_entity_path
from t4perceval.core.timeline import TimePoint, TimeRange
from t4perceval.descriptors import CLASS_ID
from t4perceval.label import UNKNOWN_CLASS_ID
from t4perceval.system.base import EntitySystem, require, require_same_frame
from t4perceval.system.filter import resolve_class_ids
from t4perceval.system.metric.base import MetricRow, nan_mean, registry_classes, reporting_time

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.system.base import SystemContext
    from t4perceval.typing import NDArrayI32, NDArrayI64

__all__ = (
    "SegmentationConfusionMatrixSystem",
    "SegmentationIoUSystem",
    "SegmentationMetricSystem",
)

#: Accumulated cells, ``(ground_truth_class_id, estimation_class_id) -> count``.
_Cells = dict[tuple[int, int], int]


def _slots(classes: NDArrayI32, ids: NDArrayI32) -> NDArrayI64:
    """Return the index of each id in sorted ``classes``, or ``-1`` when it is absent."""
    if not classes.size:
        return np.full(ids.shape, -1, dtype=np.int64)
    slots = np.minimum(np.searchsorted(classes, ids), classes.size - 1).astype(np.int64)
    slots[classes[slots] != ids] = -1
    return slots


@define(slots=True)
class SegmentationMetricSystem(EntitySystem):
    """Base for a metric over two element-wise aligned label columns.

    Sources are ``(estimation, ground_truth)``. Row ``i`` of one is compared with row ``i``
    of the other at every time, so both must hold the same number of rows per frame and
    enumerate the same elements in the same order -- the pixels of one label image, or the
    points of one cloud. A frame where the counts differ raises rather than misaligning
    silently, and the check is made per frame: a whole-range comparison would shift every
    later row when one side lacked a frame.

    Ground-truth rows labelled :data:`~t4perceval.label.UNKNOWN_CLASS_ID` are excluded, as
    are rows whose class is in :attr:`ignore`. An ignored class is *not evaluated*: it also
    leaves the class axis, so a prediction of it counts as background -- a false negative
    for the true class. An estimation whose class is not on the axis is a prediction of no
    known class and lands in the background column. A ground-truth class the registry does
    not know is malformed data and raises.

    Subclasses implement :meth:`emit`, turning the accumulated count matrix into chunks.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)

    #: Default target path is ``/metrics/<METRIC_NAME>``; may contain ``/``.
    METRIC_NAME: ClassVar[str] = "segmentation"

    ignore: tuple[str | int, ...] = field(default=(), converter=tuple, kw_only=True)
    """Classes to leave out of the evaluation, as names or ids."""

    check_frames: bool = field(default=True, kw_only=True)
    """Refuse inputs that state different coordinate frames -- two cameras, say."""

    def __attrs_post_init__(self) -> None:
        if len(self.sources) != 2:
            raise ValueError(
                f"{type(self).__name__} needs exactly two sources "
                f"(estimation, ground truth), got {len(self.sources)}",
            )

    @classmethod
    def between(
        cls,
        estimation: EntityPathLike,
        ground_truth: EntityPathLike,
        *,
        target: EntityPathLike | None = None,
        **params: Any,
    ) -> Self:
        """Build a metric between an estimation and a ground-truth entity.

        The target defaults to ``/metrics/<METRIC_NAME>``. ``params`` are the subclass's
        own fields.
        """
        return cls(
            (estimation, ground_truth),
            target if target is not None else f"/metrics/{cls.METRIC_NAME}",
            **params,
        )

    # -- subclass contract ------------------------------------------------------------

    def emit(
        self,
        counts: NDArrayI64,
        axes: list[int],
        ctx: SystemContext,
        at: int,
    ) -> tuple[Chunk, ...]:
        """Turn the ``(K+1, K+1)`` count matrix over ``axes`` into result chunks.

        ``axes`` is ``[*classes, BACKGROUND_CLASS_ID]``; rows are ground truth, columns
        estimation. The background row is always zero.
        """
        raise NotImplementedError

    # -- plumbing ----------------------------------------------------------------------

    def __call__(self, ctx: SystemContext, at: int | TimeRange) -> Iterable[Chunk]:
        estimation, ground_truth = self.sources
        time_range = at if isinstance(at, TimeRange) else TimeRange.single(at)

        times = np.union1d(
            ctx.store.times(estimation, ctx.timeline),
            ctx.store.times(ground_truth, ctx.timeline),
        )
        selected = [int(time) for time in times[time_range.contains(times)]]

        ignored = np.asarray(sorted(self._ignored_ids(ctx)), dtype=np.int32)
        known = registry_classes(ctx)
        if known is not None:
            known = np.setdiff1d(known, ignored)

        cells: _Cells = {}
        latest: int | None = None
        for time in selected:
            single = TimeRange.single(time)
            est_view = ctx.store.range(estimation, timeline=ctx.timeline, time_range=single)
            gt_view = ctx.store.range(ground_truth, timeline=ctx.timeline, time_range=single)
            if len(est_view):
                require(est_view, *self.REQUIRES)
            if len(gt_view):
                require(gt_view, *self.REQUIRES)
            if self.check_frames:
                require_same_frame(est_view, gt_view)
            if len(est_view) != len(gt_view):
                raise ValueError(
                    f"{estimation} and {ground_truth} must label the same elements, but hold "
                    f"{len(est_view)} and {len(gt_view)} row(s) at {ctx.timeline.name}={time}",
                )
            if not len(gt_view):
                continue
            latest = time
            self._accumulate(
                cells,
                gt_view.component(CLASS_ID).values,  # type: ignore[union-attr]
                est_view.component(CLASS_ID).values,  # type: ignore[union-attr]
                known,
                ignored,
                ground_truth=ground_truth,
                where=f"{ctx.timeline.name}={time}",
            )

        classes = known if known is not None else self._observed(cells)
        axes = [*(int(class_id) for class_id in classes), BACKGROUND_CLASS_ID]
        counts = self._densify(cells, axes)
        return self.emit(counts, axes, ctx, reporting_time(latest, time_range))

    def _ignored_ids(self, ctx: SystemContext) -> set[int]:
        # UNKNOWN is always excluded; registries refuse to hold it, so no registered class
        # is ever hit by that default.
        return resolve_class_ids(self.ignore, ctx, field_name="ignore") | {UNKNOWN_CLASS_ID}

    @staticmethod
    def _accumulate(
        cells: _Cells,
        gt: NDArrayI32,
        est: NDArrayI32,
        known: NDArrayI32 | None,
        ignored: NDArrayI32,
        *,
        ground_truth: EntityPath,
        where: str,
    ) -> None:
        """Add one frame's ``(ground truth, estimation)`` pairs to ``cells``."""
        keep = ~np.isin(gt, ignored)
        gt, est = gt[keep], est[keep]
        if not gt.size:
            return

        if known is not None:
            classes = known
        else:
            sentinels = np.asarray([ALL_CLASSES, BACKGROUND_CLASS_ID], dtype=np.int32)
            classes = np.setdiff1d(
                np.unique(np.concatenate((gt, est))), np.union1d(sentinels, ignored)
            )

        gt_slots = _slots(classes, gt)
        unregistered = gt_slots < 0
        if unregistered.any():
            raise ValueError(
                f"{ground_truth} holds ground-truth class id(s) "
                f"{np.unique(gt[unregistered]).tolist()} absent from the label registry at "
                f"{where}; register them or list them in `ignore`",
            )
        width = classes.size + 1
        est_slots = _slots(classes, est)
        est_slots[est_slots < 0] = width - 1  # a prediction of no known class: background

        matrix = np.bincount(gt_slots * width + est_slots, minlength=width * width)
        matrix = matrix.reshape(width, width)
        axes = [*(int(class_id) for class_id in classes), BACKGROUND_CLASS_ID]
        for row, column in zip(*np.nonzero(matrix)):
            key = (axes[row], axes[column])
            cells[key] = cells.get(key, 0) + int(matrix[row, column])

    @staticmethod
    def _observed(cells: _Cells) -> NDArrayI32:
        seen = {gt for gt, _ in cells} | {est for _, est in cells}
        return np.asarray(sorted(seen - {ALL_CLASSES, BACKGROUND_CLASS_ID}), dtype=np.int32)

    @staticmethod
    def _densify(cells: _Cells, axes: list[int]) -> NDArrayI64:
        slot = {class_id: index for index, class_id in enumerate(axes)}
        counts = np.zeros((len(axes), len(axes)), dtype=np.int64)
        for (gt, est), count in cells.items():
            counts[slot[gt], slot[est]] += count
        return counts


@define(slots=True)
class SegmentationConfusionMatrixSystem(SegmentationMetricSystem):
    """Count ground-truth/estimation class pairs over every element.

    Rows are ground-truth classes and columns estimation classes, in
    :class:`~t4perceval.archetype.ConfusionMatrix` long form. Unlike the detection matrix,
    where the background column holds false negatives and the background row false
    positives, here the background **column** holds elements predicted as no known class
    and the background **row is always zero**: a ground-truth element without a known
    class is ignored or rejected, never counted. The row is kept so the matrix stays square
    and ``as_matrix`` works unchanged.
    """

    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = ConfusionMatrix.required_descriptors()
    METRIC_NAME: ClassVar[str] = "segmentation/confusion_matrix"

    def emit(
        self,
        counts: NDArrayI64,
        axes: list[int],
        ctx: SystemContext,
        at: int,
    ) -> tuple[Chunk, ...]:
        if len(axes) == 1:
            # Background only: no class is registered and none was seen.
            result = ConfusionMatrix.empty()
        else:
            result = ConfusionMatrix.from_rows(
                [
                    (ground_truth, estimation, int(counts[row, column]))
                    for (row, ground_truth), (column, estimation) in product(
                        enumerate(axes),
                        enumerate(axes),
                    )
                ],
            )
        return (result.to_chunk(self.target, at=TimePoint(((ctx.timeline, at),))),)


@define(slots=True)
class SegmentationIoUSystem(SegmentationMetricSystem):
    """Per-class IoU with mIoU, per-class accuracy with its mean, and pixel accuracy.

    All three come from one count matrix, so one system writes them to three entities
    under :attr:`target`:

    * ``<target>/iou`` -- ``TP / (TP + FP + FN)`` per class; the aggregate row is their
      mean, the mIoU.
    * ``<target>/accuracy`` -- ``TP / (TP + FN)`` per class, the share of a class's
      elements that were labelled correctly; the aggregate row is the mean class accuracy.
    * ``<target>/pixel_accuracy`` -- one row: correctly labelled elements over all
      evaluated elements.

    A class whose union is empty -- never in the ground truth and never predicted -- has an
    undefined IoU, ``NaN``, and is left out of the mean rather than scored 0 or 1; the
    registry keeps its row so the shape does not depend on the scene. ``support`` is the
    number of evaluated ground-truth elements of the class.
    """

    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = MetricValues.required_descriptors()
    METRIC_NAME: ClassVar[str] = "segmentation"

    @property
    def targets(self) -> tuple[EntityPath, ...]:
        root = as_entity_path(self.target)
        return (root / "iou", root / "accuracy", root / "pixel_accuracy")

    def emit(
        self,
        counts: NDArrayI64,
        axes: list[int],
        ctx: SystemContext,
        at: int,
    ) -> tuple[Chunk, ...]:
        iou_target, accuracy_target, pixel_target = self.targets
        classes = axes[:-1]
        num = len(classes)
        matrix = counts.astype(np.float64)

        true_positive = np.diag(matrix)[:num]
        ground_truth = matrix[:num, :].sum(axis=1)  # row sums: the support
        estimated = matrix[:, :num].sum(axis=0)  # column sums; the background row is zero
        union = ground_truth + estimated - true_positive

        iou = np.divide(true_positive, union, out=np.full(num, np.nan), where=union > 0)
        accuracy = np.divide(
            true_positive,
            ground_truth,
            out=np.full(num, np.nan),
            where=ground_truth > 0,
        )
        total = int(ground_truth.sum())
        nan = float("nan")

        rows: dict[EntityPath, list[MetricRow]] = {
            iou_target: [
                *(
                    (class_id, nan, float(iou[index]), int(ground_truth[index]))
                    for index, class_id in enumerate(classes)
                ),
                (ALL_CLASSES, nan, nan_mean(iou), total),
            ],
            accuracy_target: [
                *(
                    (class_id, nan, float(accuracy[index]), int(ground_truth[index]))
                    for index, class_id in enumerate(classes)
                ),
                (ALL_CLASSES, nan, nan_mean(accuracy), total),
            ],
            pixel_target: [
                (ALL_CLASSES, nan, float(true_positive.sum()) / total if total else nan, total),
            ],
        }
        return tuple(
            MetricValues.from_rows(table).to_chunk(path, at=TimePoint(((ctx.timeline, at),)))
            for path, table in rows.items()
        )
