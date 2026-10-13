"""FP validation: does the perception stack still fire where it once hallucinated?

A false-positive validation dataset annotates the places where a perception stack reported
an object that was not there. Those annotations carry one class, ``false_positive`` by
convention, and the question asked of a new stack is the inverse of detection: an annotated
box that **no** estimation matches is a pass, one that an estimation does match is a fail.

The ordinary matcher already answers that question once it runs ``class_agnostic``, so that
an estimation of any class may pair with the annotation. On this ground truth a ``TP``
match row is the failure and an ``FN`` row the pass; an estimation that matched nothing is
not a mistake here and is left out. :class:`FpValidationSystem` reads those rows and reports
the pass rate, and :func:`fp_validation_by_instance` breaks the same verdicts down per
annotated instance, as the original package's ``GroundTruthStatus`` did.

One difference from the original: there a false-positive annotation is scored at the
threshold of the **estimation's** class, since its own class says nothing about size. Here
thresholds follow the ground-truth class, so the task runs at one threshold -- the
``false_positive`` class's -- which is what the original's own example configures anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, NamedTuple

import numpy as np
from attrs import define, field

from t4perceval.component import ALL_CLASSES, MatchStatus
from t4perceval.core.timeline import TimeRange
from t4perceval.descriptors import (
    CLASS_ID,
    EST_INDEX,
    GT_INDEX,
    INSTANCE_ID,
    MATCH_STATUS,
    THRESHOLD,
)
from t4perceval.system.base import require
from t4perceval.system.matching.join import MatchJoin
from t4perceval.system.metric.base import MetricRow, MetricSystem, in_class

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.core.timeline import Timeline
    from t4perceval.label import LabelRegistry
    from t4perceval.system.base import SystemContext
    from t4perceval.typing import NDArrayBool

__all__ = ("FpValidationSystem", "InstanceVerdicts", "fp_validation_by_instance")

#: The class name a false-positive annotation carries, as the original package spells it.
DEFAULT_FP_LABEL = "false_positive"


def _fp_class_id(labels: LabelRegistry | None, fp_label: str) -> int | None:
    """Return the id of ``fp_label``, or ``None`` when no registry can carry it.

    A registry without the class means no ground truth can be annotated with it, so the
    metric is undefined rather than wrong: every row reports ``NaN``.
    """
    if labels is None or fp_label not in labels:
        return None
    return labels.class_id(fp_label)


def _verdicts(join: MatchJoin, fp_class: int | None) -> tuple[NDArrayBool, NDArrayBool]:
    """Return ``(is_pass, is_fail)`` over the match rows.

    Only rows whose ground truth carries the false-positive class take part: an ``FN``
    row is a pass (nothing claimed the annotation) and a ``TP`` row a fail (something did).
    Rows with no ground truth are estimations that matched nothing, which this task
    ignores, and rows of any other ground-truth class belong to the ordinary metrics.
    """
    if fp_class is None or not len(join.matches):
        empty = np.zeros(len(join), dtype=np.bool_)
        return empty, empty

    status = join.match_component(MATCH_STATUS)
    annotated = join.has_ground_truth & in_class(join.gt_component(CLASS_ID), fp_class)
    is_pass = annotated & (status == int(MatchStatus.FN))
    is_fail = annotated & (status == int(MatchStatus.TP))
    return is_pass, is_fail


@define(slots=True)
class FpValidationSystem(MetricSystem):
    """Fraction of false-positive annotations that no estimation matched.

    Run it over a ``class_agnostic`` matching, so that an estimation of any class can
    claim an annotation. Every annotated box at every frame is one verdict: a pass when
    its match row is ``FN``, a fail when it is ``TP``. The value is the passes divided by
    the verdicts, ``NaN`` when there is no annotated box, and the support is the
    ground-truth count as in every metric of the family.

    Only the :attr:`fp_label` class carries a value. Other classes report ``NaN`` with
    their own support: their boxes are scored by the detection metrics, not here. The last
    row, class ``-1``, pools every verdict. The threshold column carries the threshold the
    matching was run at.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (
        EST_INDEX,
        GT_INDEX,
        MATCH_STATUS,
        THRESHOLD,
    )
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    METRIC_NAME: ClassVar[str] = "fp_validation"

    #: Class name of the false-positive annotations, resolved through the registry.
    fp_label: str = field(default=DEFAULT_FP_LABEL, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if not self.fp_label:
            raise ValueError("fp_label must be a class name, got an empty string")

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        classes = self.reported_classes(ctx, join)
        rows: list[MetricRow] = []

        if not len(join.matches):
            rows.extend((int(c), float("nan"), float("nan"), 0) for c in classes)
            return {self.target: rows}

        fp_class = _fp_class_id(ctx.labels, self.fp_label)
        is_pass, is_fail = _verdicts(join, fp_class)
        thresholds = join.match_component(THRESHOLD)
        gt_class = join.gt_component(CLASS_ID)
        est_class = join.est_component(CLASS_ID)
        status = join.match_component(MATCH_STATUS)
        is_false_positive = (status == int(MatchStatus.FP)) & join.has_estimation

        gt_classes_all = (
            join.ground_truth.component(CLASS_ID).values
            if len(join.ground_truth)
            else np.empty(0, dtype=np.int32)
        )

        for class_id in classes:
            num_ground_truth = int(np.count_nonzero(in_class(gt_classes_all, class_id)))

            mentions = (join.has_ground_truth & in_class(gt_class, class_id)) | (
                is_false_positive & in_class(est_class, class_id)
            )
            class_thresholds = thresholds[mentions]
            # The all-class row is read by `MetricValues.aggregate`, which keys on NaN.
            threshold = (
                float(class_thresholds[0])
                if class_thresholds.size and class_id != ALL_CLASSES
                else float("nan")
            )

            if class_id == ALL_CLASSES or class_id == fp_class:
                selected = in_class(gt_class, class_id)
                passes = int(np.count_nonzero(is_pass & selected))
                fails = int(np.count_nonzero(is_fail & selected))
                total = passes + fails
                value = passes / total if total else float("nan")
            else:
                value = float("nan")

            rows.append((int(class_id), threshold, value, num_ground_truth))

        return {self.target: rows}


class InstanceVerdicts(NamedTuple):
    """How one annotated instance fared over the frames it was evaluated in."""

    #: Frames the instance was annotated in and a verdict was reached.
    frames: int
    #: Frames in which no estimation matched it.
    passes: int
    #: Frames in which an estimation matched it.
    fails: int

    @property
    def pass_rate(self) -> float:
        """Passes over frames, ``NaN`` when the instance was never evaluated."""
        return self.passes / self.frames if self.frames else float("nan")


def fp_validation_by_instance(
    store: Store,
    matching: EntityPathLike,
    estimation: EntityPathLike,
    ground_truth: EntityPathLike,
    *,
    timeline: Timeline,
    labels: LabelRegistry,
    time_range: TimeRange | None = None,
    fp_label: str = DEFAULT_FP_LABEL,
) -> dict[int, InstanceVerdicts]:
    """Break the verdicts of :class:`FpValidationSystem` down per annotated instance.

    The matching rows and the ground truth's instance ids are both in the store, so this
    is a query over what the pipeline already recorded rather than a second metric. Keys
    are instance ids; :meth:`~t4perceval.label.InstanceRegistry.uuid` names them.
    An instance appears only for the frames it was annotated in, so its ``frames`` need
    not equal the scene's length.

    Args:
        store: The store the pipeline ran on.
        matching: The ``class_agnostic`` matching entity the metric read.
        estimation: The estimation entity that matching paired.
        ground_truth: The ground-truth entity; must carry ``instance_id``.
        timeline: The timeline the pipeline ran on.
        labels: The registry that names ``fp_label``.
        time_range: The range to query; the whole timeline by default.
        fp_label: Class name of the false-positive annotations.

    Raises:
        ValueError: If ``fp_label`` is not in ``labels``: unlike the metric, a query for
            instances of a class the registry does not have is a mistake, not an
            undefined value.
    """
    if fp_label not in labels:
        raise ValueError(f"fp_label {fp_label!r} is not a class of the registry {labels.names}")

    join = MatchJoin.of(
        store,
        matching,
        estimation,
        ground_truth,
        timeline=timeline,
        time_range=time_range if time_range is not None else TimeRange.everything(),
    )
    if not len(join.matches):
        return {}
    require(join.matches, MATCH_STATUS)
    # A range with estimations but no annotation still has FP rows, which this task
    # ignores; an empty ground-truth view carries no columns to require.
    if not len(join.ground_truth):
        return {}
    require(join.ground_truth, CLASS_ID, INSTANCE_ID)

    is_pass, is_fail = _verdicts(join, labels.class_id(fp_label))
    evaluated = is_pass | is_fail
    if not evaluated.any():
        return {}

    instances = join.gt_component(INSTANCE_ID)[evaluated].astype(np.int64)
    passed = is_pass[evaluated]
    ids, inverse = np.unique(instances, return_inverse=True)
    frames = np.bincount(inverse, minlength=ids.size)
    passes = np.bincount(inverse, weights=passed, minlength=ids.size).astype(np.int64)
    return {
        int(instance): InstanceVerdicts(int(total), int(ok), int(total - ok))
        for instance, total, ok in zip(ids, frames, passes, strict=True)
    }
