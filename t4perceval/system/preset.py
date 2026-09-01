"""Ready-made system sequences for the common evaluations.

These are plain functions returning a list of systems. Nothing here selects an evaluation
task from a config value -- the returned list is ordinary data you can print, edit, or
extend before handing it to a :class:`~t4perceval.system.base.Pipeline`. That is the
difference between sugar and the ``EvaluationTask`` enum this package set out to remove.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from t4perceval.system.matching import CenterDistanceMatchingSystem
from t4perceval.system.metric.detection import (
    AveragePrecisionHeadingSystem,
    AveragePrecisionSystem,
    MeanAveragePrecisionSystem,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.core.entity import EntityPathLike
    from t4perceval.system.base import System
    from t4perceval.system.matching import MatchingSystem
    from t4perceval.system.threshold import ThresholdsLike

__all__ = ("average_precision_sweep",)


def average_precision_sweep(
    estimation: EntityPathLike,
    ground_truth: EntityPathLike,
    *,
    matcher: type[MatchingSystem] = CenterDistanceMatchingSystem,
    thresholds: Sequence[ThresholdsLike] = (1.0,),
    heading: bool = False,
    class_agnostic: bool = False,
) -> list[System]:
    """Return the systems for average precision over several matching thresholds.

    Each threshold gets its own matching run, because a threshold changes which pairs the
    assignment is allowed to make -- re-thresholding one loose run would not free a
    rejected pair's counterpart to be matched elsewhere, and would quietly disagree with
    the reference implementation.

    Targets are indexed by declaration order (``/matching/<mode>/0``, ``/metrics/ap/0``,
    ...) rather than by threshold value, since a per-class threshold has no single value to
    put in a path.

    Args:
        estimation: Entity holding the estimated objects.
        ground_truth: Entity holding the ground-truth objects.
        matcher: Matching system class to sweep.
        thresholds: One threshold per matching run.
        heading: Also emit APH and mAPH.
        class_agnostic: Passed through to each matcher.

    Returns:
        The systems in run order: a matcher and a metric per threshold, then the means.
    """
    if not thresholds:
        raise ValueError("thresholds must not be empty")

    mode = matcher.MATCHING_NAME
    systems: list[System] = []
    ap_targets: list[str] = []
    aph_targets: list[str] = []

    for index, threshold in enumerate(thresholds):
        matching_target = f"/matching/{mode}/{index}"
        systems.append(
            matcher.between(
                estimation,
                ground_truth,
                threshold=threshold,
                class_agnostic=class_agnostic,
                target=matching_target,
            ),
        )

        ap_target = f"/metrics/ap/{index}"
        ap_targets.append(ap_target)
        systems.append(
            AveragePrecisionSystem.on(
                matching_target,
                estimation,
                ground_truth,
                target=ap_target,
            ),
        )

        if heading:
            aph_target = f"/metrics/aph/{index}"
            aph_targets.append(aph_target)
            systems.append(
                AveragePrecisionHeadingSystem.on(
                    matching_target,
                    estimation,
                    ground_truth,
                    target=aph_target,
                ),
            )

    systems.append(MeanAveragePrecisionSystem.of(ap_targets, target="/metrics/map"))
    if heading:
        systems.append(MeanAveragePrecisionSystem.of(aph_targets, target="/metrics/maph"))

    return systems
