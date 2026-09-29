"""Matching systems.

Matching pairs an estimation stream against a ground-truth stream and records the verdict
as a :class:`~t4perceval.archetype.MatchResults` chunk. Because the verdict is data --
row indices, a score, a TP/FP/FN status -- it can be stored, re-read and re-analysed,
which is what ``DynamicObjectWithPerceptionResult`` could not do: it held live object
references.

Every matcher is only its score matrix: :class:`MatchingSystem` implements the frame loop,
the feasibility rules and the assignment once. The modes differ in *what* they measure and
in whether a higher score is better.

Each mode writes to its own ``/matching/<mode>`` entity, so several can run over the same
frame and be compared afterwards.
"""

from __future__ import annotations

from t4perceval.system.matching.base import MatchingSystem
from t4perceval.system.matching.distance import (
    CenterDistanceBEVMatchingSystem,
    CenterDistanceMatchingSystem,
    PlaneDistanceMatchingSystem,
)
from t4perceval.system.matching.iou import (
    IoU3DMatchingSystem,
    IoUBEVMatchingSystem,
    IoURoiMatchingSystem,
)
from t4perceval.system.matching.join import MatchJoin
from t4perceval.system.matching.threshold import Thresholds, ThresholdsLike

__all__ = (
    "CenterDistanceBEVMatchingSystem",
    "CenterDistanceMatchingSystem",
    "IoU3DMatchingSystem",
    "IoUBEVMatchingSystem",
    "IoURoiMatchingSystem",
    "MatchJoin",
    "MatchingSystem",
    "PlaneDistanceMatchingSystem",
    "Thresholds",
    "ThresholdsLike",
)
