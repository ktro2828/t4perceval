from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import (
    BatchMatchStatus,
    BatchMatchingScore,
    BatchRowIndex,
    BatchThreshold,
    MatchStatus,
)
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import EST_INDEX, GT_INDEX, MATCH_STATUS, MATCHING_SCORE, THRESHOLD

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ("MatchResults",)


@define(frozen=True, slots=True)
class MatchResults(Archetype):
    """The outcome of matching an estimation stream against a ground-truth stream.

    One row per verdict. :attr:`est_index` and :attr:`gt_index` are row indices into the
    matched chunks, with ``-1`` meaning "no counterpart" -- so a false positive is a row
    with ``gt_index == -1`` and a false negative one with ``est_index == -1``. This
    replaces ``DynamicObjectWithPerceptionResult``, which held object references and
    therefore could not be stored or re-analysed.

    :attr:`threshold` records the threshold the verdict was reached at. It is the one
    thing here that a later stage could not recover by following the indices back to the
    objects, because only the matcher knew it -- and with per-class thresholds it differs
    from row to row. Everything else a metric needs is joined; see
    :class:`~t4perceval.system.join.MatchJoin`.
    """

    est_index = component_field(EST_INDEX, BatchRowIndex)
    gt_index = component_field(GT_INDEX, BatchRowIndex)
    matching_score = component_field(MATCHING_SCORE, BatchMatchingScore)
    match_status = component_field(MATCH_STATUS, BatchMatchStatus)
    threshold = component_field(THRESHOLD, BatchThreshold)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        statuses = set(MatchStatus)
        unknown = set(np.unique(self.match_status.values).tolist()) - {int(s) for s in statuses}
        if unknown:
            raise ValueError(f"match_status contains unknown values: {sorted(unknown)}")

    @classmethod
    def empty(cls) -> Self:
        """Return a match result with no rows."""
        return cls(
            est_index=np.empty(0, dtype=np.int64),
            gt_index=np.empty(0, dtype=np.int64),
            matching_score=np.empty(0, dtype=np.float64),
            match_status=np.empty(0, dtype=np.int8),
            threshold=np.empty(0, dtype=np.float64),
        )

    def count(self, status: MatchStatus) -> int:
        """Return how many rows carry ``status``."""
        return int(np.count_nonzero(self.match_status.values == int(status)))

    @property
    def num_tp(self) -> int:
        return self.count(MatchStatus.TP)

    @property
    def num_fp(self) -> int:
        return self.count(MatchStatus.FP)

    @property
    def num_fn(self) -> int:
        return self.count(MatchStatus.FN)
