"""Joining match verdicts back to the objects they refer to.

A :class:`~t4perceval.archetype.MatchResults` stores row *indices*, not object data,
so a metric has to follow them back to read what it needs -- confidence for AP, headings
for APH, instance ids for CLEAR, waypoints for ADE. Copying all of that into the match
result instead would duplicate most of the object columns, so the indices are resolved
here, once, for every metric to reuse.

The resolution is not a plain gather: ``est_index`` is a row index *within its frame*, so
it has to be offset by where that frame starts in the range being evaluated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, define, field

from t4perceval.descriptors import CLASS_ID, EST_INDEX, GT_INDEX
from t4perceval.system.base import require, require_same_frame

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPathLike
    from t4perceval.core.store import Store
    from t4perceval.core.timeline import TimeRange, Timeline
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayBool, NDArrayF64, NDArrayI64

__all__ = ("MatchJoin",)


def _frame_starts(view: EntityView, timeline: Timeline) -> tuple[NDArrayI64, NDArrayI64]:
    """Return the distinct frame times of ``view`` and the row each frame starts at.

    ``Store.range()`` orders partitions by time, so all rows sharing a time are
    contiguous even when they came from several chunks -- the first such partition is
    where the frame starts.
    """
    chunk = view.chunk
    index = chunk.index(timeline)
    if index is None or chunk.num_partitions == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    times, first = np.unique(index.times, return_index=True)
    return times, chunk.offsets[:-1][first]


def _resolve(
    local: NDArrayI64,
    match_times: NDArrayI64,
    frame_times: NDArrayI64,
    frame_starts: NDArrayI64,
) -> NDArrayI64:
    """Turn per-frame row indices into row indices over the whole range."""
    rows = np.full(local.shape, -1, dtype=np.int64)
    present = local >= 0
    if not present.any() or frame_times.size == 0:
        return rows

    slots = np.searchsorted(frame_times, match_times[present])
    if np.any(slots >= frame_times.size) or np.any(frame_times[slots] != match_times[present]):
        missing = sorted(set(match_times[present].tolist()) - set(frame_times.tolist()))
        raise ValueError(
            f"Match rows reference frame(s) {missing} that the joined entity does not have",
        )

    rows[present] = frame_starts[slots] + local[present]
    return rows


@define(frozen=True, slots=True)
class MatchJoin:
    """Match rows lined up with the estimation and ground-truth rows they point at.

    :attr:`est_rows` and :attr:`gt_rows` index into :attr:`estimation` / :attr:`ground_truth`
    over the whole evaluated range, with ``-1`` where the match row has no counterpart.
    """

    matches: EntityView
    estimation: EntityView
    ground_truth: EntityView
    timeline: Timeline
    est_rows: NDArrayI64 = field(eq=cmp_using(eq=np.array_equal))
    gt_rows: NDArrayI64 = field(eq=cmp_using(eq=np.array_equal))

    @classmethod
    def of(
        cls,
        store: Store,
        matching: EntityPathLike,
        estimation: EntityPathLike,
        ground_truth: EntityPathLike,
        *,
        timeline: Timeline,
        time_range: TimeRange,
        check_frames: bool = True,
    ) -> Self:
        """Build the join for one time range.

        Args:
            check_frames: Whether estimation and ground truth must agree about their
                coordinate frame. Every geometric metric reaches its inputs through this
                join, so one check here covers all of them.
        """
        matches = store.range(matching, timeline=timeline, time_range=time_range)
        est_view = store.range(estimation, timeline=timeline, time_range=time_range)
        gt_view = store.range(ground_truth, timeline=timeline, time_range=time_range)

        if check_frames:
            # The match entity is left out on purpose: its frame is derived, and a store
            # written before frames were recorded leaves it unstated.
            require_same_frame(est_view, gt_view)

        if len(matches):
            require(matches, EST_INDEX, GT_INDEX)

        match_times = matches.times(timeline) if len(matches) else np.empty(0, dtype=np.int64)
        est_times, est_starts = _frame_starts(est_view, timeline)
        gt_times, gt_starts = _frame_starts(gt_view, timeline)

        est_local = (
            matches.component(EST_INDEX).values if len(matches) else np.empty(0, dtype=np.int64)
        )
        gt_local = (
            matches.component(GT_INDEX).values if len(matches) else np.empty(0, dtype=np.int64)
        )

        return cls(
            matches,
            est_view,
            gt_view,
            timeline,
            _resolve(est_local, match_times, est_times, est_starts),
            _resolve(gt_local, match_times, gt_times, gt_starts),
        )

    def __len__(self) -> int:
        return len(self.matches)

    @property
    def has_estimation(self) -> NDArrayBool:
        """Which match rows point at an estimation."""
        return self.est_rows >= 0

    @property
    def has_ground_truth(self) -> NDArrayBool:
        """Which match rows point at a ground-truth object."""
        return self.gt_rows >= 0

    def match_component(self, descriptor: ComponentDescriptor) -> NDArrayF64:
        """Return one column of the match rows themselves."""
        column = self.matches.component(descriptor)
        if column is None:
            raise KeyError(f"{self.matches.entity_path} has no component {descriptor.component!r}")
        return column.values

    def est_component(
        self,
        descriptor: ComponentDescriptor,
        *,
        fill: float = np.nan,
    ) -> NDArrayF64:
        """Gather one estimation column onto the match rows.

        Rows with no estimation -- false negatives -- get ``fill``.
        """
        return self._gather(self.estimation, descriptor, self.est_rows, fill)

    def gt_component(
        self,
        descriptor: ComponentDescriptor,
        *,
        fill: float = np.nan,
    ) -> NDArrayF64:
        """Gather one ground-truth column onto the match rows.

        Rows with no ground truth -- false positives -- get ``fill``.
        """
        return self._gather(self.ground_truth, descriptor, self.gt_rows, fill)

    @staticmethod
    def _gather(
        view: EntityView,
        descriptor: ComponentDescriptor,
        rows: NDArrayI64,
        fill: float,
    ) -> NDArrayF64:
        if not len(view):
            # An entity with no rows carries no columns either, so there is nothing to
            # gather and nothing to complain about -- every match row must be a miss.
            return np.full(rows.shape, fill, dtype=np.float64)

        column = view.component(descriptor)
        if column is None:
            raise KeyError(f"{view.entity_path} has no component {descriptor.component!r}")

        values = column.values
        # Gathering with -1 would silently wrap to the last row, so absent rows are
        # gathered from index 0 and then overwritten.
        safe = np.where(rows >= 0, rows, 0)
        gathered = np.asarray(values[safe], dtype=np.float64)
        gathered[rows < 0] = fill
        return gathered

    def is_label_correct(self) -> NDArrayBool:
        """Which match rows pair an estimation and a ground truth of the same class.

        This reproduces the original package's ``is_label_correct``, and it is checked
        rather than assumed because a class-agnostic matcher can pair across classes.
        """
        est_class = self.est_component(CLASS_ID)
        gt_class = self.gt_component(CLASS_ID)
        return self.has_estimation & self.has_ground_truth & (est_class == gt_class)
