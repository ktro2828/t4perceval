from __future__ import annotations

import numpy as np
import pytest

from t4perceval import FRAME, TIMESTAMP, TimeColumn, TimeKind, TimePoint, TimeRange, Timeline


class TestTimeline:
    def test_builtin_timelines_declare_their_meaning(self) -> None:
        assert (FRAME.name, FRAME.kind) == ("frame", TimeKind.SEQUENCE)
        assert (TIMESTAMP.name, TIMESTAMP.kind) == ("timestamp_ns", TimeKind.TIMESTAMP)

    def test_compares_by_name_and_kind(self) -> None:
        assert Timeline("frame") == FRAME
        assert Timeline("frame", TimeKind.TIMESTAMP) != FRAME

    def test_rejects_an_empty_name(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            Timeline("")


class TestTimePoint:
    def test_carries_several_timelines_at_once(self) -> None:
        point = TimePoint.at(frame=3, timestamp_ns=1_000)

        assert point[FRAME] == 3
        assert point[TIMESTAMP] == 1_000
        assert set(point.timelines) == {FRAME, TIMESTAMP}
        assert len(point) == 2

    def test_reports_a_missing_timeline(self) -> None:
        point = TimePoint.at(frame=3)

        assert TIMESTAMP not in point
        assert point.get(TIMESTAMP) is None
        assert point.get(TIMESTAMP, -1) == -1
        with pytest.raises(KeyError, match="no time on timeline"):
            point[TIMESTAMP]

    def test_rejects_duplicates_and_emptiness(self) -> None:
        with pytest.raises(ValueError, match="duplicate timelines"):
            TimePoint([(FRAME, 1), (FRAME, 2)])
        with pytest.raises(ValueError, match="at least one of frame or timestamp_ns"):
            TimePoint.at()

    def test_is_order_independent(self) -> None:
        assert TimePoint([(FRAME, 1), (TIMESTAMP, 2)]) == TimePoint([(TIMESTAMP, 2), (FRAME, 1)])


class TestTimeRange:
    def test_is_closed_by_default(self) -> None:
        assert TimeRange(0, 10).contains([-1, 0, 5, 10, 11]).tolist() == [
            False,
            True,
            True,
            True,
            False,
        ]

    def test_endpoints_can_be_excluded(self) -> None:
        half_open = TimeRange(0, 10, include_end=False)

        assert half_open.contains([0, 9, 10]).tolist() == [True, True, False]

    def test_single_contains_only_one_time(self) -> None:
        assert TimeRange.single(4).contains([3, 4, 5]).tolist() == [False, True, False]

    def test_everything_contains_every_time(self) -> None:
        info = np.iinfo(np.int64)

        assert TimeRange.everything().contains([info.min, 0, info.max]).all()

    def test_rejects_an_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="must not precede start"):
            TimeRange(10, 0)


class TestTimeColumn:
    def test_holds_one_time_per_partition(self) -> None:
        column = TimeColumn.of(FRAME, [0, 1, 2])

        assert len(column) == 3
        assert not column.times.flags.writeable
        assert column.times.dtype == np.int64

    def test_compares_by_value(self) -> None:
        assert TimeColumn.of(FRAME, [1, 2]) == TimeColumn.of(FRAME, [1, 2])
        assert TimeColumn.of(FRAME, [1, 2]) != TimeColumn.of(FRAME, [1, 3])
        assert TimeColumn.of(FRAME, [1]) != TimeColumn.of(TIMESTAMP, [1])

    def test_selects_partitions_in_the_given_order(self) -> None:
        column = TimeColumn.of(FRAME, [10, 20, 30])

        assert column.select_partitions(np.array([2, 0])).times.tolist() == [30, 10]

    def test_rejects_a_multidimensional_column(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(P,\)"):
            TimeColumn(FRAME, np.zeros((2, 2), dtype=np.int64))

    def test_defaults_to_empty(self) -> None:
        assert len(TimeColumn(FRAME)) == 0
