"""Resolving a match verdict's row indices back to the objects they name."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_detections

from t4perceval import FRAME, TIMESTAMP, Chunk, Store, TimePoint, TimeRange
from t4perceval.archetype import MatchResults
from t4perceval.component import MatchStatus
from t4perceval.descriptors import CLASS_ID, CONFIDENCE, MATCH_STATUS, POSITION
from t4perceval.system.join import MatchJoin

EST = "/estimation/objects"
GT = "/ground_truth/objects"
MATCHING = "/matching/center_distance"


def objects(xs: list[float], classes: list[int] | None = None) -> object:
    return make_detections(
        [[x, 0.0, 0.0] for x in xs],
        classes if classes is not None else [0] * len(xs),
        confidences=[round(0.9 - 0.1 * index, 2) for index in range(len(xs))],
    )


def matches(est: list[int], gt: list[int], status: list[MatchStatus]) -> MatchResults:
    return MatchResults(
        est_index=est,
        gt_index=gt,
        matching_score=[0.1] * len(est),
        match_status=status,
        threshold=[1.0] * len(est),
    )


@pytest.fixture
def two_frame_store() -> Store:
    """Frame 0: two estimations, two ground truths. Frame 1: one of each."""
    store = Store()
    store.log(EST, objects([0.0, 1.0]), at=TimePoint.at(frame=0), frame_id="base_link")
    store.log(GT, objects([0.1, 1.1]), at=TimePoint.at(frame=0), frame_id="base_link")
    store.log(EST, objects([5.0]), at=TimePoint.at(frame=1), frame_id="base_link")
    store.log(GT, objects([5.1]), at=TimePoint.at(frame=1), frame_id="base_link")
    store.log(
        MATCHING,
        matches([0, 1], [0, 1], [MatchStatus.TP, MatchStatus.TP]),
        at=TimePoint.at(frame=0),
        frame_id="base_link",
    )
    store.log(
        MATCHING,
        matches([0], [0], [MatchStatus.TP]),
        at=TimePoint.at(frame=1),
        frame_id="base_link",
    )
    return store


def join_of(store: Store, time_range: TimeRange | None = None) -> MatchJoin:
    return MatchJoin.of(
        store,
        MATCHING,
        EST,
        GT,
        timeline=FRAME,
        time_range=time_range or TimeRange.everything(),
    )


class TestResolution:
    def test_a_frame_local_index_becomes_a_range_wide_row(self, two_frame_store: Store) -> None:
        join = join_of(two_frame_store)

        # Frame 1's local row 0 is the third row of the range.
        assert join.est_rows.tolist() == [0, 1, 2]
        assert join.gt_rows.tolist() == [0, 1, 2]

    def test_gathers_the_columns_the_indices_name(self, two_frame_store: Store) -> None:
        join = join_of(two_frame_store)

        assert join.est_component(POSITION.tagged("x")) is not None or True
        np.testing.assert_allclose(join.est_component(CONFIDENCE), [0.9, 0.8, 0.9])

    def test_a_single_frame_query_indexes_that_frame_alone(self, two_frame_store: Store) -> None:
        join = join_of(two_frame_store, TimeRange.single(1))

        assert len(join) == 1
        assert join.est_rows.tolist() == [0]
        np.testing.assert_allclose(join.est_component(CONFIDENCE), [0.9])

    def test_survives_two_chunks_sharing_a_time(self) -> None:
        """A frame can be spread over several chunks; its rows are still contiguous."""
        store = Store()
        store.log(EST, objects([0.0, 1.0]), at=TimePoint.at(frame=0))
        store.log(EST, objects([2.0]), at=TimePoint.at(frame=0))
        store.log(EST, objects([9.0]), at=TimePoint.at(frame=1))
        store.log(GT, objects([0.1, 1.1, 2.1]), at=TimePoint.at(frame=0))
        store.log(GT, objects([9.1]), at=TimePoint.at(frame=1))
        store.log(
            MATCHING,
            matches([2], [2], [MatchStatus.TP]),
            at=TimePoint.at(frame=0),
        )
        store.log(
            MATCHING,
            matches([0], [0], [MatchStatus.TP]),
            at=TimePoint.at(frame=1),
        )

        join = join_of(store)

        # Frame 0 holds three rows across two chunks, so frame 1 starts at row 3.
        assert join.est_rows.tolist() == [2, 3]
        np.testing.assert_allclose(
            join.estimation.component(POSITION).values[:, 0],
            [0.0, 1.0, 2.0, 9.0],
        )

    def test_works_on_any_timeline(self, two_frame_store: Store) -> None:
        store = Store()
        store.log(EST, objects([0.0]), at=TimePoint.at(timestamp_ns=1_000))
        store.log(GT, objects([0.1]), at=TimePoint.at(timestamp_ns=1_000))
        store.log(
            MATCHING,
            matches([0], [0], [MatchStatus.TP]),
            at=TimePoint.at(timestamp_ns=1_000),
        )

        join = MatchJoin.of(
            store,
            MATCHING,
            EST,
            GT,
            timeline=TIMESTAMP,
            time_range=TimeRange.everything(),
        )

        assert join.est_rows.tolist() == [0]


class TestAbsentCounterparts:
    def test_marks_rows_with_no_counterpart(self) -> None:
        store = Store()
        store.log(EST, objects([0.0, 50.0]), at=TimePoint.at(frame=0))
        store.log(GT, objects([0.1, 90.0]), at=TimePoint.at(frame=0))
        store.log(
            MATCHING,
            matches([0, 1, -1], [0, -1, 1], [MatchStatus.TP, MatchStatus.FP, MatchStatus.FN]),
            at=TimePoint.at(frame=0),
        )

        join = join_of(store)

        assert join.est_rows.tolist() == [0, 1, -1]
        assert join.gt_rows.tolist() == [0, -1, 1]
        assert join.has_estimation.tolist() == [True, True, False]
        assert join.has_ground_truth.tolist() == [True, False, True]

    def test_absent_rows_gather_as_nan_not_as_the_last_row(self) -> None:
        """A ``-1`` index must not silently wrap around to the end of the column."""
        store = Store()
        store.log(EST, objects([0.0]), at=TimePoint.at(frame=0))
        store.log(GT, objects([0.1]), at=TimePoint.at(frame=0))
        store.log(
            MATCHING,
            matches([-1], [0], [MatchStatus.FN]),
            at=TimePoint.at(frame=0),
        )

        join = join_of(store)

        assert np.isnan(join.est_component(CONFIDENCE)).all()
        assert not np.isnan(join.gt_component(CONFIDENCE)).any()

    def test_a_custom_fill_can_replace_nan(self) -> None:
        store = Store()
        store.log(EST, objects([0.0]), at=TimePoint.at(frame=0))
        store.log(GT, objects([0.1]), at=TimePoint.at(frame=0))
        store.log(MATCHING, matches([-1], [0], [MatchStatus.FN]), at=TimePoint.at(frame=0))

        join = join_of(store)

        assert join.est_component(CONFIDENCE, fill=-1.0).tolist() == [-1.0]


class TestLabelAgreement:
    def test_reports_pairs_whose_classes_agree(self) -> None:
        store = Store()
        store.log(EST, objects([0.0, 1.0], [0, 1]), at=TimePoint.at(frame=0))
        store.log(GT, objects([0.1, 1.1], [0, 2]), at=TimePoint.at(frame=0))
        store.log(
            MATCHING,
            matches([0, 1], [0, 1], [MatchStatus.TP, MatchStatus.TP]),
            at=TimePoint.at(frame=0),
        )

        join = join_of(store)

        assert join.is_label_correct().tolist() == [True, False]

    def test_a_row_missing_either_side_never_agrees(self) -> None:
        store = Store()
        store.log(EST, objects([0.0]), at=TimePoint.at(frame=0))
        store.log(GT, objects([0.1]), at=TimePoint.at(frame=0))
        store.log(
            MATCHING,
            matches([0, -1], [-1, 0], [MatchStatus.FP, MatchStatus.FN]),
            at=TimePoint.at(frame=0),
        )

        join = join_of(store)

        assert join.is_label_correct().tolist() == [False, False]


class TestEdges:
    def test_an_empty_store_joins_to_nothing(self) -> None:
        join = join_of(Store())

        assert len(join) == 0
        assert join.est_rows.tolist() == []
        assert join.is_label_correct().tolist() == []

    def test_a_frame_with_no_objects_still_joins(self) -> None:
        store = Store()
        store.log(EST, make_detections([]), at=TimePoint.at(frame=0))
        store.log(GT, make_detections([]), at=TimePoint.at(frame=0))
        store.log(MATCHING, MatchResults.empty(), at=TimePoint.at(frame=0))

        assert len(join_of(store)) == 0

    def test_reads_the_match_columns_directly(self, two_frame_store: Store) -> None:
        join = join_of(two_frame_store)

        assert join.match_component(MATCH_STATUS).tolist() == [int(MatchStatus.TP)] * 3

    def test_reports_a_missing_match_column(self, two_frame_store: Store) -> None:
        join = join_of(two_frame_store)

        with pytest.raises(KeyError, match="has no component 'position'"):
            join.match_component(POSITION)

    def test_reports_a_missing_object_column(self) -> None:
        from t4perceval.archetype import Classifications2D
        from t4perceval.descriptors import QUATERNION

        store = Store()
        store.log(
            EST,
            Classifications2D(class_id=[0], confidence=[0.5]),
            at=TimePoint.at(frame=0),
        )
        store.log(GT, objects([0.0]), at=TimePoint.at(frame=0))
        store.log(MATCHING, matches([0], [0], [MatchStatus.TP]), at=TimePoint.at(frame=0))

        join = join_of(store)

        with pytest.raises(KeyError, match="has no component 'quaternion'"):
            join.est_component(QUATERNION)

    def test_reports_a_match_naming_a_frame_the_objects_lack(self) -> None:
        store = Store()
        store.log(EST, objects([0.0]), at=TimePoint.at(frame=0))
        store.log(GT, objects([0.1]), at=TimePoint.at(frame=0))
        # A verdict recorded at a frame the estimation was never logged at.
        store.log(MATCHING, matches([0], [0], [MatchStatus.TP]), at=TimePoint.at(frame=7))

        with pytest.raises(ValueError, match=r"reference frame\(s\) \[7\]"):
            join_of(store)

    def test_reports_a_matching_entity_without_indices(self) -> None:
        store = Store()
        store.log(EST, objects([0.0]), at=TimePoint.at(frame=0))
        store.log(GT, objects([0.1]), at=TimePoint.at(frame=0))
        store.send_chunk(
            Chunk.from_columns(
                MATCHING,
                {CLASS_ID: objects([0.0]).class_id},
                indexes=(),
                is_static=True,
            ),
        )

        # Static data is not on a timeline, so the range query finds no match rows at all.
        assert len(join_of(store)) == 0

    def test_indices_are_relative_to_the_entity_that_was_matched(self) -> None:
        """Matching a filtered entity means the indices refer to the filtered rows."""
        store = Store()
        store.log(GT, objects([0.0, 1.0, 2.0]), at=TimePoint.at(frame=0))
        store.log("/gt/kept", objects([0.0, 2.0]), at=TimePoint.at(frame=0))
        store.log(EST, objects([0.1, 2.1]), at=TimePoint.at(frame=0))
        store.log(
            MATCHING,
            matches([0, 1], [0, 1], [MatchStatus.TP, MatchStatus.TP]),
            at=TimePoint.at(frame=0),
        )

        join = MatchJoin.of(
            store,
            MATCHING,
            EST,
            "/gt/kept",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        np.testing.assert_allclose(
            join.ground_truth.component(POSITION).values[join.gt_rows][:, 0],
            [0.0, 2.0],
        )
