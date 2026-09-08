"""Pairing ground-truth and estimation frames by timestamp."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    TIMESTAMP,
    LabelRegistry,
    MatchResults,
    Recording,
    Store,
    TimePoint,
    TimeRange,
    Transform3D,
)
from t4perceval.align import (
    AlignOptions,
    align_frames,
    align_recordings,
    drop_frames,
    frame_times,
    pair_times,
    reindex_frames,
)
from t4perceval.descriptors import POSITION
from t4perceval.evaluation import build_evaluation_store, build_evaluation_store_from
from t4perceval.system import CenterDistanceMatchingSystem, Pipeline
from tests.conftest import make_detections

EVERYTHING = TimeRange.everything()
GT = "/ground_truth/objects"
EST = "/estimation/objects"


def ms(value: float) -> int:
    return int(value * 1_000_000)


def rec(
    labels: LabelRegistry,
    frames: list[tuple[int, int]],
    *,
    path: str = GT,
    x: float | None = None,
) -> Recording:
    """One car per frame at ``x`` (default: the frame index), logged on both timelines."""
    store = Store()
    for frame, timestamp_ns in frames:
        store.log(
            path,
            make_detections([[float(frame if x is None else x), 0.0, 0.0]]),
            at=TimePoint.at(frame=frame, timestamp_ns=timestamp_ns),
            frame_id="base_link",
        )
    return Recording.of(store, labels=labels)


GT_FRAMES = [(0, 0), (1, ms(1000))]
EST_FRAMES = [(0, ms(20)), (1, ms(500)), (2, ms(1030)), (3, ms(1500))]


class TestPairTimes:
    def test_each_reference_takes_its_nearest_query_within_tolerance(self) -> None:
        ref, query = pair_times([0, ms(1000)], [ms(20), ms(500), ms(1030), ms(1500)])

        assert ref.tolist() == [0, 1]
        assert query.tolist() == [0, 2]

    def test_the_tolerance_is_inclusive(self) -> None:
        assert pair_times([0], [ms(75)])[0].tolist() == [0]
        assert pair_times([0], [ms(75) + 1])[0].tolist() == []

    def test_conflicts_go_to_the_closer_reference(self) -> None:
        # Both references are nearest to 20; 30 is closer, so 0 falls back to -25.
        ref, query = pair_times([0, 30], [-25, 20], tolerance_ns=30)
        assert (ref.tolist(), query.tolist()) == ([0, 1], [0, 1])

        ref, query = pair_times([0, 30], [-25, 20], tolerance_ns=22)
        assert (ref.tolist(), query.tolist()) == ([1], [1])

    def test_ties_go_to_the_earlier_time(self) -> None:
        assert pair_times([0], [-5, 5], tolerance_ns=10)[1].tolist() == [0]
        assert pair_times([0, 10], [5], tolerance_ns=10)[0].tolist() == [0]

    def test_inputs_need_not_be_sorted(self) -> None:
        reference = np.array([ms(1000), 0])
        query = np.array([ms(1500), ms(1030), ms(20), ms(500)])

        ref, qry = pair_times(reference, query)

        assert reference[ref].tolist() == [0, ms(1000)]
        assert query[qry].tolist() == [ms(20), ms(1030)]

    def test_an_offset_corrects_a_clock_skew(self) -> None:
        assert pair_times([0, ms(1000)], [ms(200), ms(1200)])[0].size == 0
        assert pair_times([0, ms(1000)], [ms(200), ms(1200)], offset_ns=-ms(200))[0].tolist() == [
            0,
            1,
        ]

    def test_empty_inputs(self) -> None:
        for reference, query in (([], [1]), ([1], []), ([], [])):
            ref, qry = pair_times(reference, query)
            assert ref.dtype == np.int64 and qry.dtype == np.int64
            assert ref.size == qry.size == 0

    def test_a_negative_tolerance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pair_times([0], [0], tolerance_ns=-1)

    def test_one_reference_takes_one_query(self) -> None:
        ref, query = pair_times([0], [10, 20], tolerance_ns=30)

        assert (ref.tolist(), query.tolist()) == ([0], [0])

    def test_nearest_wins_even_when_a_larger_matching_exists(self) -> None:
        # Documented: (10, 6) is the nearest pair and takes 6, leaving 0 without 30.
        ref, query = pair_times([0, 10], [6, 30], tolerance_ns=25)

        assert (ref.tolist(), query.tolist()) == ([1], [0])


class TestFrameTimes:
    def test_returns_distinct_frames_and_their_stamps(self, labels: LabelRegistry) -> None:
        frames, times = frame_times(rec(labels, EST_FRAMES, path=EST), EST)

        assert frames.tolist() == [0, 1, 2, 3]
        assert times.tolist() == [ms(20), ms(500), ms(1030), ms(1500)]

    def test_two_chunks_at_one_frame_collapse(self, labels: LabelRegistry) -> None:
        frames, times = frame_times(rec(labels, [(0, 5), (0, 5)]), GT)

        assert frames.tolist() == [0]
        assert times.tolist() == [5]

    def test_one_frame_at_two_stamps_is_an_error(self, labels: LabelRegistry) -> None:
        with pytest.raises(ValueError, match="two different timestamps"):
            frame_times(rec(labels, [(0, 5), (0, 6)]), GT)

    def test_a_chunk_on_one_timeline_only_is_an_error(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(GT, make_detections([[0.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))

        with pytest.raises(ValueError, match="missing 'timestamp_ns'"):
            frame_times(Recording.of(store, labels=labels), GT)

    def test_an_entity_without_frames_is_an_error(self, labels: LabelRegistry) -> None:
        with pytest.raises(ValueError, match="no chunks on the FRAME timeline"):
            frame_times(rec(labels, GT_FRAMES), "/nowhere")


class TestAlignFrames:
    def test_describes_the_pairing(self, labels: LabelRegistry) -> None:
        alignment = align_frames(rec(labels, GT_FRAMES), rec(labels, EST_FRAMES, path=EST))

        assert alignment.reference_frames.tolist() == [0, 1]
        assert alignment.query_frames.tolist() == [0, 2]
        assert alignment.deltas_ns.tolist() == [ms(20), ms(30)]
        assert alignment.unmatched_reference_frames.tolist() == []
        assert alignment.unmatched_query_frames.tolist() == [1, 3]
        assert alignment.mapping() == {0: 0, 2: 1}
        assert alignment.describe()["align.pairs"] == "2"
        assert alignment.describe()["align.reference_path"] == GT
        assert "2 pairs" in str(alignment)
        assert f"max |delta|={ms(30)}ns" in str(alignment)


class TestReindexFrames:
    @pytest.fixture
    def query(self, labels: LabelRegistry) -> Recording:
        store = Store()
        for frame, timestamp_ns in EST_FRAMES:
            store.log(
                EST,
                make_detections([[float(frame), 0.0, 0.0]]),
                at=TimePoint.at(frame=frame, timestamp_ns=timestamp_ns),
                frame_id="base_link",
            )
            store.log(
                "/estimation/CAM_FRONT/objects",
                make_detections([[float(frame), 0.0, 0.0]]),
                # A camera's own capture time, sharing the FRAME.
                at=TimePoint.at(frame=frame, timestamp_ns=timestamp_ns + ms(7)),
                frame_id="CAM_FRONT",
            )
        store.log(
            "/tf/base_link",
            Transform3D(
                translation=[1.0, 0.0, 0.0], rotation=[0, 0, 0, 1], child_frame_id="base_link"
            ),
            at=TimePoint.at(timestamp_ns=ms(10)),
            frame_id="map",
        )
        store.log_static(
            "/tf/lidar",
            Transform3D(translation=[0.0, 0.0, 2.0], rotation=[0, 0, 0, 1], child_frame_id="lidar"),
            frame_id="base_link",
        )
        return Recording.of(store, labels=labels)

    @pytest.fixture
    def aligned(self, labels: LabelRegistry, query: Recording) -> Recording:
        return reindex_frames(query, align_frames(rec(labels, GT_FRAMES), query))

    def test_frames_are_rewritten_and_stamps_kept(self, aligned: Recording) -> None:
        assert aligned.times(EST, FRAME).tolist() == [0, 1]
        assert aligned.times(EST, TIMESTAMP).tolist() == [ms(20), ms(1030)]

        # Frame 1 now holds what was estimation frame 2.
        view = aligned.range(EST, timeline=FRAME, time_range=TimeRange.single(1))
        assert view.component(POSITION).values[:, 0].tolist() == [2.0]

    def test_other_frame_bearing_entities_follow(self, aligned: Recording) -> None:
        assert aligned.times("/estimation/CAM_FRONT/objects", FRAME).tolist() == [0, 1]
        assert aligned.times("/estimation/CAM_FRONT/objects", TIMESTAMP).tolist() == [
            ms(27),
            ms(1037),
        ]

    def test_entity_paths_restricts_the_rewrite(
        self, labels: LabelRegistry, query: Recording
    ) -> None:
        alignment = align_frames(rec(labels, GT_FRAMES), query)
        aligned = reindex_frames(query, alignment, entity_paths=[EST])

        assert aligned.times(EST, FRAME).tolist() == [0, 1]
        assert aligned.times("/estimation/CAM_FRONT/objects", FRAME).tolist() == [0, 1, 2, 3]

    def test_chunks_without_frames_pass_through_untouched(
        self, query: Recording, aligned: Recording
    ) -> None:
        assert aligned.chunks("/tf/base_link") == query.chunks("/tf/base_link")
        assert aligned.chunks("/tf/base_link")[0] is query.chunks("/tf/base_link")[0]
        assert aligned.static_chunks("/tf/lidar")[0] is query.static_chunks("/tf/lidar")[0]

    def test_an_entity_with_only_unmatched_frames_disappears(self, labels: LabelRegistry) -> None:
        query = rec(labels, [(7, ms(5000))], path=EST)
        aligned = reindex_frames(query, align_frames(rec(labels, GT_FRAMES), query))

        assert aligned.chunks(EST) == ()

    def test_registries_and_tags(self, query: Recording, aligned: Recording) -> None:
        assert aligned.labels is query.labels
        assert aligned.instances is query.instances
        tags = dict(aligned.metadata.tags)
        assert tags["align.tolerance_ns"] == "75000000"
        assert tags["align.pairs"] == "2"
        assert tags["align.unmatched_query"] == "2"

    def test_log_order_is_preserved(self, labels: LabelRegistry) -> None:
        store = Store()
        for x in (1.0, 2.0):
            store.log(
                EST,
                make_detections([[x, 0.0, 0.0]]),
                at=TimePoint.at(frame=5, timestamp_ns=ms(20)),
                frame_id="base_link",
            )
        query = Recording.of(store, labels=labels)
        aligned = reindex_frames(query, align_frames(rec(labels, GT_FRAMES), query))

        latest = aligned.latest_at(EST, timeline=FRAME, at=0)
        assert latest.component(POSITION).values[:, 0].tolist() == [2.0]


class TestDropFrames:
    def test_removes_the_frames_from_every_entity(self, labels: LabelRegistry) -> None:
        store = Store()
        for frame in range(3):
            at = TimePoint.at(frame=frame, timestamp_ns=ms(1000 * frame))
            store.log(GT, make_detections([[0.0, 0.0, 0.0]]), at=at, frame_id="base_link")
            store.log("/ground_truth/CAM_FRONT/objects", make_detections([[0.0, 0.0, 0.0]]), at=at)
        store.log_static(
            "/tf/lidar",
            Transform3D(translation=[0.0, 0.0, 2.0], rotation=[0, 0, 0, 1], child_frame_id="lidar"),
            frame_id="base_link",
        )
        recording = Recording.of(store, labels=labels)

        dropped = drop_frames(recording, [2])

        assert dropped.times(GT, FRAME).tolist() == [0, 1]
        assert dropped.times("/ground_truth/CAM_FRONT/objects", FRAME).tolist() == [0, 1]
        assert dropped.static_chunks("/tf/lidar") == recording.static_chunks("/tf/lidar")
        assert dropped.metadata.tags == ()


class TestAlignRecordings:
    def test_keep_leaves_unanswered_reference_frames(self, labels: LabelRegistry) -> None:
        reference = rec(labels, [*GT_FRAMES, (2, ms(2000))])
        aligned = align_recordings(reference, rec(labels, EST_FRAMES, path=EST))

        assert aligned.reference.times(GT, FRAME).tolist() == [0, 1, 2]
        assert aligned.query.times(EST, FRAME).tolist() == [0, 1]
        assert dict(aligned.reference.metadata.tags)["align.unmatched_reference_policy"] == "keep"

    def test_drop_removes_them(self, labels: LabelRegistry) -> None:
        reference = rec(labels, [*GT_FRAMES, (2, ms(2000))])
        aligned = align_recordings(
            reference,
            rec(labels, EST_FRAMES, path=EST),
            options=AlignOptions(unmatched_reference="drop"),
        )

        assert aligned.reference.times(GT, FRAME).tolist() == [0, 1]
        assert dict(aligned.reference.metadata.tags)["align.unmatched_reference"] == "1"
        assert reference.times(GT, FRAME).tolist() == [0, 1, 2]  # the input is untouched

    def test_options_are_validated(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            AlignOptions(tolerance_ns=-1)
        with pytest.raises(ValueError, match="keep"):
            AlignOptions(unmatched_reference="ignore")  # type: ignore[arg-type]


class TestBuildEvaluationStoreAlign:
    def matches(self, setup) -> MatchResults:  # noqa: ANN001
        Pipeline([CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)]).run(
            setup.context(),
            EVERYTHING,
        )
        return setup.store.range(
            "/matching/center_distance",
            timeline=FRAME,
            time_range=EVERYTHING,
        ).materialize(MatchResults)

    def test_without_alignment_the_frame_axes_are_taken_as_agreeing(
        self, labels: LabelRegistry
    ) -> None:
        # Estimation frames 0..3 against ground-truth frames 0..1: frames 2 and 3 are all-FP,
        # and frame 1 pairs the wrong messages (500 ms against 1000 ms).
        setup = build_evaluation_store(
            rec(labels, GT_FRAMES, x=0.0), rec(labels, EST_FRAMES, path=EST, x=0.0)
        )

        assert setup.store.times(EST, FRAME).tolist() == [0, 1, 2, 3]
        assert self.matches(setup).num_fp == 2
        assert setup.metadata.tags == ()

    def test_alignment_puts_both_on_the_reference_frames(self, labels: LabelRegistry) -> None:
        setup = build_evaluation_store(
            rec(labels, GT_FRAMES, x=0.0),
            rec(labels, EST_FRAMES, path=EST, x=0.0),
            align=AlignOptions(),
        )

        assert setup.store.times(EST, FRAME).tolist() == [0, 1]
        matches = self.matches(setup)
        assert (matches.num_tp, matches.num_fp, matches.num_fn) == (2, 0, 0)
        assert setup.store.times("/matching/center_distance", FRAME).tolist() == [0, 1]
        tags = dict(setup.metadata.tags)
        assert tags["align.pairs"] == "2"
        assert tags["align.unmatched_reference_policy"] == "keep"

    def test_unanswered_reference_frames_score_as_misses_or_vanish(
        self, labels: LabelRegistry
    ) -> None:
        reference = rec(labels, [*GT_FRAMES, (2, ms(2000))], x=0.0)
        query = rec(labels, EST_FRAMES, path=EST, x=0.0)

        kept = build_evaluation_store(reference, query, align=AlignOptions())
        assert self.matches(kept).num_fn == 1

        dropped = build_evaluation_store(
            reference,
            query,
            align=AlignOptions(unmatched_reference="drop"),
        )
        assert dropped.store.times(GT, FRAME).tolist() == [0, 1]
        assert self.matches(dropped).num_fn == 0
        assert dict(dropped.metadata.tags)["align.unmatched_reference"] == "1"

    def test_tags_are_optional_for_the_general_builder(self, labels: LabelRegistry) -> None:
        from t4perceval.evaluation import SourceSpec

        setup = build_evaluation_store_from((SourceSpec.of(rec(labels, GT_FRAMES), GT),))

        assert setup.metadata.tags == ()
