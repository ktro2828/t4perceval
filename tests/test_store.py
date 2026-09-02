from __future__ import annotations

import numpy as np
import pytest
from conftest import make_detections

from t4perceval import FRAME, TIMESTAMP, Detections3D, Store, TimePoint, TimeRange
from t4perceval.component import BatchConfidence, BatchTimeOffset
from t4perceval.descriptors import CONFIDENCE, POSITION, TIME_OFFSET


class TestWriting:
    def test_reports_what_it_holds(self, scene_store: Store) -> None:
        assert [str(path) for path in scene_store.entity_paths()] == [
            "/ground_truth/objects",
            "/estimation/objects",
        ]
        assert {timeline.name for timeline in scene_store.timelines()} == {
            "frame",
            "timestamp_ns",
        }
        assert scene_store.times("/estimation/objects", FRAME).tolist() == [0, 1]
        assert scene_store.times("/estimation/objects", TIMESTAMP).tolist() == [1_000, 2_000]

    def test_keeps_each_logged_chunk(self, scene_store: Store) -> None:
        assert len(scene_store.chunks("/estimation/objects")) == 2

    def test_an_unknown_entity_holds_nothing(self, scene_store: Store) -> None:
        assert scene_store.chunks("/nope") == ()
        assert scene_store.times("/nope", FRAME).tolist() == []

    def test_an_unindexed_timeline_yields_no_times(self) -> None:
        store = Store()
        store.log("/x", make_detections([[0.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))

        assert store.times("/x", TIMESTAMP).tolist() == []


class TestLatestAt:
    def test_returns_the_frame_at_that_time(self, scene_store: Store) -> None:
        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=1)

        assert len(view) == 2
        assert view.times(FRAME).tolist() == [1, 1]
        assert view.component(POSITION).values[:, 0].tolist() == [1.1, 2.0]

    def test_carries_the_latest_value_forward(self, scene_store: Store) -> None:
        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=99)

        assert view.times(FRAME).tolist() == [1, 1], "the newest frame is still current"

    def test_yields_nothing_before_the_first_frame(self, scene_store: Store) -> None:
        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=-1)

        assert len(view) == 0
        assert view.descriptors == ()

    def test_works_on_any_timeline(self, scene_store: Store) -> None:
        view = scene_store.latest_at("/estimation/objects", timeline=TIMESTAMP, at=1_500)

        assert view.times(TIMESTAMP).tolist() == [1_000, 1_000]

    def test_the_most_recently_logged_chunk_wins_a_tie(self) -> None:
        store = Store()
        store.log("/x", make_detections([[1.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))
        store.log("/x", make_detections([[2.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))

        view = store.latest_at("/x", timeline=FRAME, at=0)

        assert view.component(POSITION).values[:, 0].tolist() == [2.0]

    def test_an_unknown_entity_yields_an_empty_view(self, scene_store: Store) -> None:
        view = scene_store.latest_at("/nope", timeline=FRAME, at=0)

        assert len(view) == 0
        assert str(view.entity_path) == "/nope"


class TestRange:
    def test_collects_every_frame_in_the_window(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        assert len(view) == 4
        assert view.times(FRAME).tolist() == [0, 0, 1, 1]
        assert view.partition_ids().tolist() == [0, 0, 1, 1]

    def test_respects_the_window_bounds(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.single(0),
        )

        assert view.times(FRAME).tolist() == [0, 0]

    def test_orders_partitions_by_time_not_log_order(self) -> None:
        store = Store()
        store.log("/x", make_detections([[2.0, 0.0, 0.0]]), at=TimePoint.at(frame=2))
        store.log("/x", make_detections([[0.0, 0.0, 0.0]]), at=TimePoint.at(frame=0))
        store.log("/x", make_detections([[1.0, 0.0, 0.0]]), at=TimePoint.at(frame=1))

        view = store.range("/x", timeline=FRAME, time_range=TimeRange.everything())

        assert view.times(FRAME).tolist() == [0, 1, 2]
        assert view.component(POSITION).values[:, 0].tolist() == [0.0, 1.0, 2.0]

    def test_an_empty_window_yields_an_empty_view(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange(10, 20),
        )

        assert len(view) == 0

    def test_materializes_a_whole_scene_as_one_archetype(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )

        detection = view.materialize(Detections3D)

        assert len(detection) == 4


class TestStaticData:
    def test_belongs_to_every_time(self, scene_store: Store) -> None:
        scene_store.log_static_components(
            "/estimation/objects",
            {TIME_OFFSET: BatchTimeOffset([[0, 100]])},
        )

        for at in (0, 1):
            view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=at)
            assert TIME_OFFSET in view.static

    def test_a_single_row_is_broadcast_over_the_view(self, scene_store: Store) -> None:
        scene_store.log_static_components(
            "/estimation/objects",
            {TIME_OFFSET: BatchTimeOffset([[0, 100]])},
        )

        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=1)

        assert view.component(TIME_OFFSET).values.tolist() == [[0, 100], [0, 100]]

    def test_takes_precedence_over_temporal_data(self, scene_store: Store) -> None:
        scene_store.log_static_components(
            "/estimation/objects",
            {CONFIDENCE: BatchConfidence([0.5])},
        )

        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=1)

        assert view.component(CONFIDENCE).values.tolist() == [0.5, 0.5]

    def test_a_static_archetype_can_be_logged_whole(self) -> None:
        store = Store()
        store.log_static("/reference/objects", make_detections([[0.0, 0.0, 0.0]]))

        assert POSITION in store.static("/reference/objects")
        assert store.chunks("/reference/objects") == (), "static data is not temporal"

    def test_later_static_writes_merge(self) -> None:
        store = Store()
        store.log_static_components("/x", {CONFIDENCE: BatchConfidence([0.1])})
        store.log_static_components("/x", {TIME_OFFSET: BatchTimeOffset([[0, 1]])})

        assert set(store.static("/x")) == {CONFIDENCE, TIME_OFFSET}

    def test_a_multi_row_static_column_cannot_be_broadcast(self, scene_store: Store) -> None:
        scene_store.log_static_components(
            "/estimation/objects",
            {CONFIDENCE: BatchConfidence([0.5, 0.6, 0.7])},
        )
        view = scene_store.latest_at("/estimation/objects", timeline=FRAME, at=1)

        with pytest.raises(ValueError, match="only a single row can be broadcast"):
            view.component(CONFIDENCE)


class TestColumnRestriction:
    def test_narrows_a_view_to_the_requested_columns(self, scene_store: Store) -> None:
        view = scene_store.latest_at(
            "/estimation/objects",
            timeline=FRAME,
            at=1,
            components=[POSITION],
        )

        assert [descriptor.component for descriptor in view.descriptors] == ["position"]
        assert view.component(CONFIDENCE) is None

    def test_narrows_static_columns_too(self, scene_store: Store) -> None:
        scene_store.log_static_components(
            "/estimation/objects",
            {TIME_OFFSET: BatchTimeOffset([[0, 100]])},
        )

        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
            components=[POSITION],
        )

        assert view.static == {}

    def test_restriction_survives_a_range_query(self, scene_store: Store) -> None:
        view = scene_store.range(
            "/estimation/objects",
            timeline=FRAME,
            time_range=TimeRange.everything(),
            components=[POSITION],
        )

        assert len(view) == 4
        assert np.asarray(view.component(POSITION).values).shape == (4, 3)
