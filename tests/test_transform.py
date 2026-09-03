"""Transform edges: how they are addressed, recorded and read back."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import FRAME, TIMESTAMP, Store, TimePoint, TimeRange, Transform3D
from t4perceval.descriptors import ROTATION, TRANSLATION
from t4perceval.io import chunk_from_table, chunk_to_table
from t4perceval.transform import DEFAULT_ROOT, edges, frames_of, transform_path

EVERYTHING = TimeRange.everything()


def identity(x: float = 0.0, z: float = 0.0) -> Transform3D:
    return Transform3D(translation=[[x, 0.0, z]], rotation=[[0.0, 0.0, 0.0, 1.0]])


class TestPaths:
    def test_an_edge_round_trips_through_its_path(self) -> None:
        path = transform_path("map", "base_link")

        assert str(path) == "/transforms/map/base_link"
        assert frames_of(path) == ("map", "base_link")

    def test_a_custom_root_is_honoured(self) -> None:
        path = transform_path("map", "base_link", root="/tf")

        assert str(path) == "/tf/map/base_link"
        assert frames_of(path, root="/tf") == ("map", "base_link")

    def test_an_empty_frame_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            transform_path("", "base_link")

    def test_a_frame_name_containing_a_separator_is_rejected(self) -> None:
        # `EntityPath.__truediv__` parses its argument, so this would otherwise become
        # /transforms/velodyne/top/base_link -- an edge nothing could find again.
        with pytest.raises(ValueError, match="must not contain"):
            transform_path("velodyne/top", "base_link")

    def test_a_self_edge_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="two different frames"):
            transform_path("map", "map")

    def test_a_path_that_is_not_an_edge_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a transform edge"):
            frames_of("/transforms/map")


class TestDiscovery:
    def test_edges_come_from_the_entity_paths_alone(self) -> None:
        store = Store()
        for parent, child in (("map", "base_link"), ("base_link", "LIDAR_TOP")):
            store.log(
                transform_path(parent, child),
                identity(),
                at=TimePoint.at(frame=0),
                frame_id=parent,
            )

        assert set(edges(store)) == {("map", "base_link"), ("base_link", "LIDAR_TOP")}

    def test_unrelated_entities_are_ignored(self) -> None:
        store = Store()
        store.log(transform_path("map", "base_link"), identity(), at=TimePoint.at(frame=0))
        store.log("/ground_truth/objects", identity(), at=TimePoint.at(frame=0))
        # Something filed under the root that is not an edge must not break discovery.
        store.log(DEFAULT_ROOT / "notes", identity(), at=TimePoint.at(frame=0))

        assert set(edges(store)) == {("map", "base_link")}

    def test_an_empty_store_has_no_edges(self) -> None:
        assert edges(Store()) == {}


class TestRecording:
    def test_one_sample_answers_every_later_time(self) -> None:
        # Why a fixed extrinsic is logged as a temporal sample rather than as static data:
        # static columns carry no frame and read back as zero rows on an entity that has
        # no temporal partition, whereas this reaches forward for free.
        store = Store()
        path = transform_path("base_link", "LIDAR_TOP")
        store.log(path, identity(z=2.0), at=TimePoint.at(frame=0), frame_id="base_link")

        for at in (0, 5, 999):
            view = store.latest_at(path, timeline=FRAME, at=at)
            assert view.component(TRANSLATION).values.tolist() == [[0.0, 0.0, 2.0]]

    def test_static_logging_would_lose_the_values(self) -> None:
        # The negative case the decision above rests on, pinned so a future refactor
        # towards `log_static` fails loudly instead of silently returning nothing.
        store = Store()
        path = transform_path("base_link", "LIDAR_TOP")
        store.log_static(path, identity(z=2.0))

        assert len(store.latest_at(path, timeline=FRAME, at=0)) == 0
        assert store.static(path)[TRANSLATION].values.tolist() == [[0.0, 0.0, 2.0]]

    def test_an_edge_is_queryable_on_either_timeline(self) -> None:
        store = Store()
        path = transform_path("map", "base_link")
        store.log(
            path,
            identity(x=10.0),
            at=TimePoint.at(frame=1, timestamp_ns=1_000_000_000),
            frame_id="map",
        )

        for timeline, at in ((FRAME, 1), (TIMESTAMP, 1_000_000_000)):
            assert len(store.latest_at(path, timeline=timeline, at=at)) == 1

    def test_a_series_reads_back_in_order(self) -> None:
        store = Store()
        path = transform_path("map", "base_link")
        for frame, x in enumerate((0.0, 10.0, 20.0)):
            store.log(path, identity(x=x), at=TimePoint.at(frame=frame), frame_id="map")

        view = store.range(path, timeline=FRAME, time_range=EVERYTHING)

        assert view.component(TRANSLATION).values[:, 0].tolist() == [0.0, 10.0, 20.0]


class TestArchetype:
    def test_it_declares_both_columns(self) -> None:
        assert [d.component for d in Transform3D.required_descriptors()] == [
            "translation",
            "rotation",
        ]

    def test_its_descriptors_are_not_the_object_ones(self) -> None:
        # Distinct from POSITION/QUATERNION so a system asking for a 3D position cannot be
        # pointed at a transform entity and appear to work.
        from t4perceval.descriptors import POSITION, QUATERNION

        assert TRANSLATION != POSITION
        assert ROTATION != QUATERNION

    def test_a_chunk_round_trips_through_arrow(self) -> None:
        # Persistence needs no transform-specific IO: an edge is an ordinary chunk.
        chunk = identity(z=2.0).to_chunk(
            transform_path("base_link", "LIDAR_TOP"),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored == chunk

    def test_an_empty_batch_is_allowed(self) -> None:
        empty = Transform3D(
            translation=np.empty((0, 3)),
            rotation=np.empty((0, 4)),
        )

        assert len(empty) == 0
