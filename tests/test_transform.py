"""Transform edges: how they are recorded, found and read back."""

from __future__ import annotations

import pytest

from t4perceval import FRAME, TIMESTAMP, Store, TimePoint, TimeRange, Transform3D
from t4perceval.descriptors import CHILD_FRAME_ID, ROTATION, TRANSLATION
from t4perceval.io import chunk_from_table, chunk_to_table
from t4perceval.transform import DEFAULT_ROOT, transform_edges

EVERYTHING = TimeRange.everything()


def edge(child: str, x: float = 0.0, z: float = 0.0) -> Transform3D:
    return Transform3D(
        translation=[x, 0.0, z],
        rotation=[0.0, 0.0, 0.0, 1.0],
        child_frame_id=child,
    )


def frames_of(store: Store) -> set[tuple[str, str]]:
    return {edge.frames for edge in transform_edges(store)}


class TestDiscovery:
    def test_edges_come_from_the_data_not_the_path(self) -> None:
        # The entity name says nothing: both edges are filed under names that match no
        # frame, and the graph still comes out right.
        store = Store()
        store.log(
            DEFAULT_ROOT / "ego",
            edge("base_link"),
            at=TimePoint.at(frame=0),
            frame_id="map",
        )
        store.log_static(DEFAULT_ROOT / "calibration", edge("LIDAR_TOP"), frame_id="base_link")

        assert frames_of(store) == {("map", "base_link"), ("base_link", "LIDAR_TOP")}

    def test_a_static_edge_is_found(self) -> None:
        store = Store()
        store.log_static(DEFAULT_ROOT / "lidar", edge("LIDAR_TOP", z=2.0), frame_id="base_link")

        (found,) = transform_edges(store)

        assert found.frames == ("base_link", "LIDAR_TOP")
        assert found.is_static

    def test_a_frame_name_may_contain_a_separator(self) -> None:
        # Nothing parses a frame out of a path any more, so a ROS-namespaced name is fine.
        store = Store()
        store.log_static(DEFAULT_ROOT / "lidar", edge("/robot1/lidar"), frame_id="/robot1/base")

        assert frames_of(store) == {("/robot1/base", "/robot1/lidar")}

    def test_repeated_samples_are_one_edge(self) -> None:
        store = Store()
        for frame in range(3):
            store.log(
                DEFAULT_ROOT / "ego",
                edge("base_link", x=float(frame)),
                at=TimePoint.at(frame=frame),
                frame_id="map",
            )

        assert len(transform_edges(store)) == 1

    def test_unrelated_entities_are_ignored(self) -> None:
        # An entity filed next to the transforms that holds something else must not break
        # discovery -- it simply has no child_frame_id column.
        store = Store()
        store.log(DEFAULT_ROOT / "ego", edge("base_link"), at=TimePoint.at(frame=0), frame_id="map")
        store.log_static_components(
            DEFAULT_ROOT / "notes",
            {TRANSLATION: edge("x").translation},
            frame_id="map",
        )

        assert frames_of(store) == {("map", "base_link")}

    def test_entities_outside_the_root_are_skipped_unless_asked_for(self) -> None:
        store = Store()
        store.log_static("/elsewhere/lidar", edge("LIDAR_TOP"), frame_id="base_link")

        assert transform_edges(store) == ()
        assert {e.frames for e in transform_edges(store, root=None)} == {
            ("base_link", "LIDAR_TOP"),
        }

    def test_an_empty_store_has_no_edges(self) -> None:
        assert transform_edges(Store()) == ()

    def test_a_transform_without_a_parent_frame_is_rejected(self) -> None:
        # Unlike the comparison guard, where an unstated frame is merely no opinion, an
        # unparented transform cannot be interpreted at all.
        store = Store()
        store.log(DEFAULT_ROOT / "ego", edge("base_link"), at=TimePoint.at(frame=0))

        with pytest.raises(ValueError, match="states no frame_id"):
            transform_edges(store)

    def test_the_same_edge_recorded_twice_is_rejected(self) -> None:
        store = Store()
        store.log_static(DEFAULT_ROOT / "a", edge("LIDAR_TOP", z=2.0), frame_id="base_link")
        store.log_static(DEFAULT_ROOT / "b", edge("LIDAR_TOP", z=9.0), frame_id="base_link")

        with pytest.raises(ValueError, match="recorded twice"):
            transform_edges(store)


class TestRecording:
    def test_a_static_edge_keeps_its_parent_frame(self) -> None:
        # The whole point of the static rewrite: a calibration states the frame it is
        # expressed in, so the edge stays interpretable without a fake timestamp.
        store = Store()
        path = DEFAULT_ROOT / "LIDAR_TOP"
        store.log_static(path, edge("LIDAR_TOP", z=2.0), frame_id="base_link")

        chunk = store.static_chunks(path)[0]

        assert store.static_frame_id(path) == "base_link"
        # Stored columnar, one row: mono is the archetype's shape, not the chunk's.
        assert chunk.component(TRANSLATION).values.tolist() == [[0.0, 0.0, 2.0]]
        assert chunk.component(CHILD_FRAME_ID).names() == ("LIDAR_TOP",)
        assert chunk.component(TRANSLATION).__class__.__name__ == "BatchPosition3D"

    def test_a_static_edge_needs_no_time(self) -> None:
        # It is not on a timeline at all, which is what `static` now means -- so there is
        # no window a range query could miss it from.
        store = Store()
        path = DEFAULT_ROOT / "LIDAR_TOP"
        store.log_static(path, edge("LIDAR_TOP", z=2.0), frame_id="base_link")

        assert store.times(path, FRAME).tolist() == []
        assert store.static_chunks(path)[0].timelines == ()

    def test_a_temporal_edge_is_queryable_on_either_timeline(self) -> None:
        store = Store()
        path = DEFAULT_ROOT / "base_link"
        store.log(
            path,
            edge("base_link", x=10.0),
            at=TimePoint.at(frame=1, timestamp_ns=1_000_000_000),
            frame_id="map",
        )

        for timeline, at in ((FRAME, 1), (TIMESTAMP, 1_000_000_000)):
            assert len(store.latest_at(path, timeline=timeline, at=at)) == 1

    def test_a_series_reads_back_in_order(self) -> None:
        # One sample is one chunk, so a scene of ego poses is read column-wise -- the
        # archetype describes a single edge and cannot hold the series.
        store = Store()
        path = DEFAULT_ROOT / "base_link"
        for frame, x in enumerate((0.0, 10.0, 20.0)):
            store.log(
                path,
                edge("base_link", x=x),
                at=TimePoint.at(frame=frame),
                frame_id="map",
            )

        view = store.range(path, timeline=FRAME, time_range=EVERYTHING)

        assert view.component(TRANSLATION).values[:, 0].tolist() == [0.0, 10.0, 20.0]
        with pytest.raises(ValueError, match="exactly one value"):
            view.materialize(Transform3D)


class TestArchetype:
    def test_it_declares_three_columns(self) -> None:
        assert [d.component for d in Transform3D.required_descriptors()] == [
            "translation",
            "rotation",
            "child_frame_id",
        ]

    def test_its_descriptors_are_not_the_object_ones(self) -> None:
        # Distinct from POSITION/QUATERNION so a system asking for a 3D position cannot be
        # pointed at a transform entity and appear to work.
        from t4perceval.descriptors import POSITION, QUATERNION

        assert TRANSLATION != POSITION
        assert ROTATION != QUATERNION

    def test_a_missing_child_frame_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            Transform3D(translation=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0])  # type: ignore[call-arg]

    def test_its_components_are_values_not_columns(self) -> None:
        # Every other archetype describes N objects; this one describes one relationship,
        # so there is no row to index into and no length to disagree about.
        transform = edge("LIDAR_TOP", z=2.0)

        assert transform.translation.value.tolist() == [0.0, 0.0, 2.0]
        assert transform.rotation.value.tolist() == [0.0, 0.0, 0.0, 1.0]
        assert transform.child_frame_id.name == "LIDAR_TOP"
        assert len(transform) == 1

    def test_more_than_one_edge_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one value"):
            Transform3D(
                translation=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                rotation=[[0.0, 0.0, 0.0, 1.0]] * 2,
                child_frame_id="LIDAR_TOP",
            )

    @pytest.mark.parametrize("is_static", [False, True])
    def test_a_chunk_round_trips_through_arrow(self, is_static: bool) -> None:
        # Persistence needs no transform-specific IO, static or temporal: an edge is an
        # ordinary chunk, text column and all.
        at = None if is_static else TimePoint.at(frame=0)
        chunk = edge("LIDAR_TOP", z=2.0).to_chunk(
            DEFAULT_ROOT / "LIDAR_TOP",
            at=at,
            frame_id="base_link",
            is_static=is_static,
        )

        restored, _ = chunk_from_table(chunk_to_table(chunk))

        assert restored == chunk
        assert restored.frame_id == "base_link"
        assert restored.is_static is is_static
