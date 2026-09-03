"""Walking the frame graph: inversion, composition and the lookup policies."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from t4perceval import FRAME, Chunk, Store, TimePoint, Transform3D
from t4perceval.component import BatchFrameId, BatchPosition3D, BatchQuaternion
from t4perceval.core.timeline import TimeColumn
from t4perceval.descriptors import CHILD_FRAME_ID, ROTATION, TRANSLATION
from t4perceval.io import chunk_from_table, chunk_to_table
from t4perceval.transform import (
    DEFAULT_ROOT,
    FrameGraph,
    LookupPolicy,
    TransformResolver,
    compose,
    invert,
)

#: base_link sits 2 m above the lidar's parent, unrotated.
LIDAR_OFFSET = [0.0, 0.0, 2.0]


def yaw(degrees: float) -> list[float]:
    return list(Rotation.from_euler("z", degrees, degrees=True).as_quat())


def transform(
    child: str, translation: list[float], rotation: list[float] | None = None
) -> Transform3D:
    return Transform3D(
        translation=[translation],
        rotation=[rotation or [0.0, 0.0, 0.0, 1.0]],
        child_frame_id=[child],
    )


@pytest.fixture
def scene() -> Store:
    """A store shaped like an imported scene: temporal ego pose, static calibration."""
    store = Store()
    for frame, x in enumerate((0.0, 10.0, 20.0)):
        store.log(
            DEFAULT_ROOT / "base_link",
            transform("base_link", [x, 0.0, 0.0]),
            at=TimePoint.at(frame=frame),
            frame_id="map",
        )
    store.log_static(
        DEFAULT_ROOT / "LIDAR_TOP",
        transform("LIDAR_TOP", LIDAR_OFFSET),
        frame_id="base_link",
    )
    return store


class TestGraphPaths:
    def test_a_frame_reaches_itself_without_a_hop(self, scene: Store) -> None:
        assert FrameGraph.of(scene).path(target_frame="map", source_frame="map") == ()

    def test_one_hop_is_walked_as_recorded(self, scene: Store) -> None:
        (hop,) = FrameGraph.of(scene).path(target_frame="map", source_frame="base_link")
        edge, inverted = hop

        assert edge.frames == ("map", "base_link")
        assert inverted is False

    def test_the_other_direction_inverts_the_same_edge(self, scene: Store) -> None:
        # A rigid transform inverts exactly, so recording one direction answers both.
        (hop,) = FrameGraph.of(scene).path(target_frame="base_link", source_frame="map")
        edge, inverted = hop

        assert edge.frames == ("map", "base_link")
        assert inverted is True

    def test_a_chain_is_ordered_from_the_source(self, scene: Store) -> None:
        hops = FrameGraph.of(scene).path(target_frame="map", source_frame="LIDAR_TOP")

        assert [edge.frames for edge, _ in hops] == [
            ("base_link", "LIDAR_TOP"),
            ("map", "base_link"),
        ]

    def test_an_unknown_frame_is_rejected(self, scene: Store) -> None:
        with pytest.raises(ValueError, match="Unknown coordinate frame 'radar'"):
            FrameGraph.of(scene).path(target_frame="map", source_frame="radar")

    def test_a_disconnected_frame_is_rejected(self) -> None:
        # Never quietly identity: a missing calibration must be loud.
        store = Store()
        store.log_static(DEFAULT_ROOT / "a", transform("lidar", LIDAR_OFFSET), frame_id="base_link")
        store.log_static(DEFAULT_ROOT / "b", transform("cam", LIDAR_OFFSET), frame_id="rig")

        with pytest.raises(ValueError, match="No recorded transform connects"):
            FrameGraph.of(store).path(target_frame="base_link", source_frame="cam")


class TestLookup:
    def test_a_frame_is_the_identity_of_itself(self, scene: Store) -> None:
        pose = TransformResolver.of(scene).lookup(target_frame="map", source_frame="map", at=0)

        assert pose.translation.value.tolist() == [0.0, 0.0, 0.0]
        assert pose.rotation.value.tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_it_composes_a_static_edge_with_a_temporal_one(self, scene: Store) -> None:
        # T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar
        resolver = TransformResolver.of(scene, timeline=FRAME)

        pose = resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=1)

        assert pose.translation.value.tolist() == [10.0, 0.0, 2.0]
        assert pose.child_frame_id.name == "LIDAR_TOP"

    def test_the_answer_states_the_frame_it_is_of(self, scene: Store) -> None:
        pose = TransformResolver.of(scene).lookup(
            target_frame="LIDAR_TOP",
            source_frame="map",
            at=0,
        )

        assert pose.child_frame_id.name == "map"

    def test_rotation_carries_into_the_composed_translation(self) -> None:
        # A quarter turn about z puts a sensor 1 m ahead of the ego 1 m to the ego's left
        # in the map -- the case a translation-only composition gets wrong.
        store = Store()
        store.log(
            DEFAULT_ROOT / "base_link",
            transform("base_link", [0.0, 0.0, 0.0], yaw(90.0)),
            at=TimePoint.at(frame=0),
            frame_id="map",
        )
        store.log_static(
            DEFAULT_ROOT / "cam",
            transform("cam", [1.0, 0.0, 0.0]),
            frame_id="base_link",
        )

        pose = TransformResolver.of(store).lookup(target_frame="map", source_frame="cam", at=0)

        np.testing.assert_allclose(pose.translation.value, [0.0, 1.0, 0.0], atol=1e-12)

    def test_the_two_directions_undo_each_other(self, scene: Store) -> None:
        resolver = TransformResolver.of(scene)
        forward = resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=2)
        backward = resolver.lookup(target_frame="LIDAR_TOP", source_frame="map", at=2)

        translation, rotation = compose(
            (forward.translation.value, forward.rotation.value),
            (backward.translation.value, backward.rotation.value),
        )

        np.testing.assert_allclose(translation, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.abs(rotation), [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_inverting_a_pose_matches_looking_it_up_backwards(self, scene: Store) -> None:
        resolver = TransformResolver.of(scene)
        forward = resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=1)
        backward = resolver.lookup(target_frame="LIDAR_TOP", source_frame="map", at=1)

        translation, _ = invert((forward.translation.value, forward.rotation.value))

        np.testing.assert_allclose(translation, backward.translation.value, atol=1e-12)

    def test_a_static_chain_needs_no_time(self) -> None:
        store = Store()
        store.log_static(
            DEFAULT_ROOT / "LIDAR_TOP",
            transform("LIDAR_TOP", LIDAR_OFFSET),
            frame_id="base_link",
        )

        pose = TransformResolver.of(store).lookup(
            target_frame="base_link",
            source_frame="LIDAR_TOP",
        )

        assert pose.translation.value.tolist() == LIDAR_OFFSET

    def test_a_temporal_chain_without_a_time_is_rejected(self, scene: Store) -> None:
        with pytest.raises(ValueError, match="needs a time on the 'frame' timeline"):
            TransformResolver.of(scene).lookup(target_frame="map", source_frame="LIDAR_TOP")

    def test_a_child_missing_at_that_time_is_reported(self) -> None:
        store = Store()
        store.log(
            DEFAULT_ROOT / "ego",
            transform("base_link", [0.0, 0.0, 0.0]),
            at=TimePoint.at(frame=5),
            frame_id="map",
        )

        with pytest.raises(ValueError, match="holds no transform for child frame"):
            TransformResolver.of(store).lookup(target_frame="map", source_frame="base_link", at=0)

    def test_two_poses_for_one_child_at_one_time_are_rejected(self) -> None:
        # `Transform3D` cannot express this -- it is one edge -- so the chunk is built by
        # hand, which is the only way the ambiguity can reach the resolver.
        store = Store()
        store.send_chunk(
            Chunk.from_columns(
                DEFAULT_ROOT / "ego",
                {
                    TRANSLATION: BatchPosition3D([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
                    ROTATION: BatchQuaternion([[0.0, 0.0, 0.0, 1.0]] * 2),
                    CHILD_FRAME_ID: BatchFrameId(["base_link", "base_link"]),
                },
                indexes=(TimeColumn.of(FRAME, [0]),),
                frame_id="map",
            ),
        )

        with pytest.raises(ValueError, match="cannot be in two places"):
            TransformResolver.of(store).lookup(target_frame="map", source_frame="base_link", at=0)


class TestPolicies:
    def test_latest_reaches_backwards(self, scene: Store) -> None:
        resolver = TransformResolver.of(scene, policy=LookupPolicy.LATEST)

        pose = resolver.lookup(target_frame="map", source_frame="base_link", at=999)

        assert pose.translation.value[0] == 20.0

    def test_exact_refuses_a_time_with_no_sample(self, scene: Store) -> None:
        resolver = TransformResolver.of(scene, policy=LookupPolicy.EXACT)

        assert (
            resolver.lookup(target_frame="map", source_frame="base_link", at=2).translation.values[
                0
            ][0]
            == 20.0
        )
        with pytest.raises(ValueError, match="no sample at frame=3"):
            resolver.lookup(target_frame="map", source_frame="base_link", at=3)

    def test_nearest_may_reach_forwards(self) -> None:
        store = Store()
        for frame, x in ((0, 0.0), (10, 100.0)):
            store.log(
                DEFAULT_ROOT / "base_link",
                transform("base_link", [x, 0.0, 0.0]),
                at=TimePoint.at(frame=frame),
                frame_id="map",
            )
        resolver = TransformResolver.of(store, policy=LookupPolicy.NEAREST)

        assert (
            resolver.lookup(target_frame="map", source_frame="base_link", at=7).translation.values[
                0
            ][0]
            == 100.0
        )

    def test_interpolate_lands_between_two_samples(self) -> None:
        store = Store()
        store.log(
            DEFAULT_ROOT / "base_link",
            transform("base_link", [0.0, 0.0, 0.0], yaw(0.0)),
            at=TimePoint.at(frame=0),
            frame_id="map",
        )
        store.log(
            DEFAULT_ROOT / "base_link",
            transform("base_link", [10.0, 0.0, 0.0], yaw(90.0)),
            at=TimePoint.at(frame=10),
            frame_id="map",
        )
        resolver = TransformResolver.of(store, policy=LookupPolicy.INTERPOLATE)

        pose = resolver.lookup(target_frame="map", source_frame="base_link", at=5)
        angle = pose.rotation.as_rotation().as_euler("xyz", degrees=True)[2]

        np.testing.assert_allclose(pose.translation.value, [5.0, 0.0, 0.0], atol=1e-12)
        assert angle == pytest.approx(45.0)

    def test_interpolate_holds_the_ends(self, scene: Store) -> None:
        # Beyond the recorded span there is nothing to interpolate between, and guessing
        # would invent motion the recording never saw.
        resolver = TransformResolver.of(scene, policy=LookupPolicy.INTERPOLATE)

        assert (
            resolver.lookup(target_frame="map", source_frame="base_link", at=999).translation.value[
                0
            ]
            == 20.0
        )

    def test_a_static_edge_ignores_the_policy(self, scene: Store) -> None:
        # "Interpolate something that never changes" is a question with one answer, not an
        # error.
        for policy in LookupPolicy:
            resolver = TransformResolver.of(scene, policy=policy)

            pose = resolver.lookup(target_frame="base_link", source_frame="LIDAR_TOP", at=1)

            assert pose.translation.value.tolist() == LIDAR_OFFSET


class TestPersistence:
    def test_a_graph_rebuilds_from_restored_chunks(self, scene: Store) -> None:
        # What "a saved recording still knows its frame tree" reduces to: chunks in, the
        # same graph out. Whole-recording IO does not exist yet; this is the half that
        # does not depend on it.
        restored = Store()
        for path in scene.entity_paths():
            for chunk in (*scene.static_chunks(path), *scene.chunks(path)):
                restored.send_chunk(chunk_from_table(chunk_to_table(chunk))[0])

        assert FrameGraph.of(restored) == FrameGraph.of(scene)
        assert TransformResolver.of(restored).lookup(
            target_frame="map",
            source_frame="LIDAR_TOP",
            at=1,
        ).translation.value.tolist() == [10.0, 0.0, 2.0]
