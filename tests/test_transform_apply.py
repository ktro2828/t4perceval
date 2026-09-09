"""Applying a pose to a chunk: which columns move, and how."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_detections, make_predictions, tf_edge, yaw
from scipy.spatial.transform import Rotation

from t4perceval import Chunk, TimePoint, concat_chunks
from t4perceval.component import (
    BatchClassId,
    BatchConfidence,
    BatchFrameId,
    BatchMask,
    BatchNumPoints,
    BatchPosition3D,
    BatchQuaternion,
    BatchSize3D,
    BatchTimeOffset,
    BatchVelocity,
    BatchWaypoints3D,
    Position3D,
    Quaternion,
)
from t4perceval.component.vector import BatchVector3D
from t4perceval.core.component import ColumnarComponent
from t4perceval.core.descriptor import ComponentDescriptor
from t4perceval.descriptors import (
    CHILD_FRAME_ID,
    CLASS_ID,
    CONFIDENCE,
    MASK,
    MODE_CONFIDENCE,
    NUM_POINTS,
    POINT,
    POSITION,
    QUATERNION,
    ROTATION,
    SIZE,
    TIME_OFFSET,
    TRANSLATION,
    VELOCITY,
    WAYPOINTS,
)
from t4perceval.transform import (
    TransformKind,
    compose,
    identity,
    invert,
    pose_of,
    transform_chunk,
    transform_kind,
)

EST = "/estimation/objects"
POSE_90 = (np.array([10.0, 0.0, 0.0]), np.array(yaw(90.0)))
WEIRD = ComponentDescriptor("weird", component_type="Weird")


class Weird(ColumnarComponent):
    SHAPE = (2,)


def box_chunk(positions, *, quaternions=None, velocity=None, mask=None, frame_id="base_link"):  # noqa: ANN001
    chunk = make_detections(positions, velocity=velocity).to_chunk(
        EST, at=TimePoint.at(frame=0), frame_id=frame_id
    )
    extra = {}
    if quaternions is not None:
        extra[QUATERNION] = BatchQuaternion(quaternions)
    if mask is not None:
        extra[MASK] = BatchMask(mask)
    return chunk.with_columns(extra) if extra else chunk


def values(chunk: Chunk, descriptor: ComponentDescriptor) -> np.ndarray:
    return chunk.columns[descriptor].values


class TestKinds:
    def test_the_five_kinds(self) -> None:
        assert {k.name for k in TransformKind} == {
            "POINT",
            "DIRECTION",
            "ROTATION",
            "INVARIANT",
            "DROP",
        }

    @pytest.mark.parametrize(
        ("component", "kind"),
        [
            (BatchPosition3D, TransformKind.POINT),
            (BatchWaypoints3D, TransformKind.POINT),
            (BatchVelocity, TransformKind.DIRECTION),
            (BatchQuaternion, TransformKind.ROTATION),
            (BatchMask, TransformKind.DROP),
        ],
    )
    def test_the_registry(self, component: type, kind: TransformKind) -> None:
        assert transform_kind(component) is kind

    def test_sizes_are_invariant(self) -> None:
        # BatchSize3D shares BatchVector3D with BatchPosition3D; only positions move.
        assert transform_kind(BatchSize3D) is TransformKind.INVARIANT
        assert transform_kind(BatchVector3D) is TransformKind.INVARIANT

    @pytest.mark.parametrize(
        "component", [BatchConfidence, BatchClassId, BatchFrameId, BatchTimeOffset, Weird]
    )
    def test_unknown_classes_are_invariant(self, component: type) -> None:
        assert transform_kind(component) is TransformKind.INVARIANT

    def test_mono_components_and_subclasses_resolve_through_the_mro(self) -> None:
        class MyPosition(BatchPosition3D):
            pass

        assert transform_kind(Position3D) is TransformKind.POINT
        assert transform_kind(Quaternion) is TransformKind.ROTATION
        assert transform_kind(MyPosition) is TransformKind.POINT
        assert transform_kind(BatchMask([True])) is TransformKind.DROP


class TestPoseOf:
    def test_reads_translation_and_xyzw_rotation(self) -> None:
        translation, rotation = pose_of(tf_edge("x", [1.0, 2.0, 3.0], yaw(90.0)))
        assert translation.tolist() == [1.0, 2.0, 3.0]
        np.testing.assert_allclose(rotation, yaw(90.0), atol=1e-12)
        assert translation.shape == (3,) and rotation.shape == (4,)


class TestPoints:
    def test_a_point_is_rotated_then_translated(self) -> None:
        moved = transform_chunk(
            box_chunk([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), POSE_90, target_frame="map"
        )
        np.testing.assert_allclose(
            values(moved, POSITION), [[10.0, 1.0, 0.0], [8.0, 0.0, 0.0]], atol=1e-12
        )

    def test_segmentation_points_transform_as_points(self) -> None:
        chunk = Chunk.from_columns(
            "/points",
            {POINT: BatchPosition3D([[1.0, 0.0, 0.0]]), CLASS_ID: BatchClassId([2])},
            frame_id="lidar",
        )
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        np.testing.assert_allclose(values(moved, POINT), [[10.0, 1.0, 0.0]], atol=1e-12)
        assert values(moved, CLASS_ID).tolist() == [2]

    def test_non_geometry_columns_are_untouched(self) -> None:
        chunk = box_chunk([[1.0, 0.0, 0.0]]).with_columns({NUM_POINTS: BatchNumPoints([7])})
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        for descriptor in (SIZE, CLASS_ID, CONFIDENCE, NUM_POINTS):
            assert np.array_equal(values(moved, descriptor), values(chunk, descriptor))
            assert moved.columns[descriptor] is chunk.columns[descriptor], "shared, not copied"


class TestDirections:
    def test_velocity_is_rotated_not_translated(self) -> None:
        chunk = box_chunk([[0.0, 0.0, 0.0]], velocity=[[1.0, 0.0, 0.0]])
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        np.testing.assert_allclose(values(moved, VELOCITY), [[0.0, 1.0, 0.0]], atol=1e-12)
        assert BatchVelocity(values(moved, VELOCITY)).speed.tolist() == pytest.approx([1.0])


class TestRotations:
    def test_yaw_adds_in_a_yawed_frame(self) -> None:
        moved = transform_chunk(
            box_chunk([[0.0, 0.0, 0.0]], quaternions=[yaw(30.0)]), POSE_90, target_frame="map"
        )
        heading = (
            BatchQuaternion(values(moved, QUATERNION))
            .as_rotation()
            .as_euler("xyz", degrees=True)[:, 2]
        )
        assert heading.tolist() == pytest.approx([120.0])

    def test_the_pose_is_applied_after_the_box_rotation(self) -> None:
        about_x = Rotation.from_euler("x", 90, degrees=True).as_quat()
        moved = transform_chunk(
            box_chunk([[0.0, 0.0, 0.0]], quaternions=[about_x]), POSE_90, target_frame="map"
        )
        forward = Rotation.from_quat(values(moved, QUATERNION)[0]).apply([1.0, 0.0, 0.0])
        # Box x-axis (unchanged by the roll about x) then yawed by the frame: +y. The other
        # composition order would leave it along +x.
        np.testing.assert_allclose(forward, [0.0, 1.0, 0.0], atol=1e-12)

    def test_a_box_front_moves_like_a_point_would(self) -> None:
        about_x = Rotation.from_euler("x", 90, degrees=True).as_quat()
        chunk = box_chunk([[1.0, 0.0, 0.0]], quaternions=[about_x])
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        front = values(moved, POSITION)[0] + Rotation.from_quat(values(moved, QUATERNION)[0]).apply(
            [1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(front, [10.0, 2.0, 0.0], atol=1e-12)

    def test_output_quaternions_are_unit(self) -> None:
        moved = transform_chunk(
            box_chunk([[0.0, 0.0, 0.0]], quaternions=[yaw(30.0)]), POSE_90, target_frame="map"
        )
        np.testing.assert_allclose(
            np.linalg.norm(values(moved, QUATERNION), axis=1), [1.0], atol=1e-12
        )


class TestWaypoints:
    def test_every_waypoint_of_every_mode_is_transformed(self) -> None:
        prediction = make_predictions([[0.0, 0.0, 0.0]], [1], num_modes=2, num_timesteps=2)
        chunk = prediction.to_chunk(
            EST, at=TimePoint.at(frame=0), frame_id="base_link"
        ).with_columns(
            {
                WAYPOINTS: BatchWaypoints3D(
                    [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]]]
                )
            },
        )
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        expected = [[[[10.0, 1.0, 0.0], [10.0, 2.0, 0.0]], [[9.0, 0.0, 0.0], [8.0, 0.0, 0.0]]]]
        np.testing.assert_allclose(values(moved, WAYPOINTS), expected, atol=1e-12)
        assert values(moved, WAYPOINTS).shape == (1, 2, 2, 3)

    def test_rows_are_not_shuffled(self) -> None:
        waypoints = np.arange(2 * 2 * 3 * 3, dtype=np.float64).reshape(2, 2, 3, 3)
        chunk = (
            make_predictions([[0.0, 0.0, 0.0]] * 2, [1, 2], num_modes=2, num_timesteps=3)
            .to_chunk(EST, at=TimePoint.at(frame=0), frame_id="base_link")
            .with_columns({WAYPOINTS: BatchWaypoints3D(waypoints)})
        )
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        translation, quaternion = POSE_90
        expected = (
            Rotation.from_quat(quaternion).apply(waypoints.reshape(-1, 3)).reshape(waypoints.shape)
            + translation
        )
        np.testing.assert_allclose(values(moved, WAYPOINTS), expected, atol=1e-12)
        assert values(moved, TIME_OFFSET).tolist() == values(chunk, TIME_OFFSET).tolist()
        assert np.array_equal(values(moved, MODE_CONFIDENCE), values(chunk, MODE_CONFIDENCE))


class TestColumnsAndStructure:
    def test_masks_are_dropped(self) -> None:
        chunk = box_chunk([[1.0, 0.0, 0.0]], mask=[True])
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        assert MASK not in moved.columns
        assert set(moved.descriptors) == set(chunk.descriptors) - {MASK}

    def test_unknown_components_are_carried_unchanged(self) -> None:
        chunk = box_chunk([[1.0, 0.0, 0.0]]).with_columns({WEIRD: Weird([[1.0, 2.0]])})
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        assert moved.columns[WEIRD] is chunk.columns[WEIRD]

    def test_partitions_indexes_and_offsets_are_preserved(self) -> None:
        first = make_detections([[1.0, 0.0, 0.0]]).to_chunk(
            EST, at=TimePoint.at(frame=0), frame_id="base_link"
        )
        second = make_detections([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]).to_chunk(
            EST, at=TimePoint.at(frame=1), frame_id="base_link"
        )
        chunk = concat_chunks([first, second])
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        assert moved.indexes == chunk.indexes
        assert np.array_equal(moved.offsets, chunk.offsets)
        assert moved.num_partitions == 2
        np.testing.assert_allclose(values(moved, POSITION)[:, 0], [10.0, 10.0, 10.0], atol=1e-12)
        np.testing.assert_allclose(values(moved, POSITION)[:, 1], [1.0, 2.0, 3.0], atol=1e-12)

    def test_zero_rows(self) -> None:
        prediction = make_predictions([], [], num_modes=2, num_timesteps=3)
        chunk = prediction.to_chunk(EST, at=TimePoint.at(frame=0), frame_id="base_link")
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        assert moved.num_rows == 0
        assert moved.offsets.tolist() == [0, 0]
        assert set(moved.descriptors) == set(chunk.descriptors)
        assert values(moved, WAYPOINTS).shape == (0, 2, 3, 3)
        assert moved.frame_id == "map"

    def test_frame_path_and_static_flag(self) -> None:
        chunk = box_chunk([[1.0, 0.0, 0.0]])
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        assert moved.frame_id == "map"
        assert moved.entity_path == chunk.entity_path
        assert (
            str(transform_chunk(chunk, POSE_90, target_frame="map", entity_path="/x/y").entity_path)
            == "/x/y"
        )
        static = Chunk.from_columns(
            EST, {POSITION: BatchPosition3D([[1.0, 0.0, 0.0]])}, is_static=True, frame_id="a"
        )
        assert transform_chunk(static, POSE_90, target_frame="map").is_static

    def test_read_only_input_is_accepted(self) -> None:
        chunk = box_chunk([[1.0, 0.0, 0.0]], velocity=[[1.0, 0.0, 0.0]], quaternions=[yaw(10.0)])
        assert not values(chunk, POSITION).flags.writeable, "components are read-only by design"
        translation, rotation = np.array([1.0, 0.0, 0.0]), np.array(yaw(90.0))
        translation.flags.writeable = False
        rotation.flags.writeable = False
        moved = transform_chunk(chunk, (translation, rotation), target_frame="map")
        np.testing.assert_allclose(values(moved, POSITION), [[1.0, 1.0, 0.0]], atol=1e-12)

    def test_the_source_is_not_mutated(self) -> None:
        chunk = box_chunk([[1.0, 0.0, 0.0]])
        before = values(chunk, POSITION).copy()
        moved = transform_chunk(chunk, POSE_90, target_frame="map")
        assert np.array_equal(values(chunk, POSITION), before)
        assert moved.columns[POSITION] is not chunk.columns[POSITION]
        assert not values(moved, POSITION).flags.writeable


class TestIdentityAndRoundTrip:
    def test_an_identity_pose_is_bit_identical(self) -> None:
        chunk = box_chunk([[1.0, 2.0, 3.0]], velocity=[[0.1, 0.2, 0.3]], quaternions=[yaw(33.0)])
        moved = transform_chunk(chunk, identity(), target_frame="base_link")
        for descriptor in (POSITION, VELOCITY, QUATERNION):
            assert moved.columns[descriptor] is chunk.columns[descriptor]

    def test_invert_undoes_the_transform(self) -> None:
        chunk = box_chunk([[1.0, 2.0, 3.0]], velocity=[[0.1, 0.2, 0.3]], quaternions=[yaw(33.0)])
        back = transform_chunk(
            transform_chunk(chunk, POSE_90, target_frame="map"),
            invert(POSE_90),
            target_frame="base_link",
        )
        for descriptor in (POSITION, VELOCITY):
            np.testing.assert_allclose(
                values(back, descriptor), values(chunk, descriptor), atol=1e-12
            )
        difference = (
            Rotation.from_quat(values(back, QUATERNION))
            * Rotation.from_quat(values(chunk, QUATERNION)).inv()
        )
        assert difference.magnitude().max() < 1e-12

    def test_composition_equals_two_steps(self) -> None:
        chunk = box_chunk([[1.0, 2.0, 3.0]], velocity=[[0.1, 0.2, 0.3]], quaternions=[yaw(33.0)])
        inner = (np.array([1.0, -2.0, 0.5]), np.array(yaw(45.0)))
        outer = (
            np.array([-3.0, 0.0, 2.0]),
            np.array(Rotation.from_euler("x", 30, degrees=True).as_quat()),
        )
        at_once = transform_chunk(chunk, compose(outer, inner), target_frame="c")
        stepwise = transform_chunk(
            transform_chunk(chunk, inner, target_frame="b"), outer, target_frame="c"
        )
        for descriptor in (POSITION, VELOCITY):
            np.testing.assert_allclose(
                values(at_once, descriptor), values(stepwise, descriptor), atol=1e-12
            )
        difference = (
            Rotation.from_quat(values(at_once, QUATERNION))
            * Rotation.from_quat(values(stepwise, QUATERNION)).inv()
        )
        assert difference.magnitude().max() < 1e-12

    def test_a_transform_edge_is_re_parented(self) -> None:
        # TRANSLATION/ROTATION resolve to POINT/ROTATION through the mono classes, so moving
        # an edge `A <- child` into frame B yields exactly `T_B_A @ T_A_child`.
        edge = tf_edge("child", [1.0, 0.0, 0.0], yaw(30.0)).to_chunk(
            "/tf/child", at=TimePoint.at(frame=0), frame_id="A"
        )
        moved = transform_chunk(edge, POSE_90, target_frame="B")
        expected_t, expected_q = compose(POSE_90, (np.array([1.0, 0.0, 0.0]), np.array(yaw(30.0))))
        np.testing.assert_allclose(values(moved, TRANSLATION)[0], expected_t, atol=1e-12)
        difference = (
            Rotation.from_quat(values(moved, ROTATION)[0]) * Rotation.from_quat(expected_q).inv()
        )
        assert difference.magnitude() < 1e-12
        assert values(moved, CHILD_FRAME_ID).tolist() == ["child"]
        assert moved.frame_id == "B"
