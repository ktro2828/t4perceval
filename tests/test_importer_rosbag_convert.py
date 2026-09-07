"""Converting decoded Autoware messages into columns, without a bag or the MCAP libraries.

The converter reads attributes only, so plain namespaces stand in for decoded messages.
"""

from __future__ import annotations

import math
from types import SimpleNamespace as ns
from typing import Any

import numpy as np
import pytest

from t4perceval import InstanceRegistry, LabelRegistry
from t4perceval.component import BatchQuaternion
from t4perceval.importer.rosbag.convert import (
    SHAPE_CYLINDER,
    kind_of_schema,
    objects_to_columns,
    stamp_ns,
    trajectory_shape_of,
)
from t4perceval.importer.rosbag.labels import (
    AUTOWARE_CLASS_NAMES,
    class_name_of,
    classification_name,
    label_registry_from_autoware,
)

NS = 1_000_000_000
UUID_BYTES = bytes(range(16))
UUID_STR = "00010203-0405-0607-0809-0a0b0c0d0e0f"


# -- message stand-ins --------------------------------------------------------------------


def xyz(values: tuple[float, float, float]) -> Any:
    return ns(x=values[0], y=values[1], z=values[2])


def yawed(degrees: float) -> Any:
    half = math.radians(degrees) / 2
    return ns(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def pose(position: tuple[float, float, float] = (1.0, 2.0, 0.0), yaw_deg: float = 0.0) -> Any:
    return ns(position=xyz(position), orientation=yawed(yaw_deg))


def classified(*pairs: tuple[int, float]) -> list[Any]:
    return [ns(label=label, probability=probability) for label, probability in pairs]


def shape(dims: tuple[float, float, float] = (4.0, 2.0, 1.5), kind: int = 0) -> Any:
    return ns(type=kind, dimensions=xyz(dims))


def detected(
    *,
    position: tuple[float, float, float] = (1.0, 2.0, 0.0),
    yaw_deg: float = 0.0,
    dims: tuple[float, float, float] = (4.0, 2.0, 1.5),
    shape_type: int = 0,
    classification: list[Any] | None = None,
    existence: float = 0.5,
    twist: tuple[float, float, float] | None = None,
) -> Any:
    return ns(
        existence_probability=existence,
        classification=classification if classification is not None else classified((1, 0.9)),
        kinematics=ns(
            pose_with_covariance=ns(pose=pose(position, yaw_deg)),
            twist_with_covariance=ns(twist=ns(linear=xyz(twist or (0.0, 0.0, 0.0)))),
            has_twist=twist is not None,
        ),
        shape=shape(dims, shape_type),
    )


def tracked(*, uuid: Any = UUID_BYTES, twist: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Any:
    return ns(
        object_id=ns(uuid=uuid),
        existence_probability=0.5,
        classification=classified((1, 0.9)),
        kinematics=ns(
            pose_with_covariance=ns(pose=pose()),
            twist_with_covariance=ns(twist=ns(linear=xyz(twist))),
        ),
        shape=shape(),
    )


def path(
    positions: list[tuple[float, float, float]], *, step_ns: int = NS // 2, confidence: float = 0.5
) -> Any:
    return ns(
        path=[pose(p) for p in positions],
        time_step=ns(sec=step_ns // NS, nanosec=step_ns % NS),
        confidence=confidence,
    )


def predicted(*, paths: list[Any], position: tuple[float, float, float] = (1.0, 2.0, 0.0)) -> Any:
    return ns(
        object_id=ns(uuid=UUID_BYTES),
        existence_probability=0.5,
        classification=classified((1, 0.9)),
        kinematics=ns(
            initial_pose_with_covariance=ns(pose=pose(position)),
            initial_twist_with_covariance=ns(twist=ns(linear=xyz((1.0, 0.0, 0.0)))),
            predicted_paths=paths,
        ),
        shape=shape(),
    )


def message(objects: list[Any], frame_id: str = "map") -> Any:
    return ns(header=ns(stamp=ns(sec=1, nanosec=5), frame_id=frame_id), objects=objects)


@pytest.fixture
def labels() -> LabelRegistry:
    return label_registry_from_autoware()


def convert(msg: Any, kind: str = "detections", **kwargs: Any) -> Any:
    kwargs.setdefault("labels", label_registry_from_autoware())
    kwargs.setdefault("instances", InstanceRegistry())
    return objects_to_columns(msg, kind=kind, **kwargs)  # type: ignore[arg-type]


# -- schemas and labels -------------------------------------------------------------------


class TestSchemas:
    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("autoware_perception_msgs/msg/DetectedObjects", "detections"),
            ("autoware_perception_msgs/msg/TrackedObjects", "trackings"),
            ("autoware_perception_msgs/msg/PredictedObjects", "predictions"),
            ("autoware_auto_perception_msgs/msg/TrackedObjects", "trackings"),
        ],
    )
    def test_kind_follows_the_message_name(self, name: str, kind: str) -> None:
        assert kind_of_schema(name) == kind

    def test_anything_else_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not an Autoware object message"):
            kind_of_schema("sensor_msgs/msg/PointCloud2")

    def test_a_stamp_folds_to_nanoseconds(self) -> None:
        assert stamp_ns(ns(sec=3, nanosec=7)) == 3 * NS + 7


class TestLabels:
    def test_the_enum_maps_to_canonical_names(self) -> None:
        assert class_name_of(1) == "car"
        assert class_name_of(7) == "pedestrian"
        assert AUTOWARE_CLASS_NAMES[0] == "unknown"

    def test_an_unknown_enum_value_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="outside the known enum"):
            class_name_of(len(AUTOWARE_CLASS_NAMES))

    def test_the_most_probable_classification_names_the_object(self) -> None:
        assert classification_name(classified((1, 0.2), (2, 0.7), (7, 0.1))) == "truck"

    def test_no_classification_is_unknown(self) -> None:
        assert classification_name([]) == "unknown"

    def test_the_registry_follows_enum_order(self) -> None:
        registry = label_registry_from_autoware()

        assert registry.names == AUTOWARE_CLASS_NAMES
        assert registry.class_id("unknown") == 0

    def test_unknown_labels_policy_applies_to_missing_names(self) -> None:
        narrow = LabelRegistry.from_names(["car"])
        msg = message([detected(), detected(classification=classified((7, 0.9)))])

        with pytest.raises(KeyError, match="pedestrian"):
            convert(msg, labels=narrow)

        dropped = convert(msg, labels=narrow, unknown_labels="drop")
        assert dropped.kept.tolist() == [0]
        assert len(dropped) == 1


# -- geometry -----------------------------------------------------------------------------


class TestGeometry:
    def test_dimensions_swap_length_and_width(self) -> None:
        columns = convert(message([detected(dims=(4.0, 2.0, 1.5))]))

        assert columns.size.tolist() == [[2.0, 4.0, 1.5]]

    def test_a_cylinder_is_as_wide_as_it_is_long(self) -> None:
        columns = convert(message([detected(dims=(1.0, 7.0, 2.0), shape_type=SHAPE_CYLINDER)]))

        assert columns.size.tolist() == [[1.0, 1.0, 2.0]]

    def test_quaternions_are_not_reordered(self) -> None:
        columns = convert(message([detected(yaw_deg=30.0)]))

        assert BatchQuaternion(columns.quaternion).yaw()[0] == pytest.approx(math.radians(30.0))

    def test_a_zero_quaternion_is_an_error(self) -> None:
        obj = detected()
        obj.kinematics.pose_with_covariance.pose.orientation = ns(x=0.0, y=0.0, z=0.0, w=0.0)

        with pytest.raises(ValueError, match="zero quaternion"):
            convert(message([obj]))

    def test_an_empty_message_keeps_its_shapes_and_frame(self) -> None:
        columns = convert(message([], frame_id="base_link"))

        assert len(columns) == 0
        assert columns.position.shape == (0, 3)
        assert columns.quaternion.shape == (0, 4)
        assert columns.size.shape == (0, 3)
        assert columns.class_id.shape == (0,)
        assert columns.frame_id == "base_link"
        assert columns.as_detections().position.values.shape == (0, 3)


class TestVelocity:
    def test_a_body_frame_twist_is_rotated_into_the_message_frame(self) -> None:
        columns = convert(message([detected(yaw_deg=90.0, twist=(1.0, 0.0, 0.0))]))

        assert columns.velocity is not None
        np.testing.assert_allclose(columns.velocity, [[0.0, 1.0, 0.0]], atol=1e-12)

    def test_an_object_without_a_twist_is_nan(self) -> None:
        columns = convert(message([detected(twist=(1.0, 0.0, 0.0)), detected()]))

        assert columns.velocity is not None
        assert np.isfinite(columns.velocity[0]).all()
        assert np.isnan(columns.velocity[1]).all()

    def test_auto_emits_nothing_when_no_object_has_a_twist(self) -> None:
        assert convert(message([detected(), detected()])).velocity is None

    def test_always_emits_a_nan_column(self) -> None:
        columns = convert(message([detected()]), velocity="always")

        assert columns.velocity is not None
        assert np.isnan(columns.velocity).all()

    def test_never_emits_nothing(self) -> None:
        assert (
            convert(message([detected(twist=(1.0, 0.0, 0.0))]), velocity="never").velocity is None
        )

    def test_tracked_objects_always_carry_a_twist(self) -> None:
        columns = convert(message([tracked(twist=(3.0, 0.0, 0.0))]), kind="trackings")

        assert columns.velocity is not None
        np.testing.assert_allclose(columns.velocity, [[3.0, 0.0, 0.0]])


class TestConfidence:
    @pytest.fixture
    def msg(self) -> Any:
        return message([detected(classification=classified((1, 0.8), (2, 0.2)), existence=0.5)])

    def test_classification_is_the_top_probability(self, msg: Any) -> None:
        assert convert(msg).confidence.tolist() == [pytest.approx(0.8)]

    def test_existence_is_the_existence_probability(self, msg: Any) -> None:
        assert convert(msg, confidence="existence").confidence.tolist() == [pytest.approx(0.5)]

    def test_product_multiplies_the_two(self, msg: Any) -> None:
        assert convert(msg, confidence="product").confidence.tolist() == [pytest.approx(0.4)]

    def test_no_classification_falls_back_to_existence(self) -> None:
        msg = message([detected(classification=[], existence=0.3)])

        assert convert(msg).confidence.tolist() == [pytest.approx(0.3)]
        assert convert(msg).class_id.tolist() == [0]


# -- identities ---------------------------------------------------------------------------


class TestInstances:
    def test_detections_have_no_identity(self) -> None:
        columns = convert(message([detected()]))

        assert columns.instance_id is None
        with pytest.raises(ValueError, match="no object_id"):
            columns.as_trackings()
        with pytest.raises(ValueError, match="Predictions3D"):
            columns.as_predictions()

    def test_uuid_bytes_and_int_lists_spell_the_same_identity(self) -> None:
        registry = InstanceRegistry()
        as_bytes = convert(
            message([tracked(uuid=UUID_BYTES)]), kind="trackings", instances=registry
        )
        as_list = convert(
            message([tracked(uuid=list(UUID_BYTES))]),
            kind="trackings",
            instances=registry,
        )

        assert as_bytes.instance_id is not None and as_list.instance_id is not None
        assert as_bytes.instance_id.tolist() == as_list.instance_id.tolist()
        assert registry.uuid(int(as_bytes.instance_id[0])) == UUID_STR

    def test_identities_are_namespaced(self) -> None:
        registry = InstanceRegistry()
        columns = convert(
            message([tracked()]),
            kind="trackings",
            instances=registry,
            instance_namespace="est",
        )

        assert columns.instance_id is not None
        assert registry.uuid(int(columns.instance_id[0])) == f"est/{UUID_STR}"


# -- trajectories -------------------------------------------------------------------------


class TestTrajectories:
    def two_paths(self) -> Any:
        return predicted(
            position=(10.0, 0.0, 0.0),
            paths=[
                path(
                    [(10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (12.0, 0.0, 0.0), (13.0, 0.0, 0.0)],
                    confidence=0.7,
                ),
                path([(10.0, 0.0, 0.0), (10.5, 0.5, 0.0)], confidence=0.3),
            ],
        )

    def test_the_shape_covers_every_object_and_drops_the_current_pose(self) -> None:
        assert trajectory_shape_of([message([self.two_paths(), predicted(paths=[])])]) == (2, 3)

    def test_a_topic_without_paths_still_has_a_shape(self) -> None:
        assert trajectory_shape_of([message([predicted(paths=[])]), message([])]) == (1, 1)

    def test_a_single_pose_path_predicts_nothing(self) -> None:
        assert trajectory_shape_of([message([predicted(paths=[path([(1.0, 1.0, 0.0)])])])]) == (
            1,
            1,
        )

    def test_paths_become_padded_masked_modes(self) -> None:
        columns = convert(
            message([self.two_paths(), predicted(paths=[], position=(5.0, 3.0, 0.0))]),
            kind="predictions",
            trajectory=(2, 3),
        )
        trajectory = columns.trajectory
        assert trajectory is not None

        # The first pose is the present, so three future waypoints remain.
        assert trajectory.waypoints[0, 0].tolist() == [[11, 0, 0], [12, 0, 0], [13, 0, 0]]
        # A shorter mode repeats its last real waypoint rather than jumping anywhere.
        assert trajectory.waypoints[0, 1].tolist() == [[10.5, 0.5, 0]] * 3
        assert trajectory.timestep_valid[0].tolist() == [[True] * 3, [True, False, False]]
        assert trajectory.mode_valid.tolist() == [[True, True], [False, False]]
        assert trajectory.mode_confidence[0].tolist() == [pytest.approx(0.7), pytest.approx(0.3)]
        # Offsets are multiples of the path's time step, out to the padded end.
        assert trajectory.time_offset[0].tolist() == [NS // 2, NS, 3 * NS // 2]

        # An object with no path holds station and is fully masked.
        assert trajectory.waypoints[1].tolist() == [[[5.0, 3.0, 0.0]] * 3] * 2
        assert trajectory.mode_confidence[1].tolist() == [0.0, 0.0]
        assert trajectory.time_offset[1].tolist() == [1, 2, 3]

        columns.as_predictions()  # every constraint the archetype checks holds

    def test_a_pinned_shape_too_small_for_the_data_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            convert(message([self.two_paths()]), kind="predictions", trajectory=(1, 3))
        with pytest.raises(ValueError, match="does not fit"):
            convert(message([self.two_paths()]), kind="predictions", trajectory=(2, 2))

    def test_a_larger_pinned_shape_pads(self) -> None:
        columns = convert(message([self.two_paths()]), kind="predictions", trajectory=(3, 5))

        assert columns.trajectory is not None
        assert columns.trajectory.waypoints.shape == (1, 3, 5, 3)
        assert columns.trajectory.time_offset[0].tolist() == [NS // 2 * i for i in range(1, 6)]

    def test_paths_of_one_object_must_share_a_time_step(self) -> None:
        obj = predicted(
            paths=[
                path([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], step_ns=NS),
                path([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], step_ns=NS // 2),
            ],
        )
        with pytest.raises(ValueError, match="different time steps"):
            convert(message([obj]), kind="predictions", trajectory=(2, 1))

    def test_a_zero_time_step_is_an_error(self) -> None:
        obj = predicted(paths=[path([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], step_ns=0)])
        with pytest.raises(ValueError, match="non-positive"):
            convert(message([obj]), kind="predictions", trajectory=(1, 1))

    def test_predictions_read_the_initial_pose_and_twist(self) -> None:
        columns = convert(
            message([predicted(paths=[], position=(7.0, 8.0, 0.0))]), kind="predictions"
        )

        assert columns.position.tolist() == [[7.0, 8.0, 0.0]]
        assert columns.velocity is not None
        np.testing.assert_allclose(columns.velocity, [[1.0, 0.0, 0.0]])
