"""Conversion from devkit boxes to columns, without touching a dataset."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import Detections3D, InstanceRegistry, LabelRegistry, Trackings3D
from t4perceval.component import BatchQuaternion, BatchRoi, VisibilityLevel
from t4perceval.importer.t4.convert import (
    boxes2d_to_columns,
    boxes3d_to_columns,
    trajectory_shape_of,
)

# Before importing the builders: they construct real devkit boxes at module scope, so
# without the extra this file must skip rather than fail to collect.
pytest.importorskip("t4_devkit")

DEVKIT_VISIBILITY = pytest.importorskip("t4_devkit.schema").VisibilityLevel

from tests.t4_builder import (  # noqa: E402
    FRAME_TIME_US,
    box2d,
    box3d,
    columns_of,
    future_of,
    yawed,
)


class TestGeometry:
    def test_quaternion_is_reordered_to_xyzw(self) -> None:
        columns = columns_of([box3d(rotation=yawed(30.0))])

        # pyquaternion stores wxyz and this package stores xyzw. Both are (4,) float
        # columns, so taking the elements verbatim yields a plausible rotation rather
        # than an error -- here it would silently read as no rotation at all.
        assert BatchQuaternion(columns.quaternion).yaw() == pytest.approx([np.deg2rad(30.0)])

    def test_taking_the_devkit_order_verbatim_would_lose_the_rotation(self) -> None:
        box = box3d(rotation=yawed(30.0))
        verbatim = BatchQuaternion(np.asarray([box.rotation.elements]))

        assert verbatim.yaw() == pytest.approx([0.0])

    def test_quaternions_are_normalized(self) -> None:
        columns = columns_of([box3d(rotation=(2.0, 0.0, 0.0, 0.0))])

        assert np.linalg.norm(columns.quaternion, axis=1) == pytest.approx([1.0])

    def test_size_is_copied_verbatim(self) -> None:
        columns = columns_of([box3d(size=(2.0, 4.5, 1.6))])

        # Both sides order it (width, length, height), so this one needs no permutation.
        assert columns.size.tolist() == [[2.0, 4.5, 1.6]]


class TestRoi:
    def test_corners_become_offset_and_extent(self) -> None:
        columns = boxes2d_to_columns(
            [box2d(roi=(100, 100, 200, 150))],
            labels=LabelRegistry.from_names(["car"]),
            instances=InstanceRegistry(),
        )

        # The devkit stores (xmin, ymin, xmax, ymax); this package stores
        # (x_min, y_min, height, width). Four ints either way.
        assert columns.roi.tolist() == [[100, 100, 50, 100]]

    def test_the_original_corners_are_recoverable(self) -> None:
        columns = boxes2d_to_columns(
            [box2d(roi=(100, 100, 200, 150))],
            labels=LabelRegistry.from_names(["car"]),
            instances=InstanceRegistry(),
        )
        roi = BatchRoi(columns.roi)

        assert (int(roi.x_max[0]), int(roi.y_max[0])) == (200, 150)

    def test_a_box_without_a_roi_is_named(self) -> None:
        with pytest.raises(ValueError, match="Box 1 has no region of interest"):
            boxes2d_to_columns(
                [box2d(), box2d(roi=None)],
                labels=LabelRegistry.from_names(["car"]),
                instances=InstanceRegistry(),
            )


class TestVisibility:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            ("full", VisibilityLevel.FULL),
            ("most", VisibilityLevel.MOST),
            ("partial", VisibilityLevel.PARTIAL),
            ("none", VisibilityLevel.NONE),
            ("unavailable", VisibilityLevel.UNAVAILABLE),
        ],
    )
    def test_maps_by_name(self, level: str, expected: VisibilityLevel) -> None:
        columns = columns_of([box3d(visibility=DEVKIT_VISIBILITY(level))])

        assert columns.visibility.tolist() == [int(expected)]

    def test_does_not_use_the_devkit_rank(self) -> None:
        # The devkit ranks full=4 .. none=1 with None for unavailable; this package uses
        # FULL=3 .. NONE=0 with UNAVAILABLE=-1. Every level is off by one.
        columns = columns_of([box3d(visibility=DEVKIT_VISIBILITY("partial"))])

        assert columns.visibility.tolist() == [int(VisibilityLevel.PARTIAL)]
        assert columns.visibility.tolist() != [DEVKIT_VISIBILITY("partial").rank()]


class TestOptionalColumns:
    def test_an_inestimable_velocity_stays_nan(self) -> None:
        columns = columns_of(
            [box3d(velocity=(1.0, 0.0, 0.0)), box3d(velocity=None)], velocity="always"
        )

        # NaN is the absence of a claim; zero would be a claim that the object is still.
        assert np.isfinite(columns.velocity).all(axis=1).tolist() == [True, False]

    def test_auto_emits_when_any_row_is_estimable(self) -> None:
        columns = columns_of([box3d(velocity=None), box3d(velocity=(1.0, 0.0, 0.0))])

        assert columns.velocity is not None

    def test_auto_omits_when_no_row_is_estimable(self) -> None:
        assert columns_of([box3d(velocity=None)]).velocity is None

    def test_never_omits_an_estimable_column(self) -> None:
        columns = columns_of([box3d(velocity=(1.0, 0.0, 0.0))], velocity="never")

        assert columns.velocity is None

    def test_always_emits_an_empty_column_for_an_empty_batch(self) -> None:
        # A zero-row frame must still carry the scene's columns, or concatenating the
        # scene fails on a mismatched column set.
        columns = columns_of([], velocity="always", num_points="always", visibility="always")

        assert columns.velocity.shape == (0, 3)
        assert columns.num_points.shape == (0,)
        assert columns.visibility.shape == (0,)


class TestTrajectories:
    def test_shape_fits_the_widest_future_in_the_scene(self) -> None:
        short = box3d(future=future_of(waypoints=[[[1, 0, 0]]], timestamps=[FRAME_TIME_US + 1]))
        long = box3d(
            future=future_of(
                waypoints=[[[1, 0, 0], [2, 0, 0]]],
                timestamps=[FRAME_TIME_US + 1, FRAME_TIME_US + 2],
            ),
        )

        assert trajectory_shape_of([[short], [long]]) == (1, 2)

    def test_shape_falls_back_when_nothing_has_a_future(self) -> None:
        # A zero-length mode or timestep axis is rejected outright, so a scene with no
        # futures still has to produce a well-formed, fully masked batch.
        assert trajectory_shape_of([[box3d()]]) == (1, 1)

    def test_absolute_microseconds_become_relative_nanoseconds(self) -> None:
        columns = columns_of(
            [box3d(future=future_of(waypoints=[[[1, 0, 0]]], timestamps=[FRAME_TIME_US + 500]))],
            trajectory=(1, 1),
        )

        assert columns.trajectory.time_offset.tolist() == [[500_000]]

    def test_padding_is_finite_and_masked(self) -> None:
        columns = columns_of(
            [
                box3d(
                    position=(7.0, 8.0, 9.0),
                    future=future_of(waypoints=[[[1, 0, 0]]], timestamps=[FRAME_TIME_US + 1]),
                ),
                box3d(position=(7.0, 8.0, 9.0)),
            ],
            trajectory=(1, 2),
        )
        trajectory = columns.trajectory

        assert np.isfinite(trajectory.waypoints).all()
        assert trajectory.mode_valid.ravel().tolist() == [True, False]
        assert trajectory.timestep_valid[0].ravel().tolist() == [True, False]
        assert not trajectory.timestep_valid[1].any()

    def test_a_row_without_a_future_holds_its_own_position(self) -> None:
        columns = columns_of([box3d(position=(7.0, 8.0, 9.0))], trajectory=(1, 2))

        # Zeros would teleport a masked row to the origin, which reads as a real and badly
        # wrong prediction to anything that forgets the mask.
        assert columns.trajectory.waypoints[0, 0].tolist() == [[7.0, 8.0, 9.0]] * 2

    def test_time_offsets_stay_strictly_increasing_through_padding(self) -> None:
        columns = columns_of(
            [box3d(future=future_of(waypoints=[[[1, 0, 0]]], timestamps=[FRAME_TIME_US + 1]))],
            trajectory=(1, 3),
        )
        offsets = columns.trajectory.time_offset

        assert (offsets > 0).all()
        assert (np.diff(offsets, axis=1) > 0).all()

    def test_a_future_that_does_not_advance_is_rejected(self) -> None:
        box = box3d(
            future=future_of(
                waypoints=[[[1, 0, 0], [2, 0, 0]]],
                timestamps=[FRAME_TIME_US + 2, FRAME_TIME_US + 1],
            ),
        )

        with pytest.raises(ValueError, match="do not increase after the frame"):
            columns_of([box], trajectory=(1, 2))

    def test_projecting_to_predictions_without_a_trajectory_says_so(self) -> None:
        with pytest.raises(ValueError, match="carries no trajectory"):
            columns_of([box3d()]).as_predictions()


class TestLabels:
    def test_unknown_categories_are_rejected_by_default(self) -> None:
        with pytest.raises(KeyError, match="Categories not in the label registry"):
            boxes3d_to_columns(
                [box3d("traffic_cone")],
                labels=LabelRegistry.from_names(["car"]),
                instances=InstanceRegistry(),
                base_time_ns=0,
            )

    def test_unknown_categories_can_be_kept_as_unknown(self) -> None:
        columns = boxes3d_to_columns(
            [box3d("car"), box3d("traffic_cone")],
            labels=LabelRegistry.from_names(["car"]),
            instances=InstanceRegistry(),
            base_time_ns=0,
            unknown_labels="unknown",
        )

        assert columns.class_id.tolist() == [0, -1]

    def test_unknown_categories_can_be_dropped(self) -> None:
        columns = boxes3d_to_columns(
            [box3d("car"), box3d("traffic_cone")],
            labels=LabelRegistry.from_names(["car"]),
            instances=InstanceRegistry(),
            base_time_ns=0,
            unknown_labels="drop",
        )

        assert len(columns) == 1
        assert columns.kept.tolist() == [0]


class TestInstances:
    def test_identities_are_namespaced(self) -> None:
        instances = InstanceRegistry()
        columns_of([box3d(uuid="abc")], instances=instances, instance_namespace="gt")

        assert "gt/abc" in instances
        assert "abc" not in instances

    def test_two_sources_do_not_collide_on_one_id(self) -> None:
        instances = InstanceRegistry()
        first = columns_of([box3d(uuid="abc")], instances=instances, instance_namespace="gt")
        second = columns_of([box3d(uuid="abc")], instances=instances, instance_namespace="est")

        assert first.instance_id.tolist() != second.instance_id.tolist()

    def test_the_same_identity_is_stable(self) -> None:
        instances = InstanceRegistry()
        first = columns_of([box3d(uuid="abc")], instances=instances, instance_namespace="gt")
        second = columns_of([box3d(uuid="abc")], instances=instances, instance_namespace="gt")

        assert first.instance_id.tolist() == second.instance_id.tolist()


class TestProjections:
    def test_the_archetypes_share_their_columns(self) -> None:
        columns = columns_of([box3d()])
        detections = columns.as_detections().to_chunk("/objects", at=None, is_static=True)
        trackings = columns.as_trackings().to_chunk("/objects", at=None, is_static=True)

        # One extraction serves every 3D archetype because they are a nested superset
        # chain over the same rows, and a descriptor is identified by its name alone.
        assert set(detections.columns) < set(trackings.columns)
        for descriptor, column in detections.columns.items():
            assert np.array_equal(column.values, trackings.columns[descriptor].values)

    def test_a_tracking_satisfies_a_detection_system(self) -> None:
        columns = columns_of([box3d()])

        assert columns.as_trackings().has(*Detections3D.required_descriptors())

    def test_an_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown 3D archetype kind"):
            columns_of([box3d()]).as_archetype("segmentations")  # type: ignore[arg-type]


class TestEmptyBatch:
    def test_every_column_keeps_its_row_shape(self) -> None:
        columns = columns_of([], trajectory=(2, 3))

        assert len(columns) == 0
        assert columns.position.shape == (0, 3)
        assert columns.quaternion.shape == (0, 4)
        assert columns.trajectory.waypoints.shape == (0, 2, 3, 3)

    def test_it_projects_to_a_zero_row_archetype(self) -> None:
        assert len(columns_of([]).as_archetype("trackings")) == 0
        assert isinstance(columns_of([]).as_trackings(), Trackings3D)
