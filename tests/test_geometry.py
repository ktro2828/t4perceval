"""Box geometry primitives, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval.geometry import (
    bev_area,
    bev_corners,
    canonical_bev_corners,
    pairwise_bev_intersection_area,
    pairwise_bev_iou,
    pairwise_height_intersection,
    pairwise_plane_distance,
    pairwise_roi_iou,
    pairwise_volume_iou,
    volume,
)

#: width=2, length=4, height=2 -> footprint 8 m^2, volume 16 m^3.
SIZE = [[2.0, 4.0, 2.0]]
ORIGIN = [[0.0, 0.0, 0.0]]


def yaw(angle: float) -> list[float]:
    """Return an ``xyzw`` quaternion for a rotation about z."""
    return [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]


NO_ROTATION = [yaw(0.0)]


class TestCorners:
    def test_uses_the_dataset_size_convention(self) -> None:
        # size is (width, length, height) with x forward, so length spans x.
        corners = bev_corners(ORIGIN, NO_ROTATION, SIZE)

        assert corners.shape == (1, 4, 2)
        assert np.ptp(corners[0][:, 0]) == pytest.approx(4.0), "length spans x"
        assert np.ptp(corners[0][:, 1]) == pytest.approx(2.0), "width spans y"

    def test_a_quarter_turn_swaps_the_extents(self) -> None:
        corners = bev_corners(ORIGIN, [yaw(np.pi / 2)], SIZE)

        assert np.ptp(corners[0][:, 0]) == pytest.approx(2.0)
        assert np.ptp(corners[0][:, 1]) == pytest.approx(4.0)

    def test_translates_with_the_centre(self) -> None:
        corners = bev_corners([[10.0, 5.0, 0.0]], NO_ROTATION, SIZE)

        np.testing.assert_allclose(corners[0].mean(axis=0), [10.0, 5.0])

    def test_reports_area_and_volume(self) -> None:
        assert bev_area(SIZE).tolist() == [8.0]
        assert volume(SIZE).tolist() == [16.0]

    def test_handles_an_empty_column(self) -> None:
        empty3 = np.empty((0, 3))
        empty4 = np.empty((0, 4))

        assert bev_corners(empty3, empty4, empty3).shape == (0, 4, 2)
        assert bev_area(empty3).shape == (0,)

    def test_canonical_order_is_one_cycle_per_box(self) -> None:
        corners = canonical_bev_corners(bev_corners(ORIGIN, NO_ROTATION, SIZE))

        centroid = corners[0].mean(axis=0)
        angles = np.arctan2(corners[0][:, 1] - centroid[1], corners[0][:, 0] - centroid[0])

        assert np.all(np.diff(angles) > 0), "counter-clockwise about the centroid"
        assert set(map(tuple, np.round(corners[0], 6))) == {
            (2.0, 1.0),
            (2.0, -1.0),
            (-2.0, -1.0),
            (-2.0, 1.0),
        }

    def test_canonical_order_is_stable_under_rotation_of_the_input(self) -> None:
        corners = bev_corners(ORIGIN, NO_ROTATION, SIZE)

        rolled = np.roll(corners, 2, axis=1)

        np.testing.assert_allclose(
            canonical_bev_corners(corners),
            canonical_bev_corners(rolled),
        )

    def test_canonical_order_passes_an_empty_column_through(self) -> None:
        assert canonical_bev_corners(np.empty((0, 4, 2))).shape == (0, 4, 2)


class TestBevIou:
    def test_a_box_against_itself_is_one(self) -> None:
        assert pairwise_bev_iou(
            ORIGIN, NO_ROTATION, SIZE, ORIGIN, NO_ROTATION, SIZE
        ) == pytest.approx(
            1.0,
        )

    def test_disjoint_boxes_are_zero(self) -> None:
        far = [[100.0, 0.0, 0.0]]

        assert pairwise_bev_iou(ORIGIN, NO_ROTATION, SIZE, far, NO_ROTATION, SIZE) == pytest.approx(
            0.0,
        )

    def test_a_half_length_shift(self) -> None:
        # 4-long boxes offset by 2: intersection 2 * 2 = 4, union 8 + 8 - 4 = 12.
        shifted = [[2.0, 0.0, 0.0]]

        iou = pairwise_bev_iou(ORIGIN, NO_ROTATION, SIZE, shifted, NO_ROTATION, SIZE)

        assert iou[0, 0] == pytest.approx(4.0 / 12.0)

    def test_a_rotated_box_needs_polygon_clipping(self) -> None:
        # An axis-aligned overlap would report the full 8 m^2; the true intersection of
        # the two rotated rectangles is the 2 x 2 square where they cross.
        rotated = [yaw(np.pi / 2)]

        area = pairwise_bev_intersection_area(
            bev_corners(ORIGIN, NO_ROTATION, SIZE),
            bev_corners(ORIGIN, rotated, SIZE),
        )

        assert area[0, 0] == pytest.approx(4.0)

    def test_ignores_the_z_axis(self) -> None:
        lifted = [[0.0, 0.0, 100.0]]

        assert pairwise_bev_iou(
            ORIGIN,
            NO_ROTATION,
            SIZE,
            lifted,
            NO_ROTATION,
            SIZE,
        ) == pytest.approx(1.0)

    def test_a_degenerate_box_is_zero_not_nan(self) -> None:
        zero = [[0.0, 0.0, 0.0]]

        iou = pairwise_bev_iou(ORIGIN, NO_ROTATION, zero, ORIGIN, NO_ROTATION, zero)

        assert iou[0, 0] == 0.0

    def test_builds_the_whole_pairwise_matrix(self) -> None:
        est = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        gt = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        iou = pairwise_bev_iou(est, [yaw(0.0)] * 3, SIZE * 3, gt, [yaw(0.0)] * 2, SIZE * 2)

        assert iou.shape == (3, 2)
        np.testing.assert_allclose(np.diag(iou[:2]), [1.0, 1.0])
        np.testing.assert_allclose(iou[2], [0.0, 0.0])

    def test_handles_empty_inputs(self) -> None:
        empty3 = np.empty((0, 3))
        empty4 = np.empty((0, 4))

        assert pairwise_bev_iou(empty3, empty4, empty3, ORIGIN, NO_ROTATION, SIZE).shape == (0, 1)
        assert pairwise_bev_iou(ORIGIN, NO_ROTATION, SIZE, empty3, empty4, empty3).shape == (1, 0)


class TestVolumeIou:
    def test_a_box_against_itself_is_one(self) -> None:
        assert pairwise_volume_iou(
            ORIGIN,
            NO_ROTATION,
            SIZE,
            ORIGIN,
            NO_ROTATION,
            SIZE,
        ) == pytest.approx(1.0)

    def test_a_vertical_shift_reduces_it(self) -> None:
        # 2-tall boxes offset by 1: height overlap 1, footprint overlap 8,
        # intersection 8, union 16 + 16 - 8 = 24.
        lifted = [[0.0, 0.0, 1.0]]

        iou = pairwise_volume_iou(ORIGIN, NO_ROTATION, SIZE, lifted, NO_ROTATION, SIZE)

        assert iou[0, 0] == pytest.approx(8.0 / 24.0)

    def test_no_vertical_overlap_is_zero(self) -> None:
        lifted = [[0.0, 0.0, 2.0]]

        assert pairwise_volume_iou(
            ORIGIN,
            NO_ROTATION,
            SIZE,
            lifted,
            NO_ROTATION,
            SIZE,
        ) == pytest.approx(0.0)

    def test_reports_the_height_overlap(self) -> None:
        assert pairwise_height_intersection(ORIGIN, SIZE, [[0.0, 0.0, 1.0]], SIZE)[0, 0] == 1.0
        assert pairwise_height_intersection(ORIGIN, SIZE, [[0.0, 0.0, 9.0]], SIZE)[0, 0] == 0.0

    def test_differs_from_bev_iou_when_heights_disagree(self) -> None:
        short = [[2.0, 4.0, 1.0]]

        bev = pairwise_bev_iou(ORIGIN, NO_ROTATION, SIZE, ORIGIN, NO_ROTATION, short)
        volumetric = pairwise_volume_iou(ORIGIN, NO_ROTATION, SIZE, ORIGIN, NO_ROTATION, short)

        assert bev[0, 0] == pytest.approx(1.0)
        assert volumetric[0, 0] == pytest.approx(8.0 / 16.0)


class TestRoiIou:
    def test_a_roi_against_itself_is_one(self) -> None:
        roi = [[0, 0, 10, 10]]

        assert pairwise_roi_iou(roi, roi) == pytest.approx(1.0)

    def test_uses_the_dataset_roi_layout(self) -> None:
        # (x_min, y_min, height, width): shifting x by 2 leaves an 8 x 10 overlap.
        iou = pairwise_roi_iou([[0, 0, 10, 10]], [[2, 0, 10, 10]])

        assert iou[0, 0] == pytest.approx(80.0 / 120.0)

    def test_height_and_width_are_not_interchangeable(self) -> None:
        # 20 tall by 10 wide against 10 tall by 20 wide: overlap 10 x 10 = 100,
        # union 200 + 200 - 100 = 300.
        iou = pairwise_roi_iou([[0, 0, 20, 10]], [[0, 0, 10, 20]])

        assert iou[0, 0] == pytest.approx(100.0 / 300.0)

    def test_disjoint_rois_are_zero(self) -> None:
        assert pairwise_roi_iou([[0, 0, 10, 10]], [[100, 0, 10, 10]]) == pytest.approx(0.0)

    def test_a_degenerate_roi_is_zero_not_nan(self) -> None:
        assert pairwise_roi_iou([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0] == 0.0

    def test_handles_empty_inputs(self) -> None:
        assert pairwise_roi_iou(np.empty((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)


class TestPlaneDistance:
    GT_POSITION = [[10.0, 0.0, 0.0]]

    def distance(
        self,
        position: list[list[float]],
        size: list[list[float]] | None = None,
        rotation: list[list[float]] | None = None,
    ) -> float:
        return float(
            pairwise_plane_distance(
                position,
                rotation or NO_ROTATION,
                size or SIZE,
                self.GT_POSITION,
                NO_ROTATION,
                SIZE,
            )[0, 0],
        )

    def test_a_box_against_itself_is_zero(self) -> None:
        assert self.distance(self.GT_POSITION) == pytest.approx(0.0)

    def test_a_longitudinal_shift_moves_the_near_face(self) -> None:
        assert self.distance([[11.0, 0.0, 0.0]]) == pytest.approx(1.0)

    def test_a_lateral_shift_moves_both_near_corners(self) -> None:
        assert self.distance([[10.0, 1.0, 0.0]]) == pytest.approx(1.0)

    def test_an_error_only_on_the_far_face_does_not_count(self) -> None:
        """This is the whole point of the metric.

        A box 2 m longer, recentred so its near face still coincides, scores 0 even though
        its centre is 1 m away -- the sensor could not have seen the disagreement.
        """
        assert self.distance([[11.0, 0.0, 0.0]], [[2.0, 6.0, 2.0]]) == pytest.approx(0.0)

        centre_distance = np.linalg.norm(np.array([11.0, 0.0, 0.0]) - np.array([10.0, 0.0, 0.0]))
        assert centre_distance == pytest.approx(1.0), "centre distance would penalise it"

    def test_a_width_error_moves_both_near_corners(self) -> None:
        # 2 m wider: each near corner moves 1 m sideways, so the RMS gap is 1.
        assert self.distance(self.GT_POSITION, [[4.0, 4.0, 2.0]]) == pytest.approx(1.0)

    def test_a_heading_error_is_penalised(self) -> None:
        assert self.distance(self.GT_POSITION, rotation=[yaw(np.pi / 2)]) == pytest.approx(
            np.sqrt(2.0),
        )

    def test_ignores_the_z_axis(self) -> None:
        assert self.distance([[10.0, 0.0, 5.0]]) == pytest.approx(0.0)

    def test_builds_the_whole_pairwise_matrix(self) -> None:
        est = np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0]])
        gt = np.array([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])

        distance = pairwise_plane_distance(
            est,
            [yaw(0.0)] * 3,
            SIZE * 3,
            gt,
            [yaw(0.0)] * 2,
            SIZE * 2,
        )

        assert distance.shape == (3, 2)
        assert distance[0, 0] == pytest.approx(0.0)
        assert distance[1, 0] == pytest.approx(1.0)
        assert distance[1, 1] == pytest.approx(1.0)

    def test_is_measured_from_the_origin(self) -> None:
        """The near face is the one closest to the ego, so the sign of x matters."""
        behind = pairwise_plane_distance(
            [[-11.0, 0.0, 0.0]],
            NO_ROTATION,
            SIZE,
            [[-10.0, 0.0, 0.0]],
            NO_ROTATION,
            SIZE,
        )

        assert behind[0, 0] == pytest.approx(1.0)

    def test_handles_empty_inputs(self) -> None:
        empty3 = np.empty((0, 3))
        empty4 = np.empty((0, 4))

        assert pairwise_plane_distance(
            empty3,
            empty4,
            empty3,
            self.GT_POSITION,
            NO_ROTATION,
            SIZE,
        ).shape == (0, 1)
        assert pairwise_plane_distance(
            self.GT_POSITION,
            NO_ROTATION,
            SIZE,
            empty3,
            empty4,
            empty3,
        ).shape == (1, 0)
