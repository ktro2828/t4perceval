"""Vectorized box geometry for the matching systems.

Everything here works on whole columns and on *pairs* of columns: a matcher needs an
``(N, M)`` score for every estimation against every ground truth, so the pairwise helpers
build that matrix directly rather than being called per pair from Python.

Boxes follow the dataset convention: :class:`~t4perceval.component.BatchSize3D` is
``(width, length, height)`` with ``x`` forward, ``y`` left and ``z`` up in the box frame,
and :class:`~t4perceval.component.BatchQuaternion` is ``xyzw``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from t4perceval.typing import ArrayLike, NDArrayF64, NDArrayI32, NDArrayI64

__all__ = (
    "bev_area",
    "bev_corners",
    "canonical_bev_corners",
    "pairwise_bev_intersection_area",
    "pairwise_bev_iou",
    "pairwise_height_intersection",
    "pairwise_plane_distance",
    "pairwise_roi_iou",
    "pairwise_volume_iou",
    "volume",
)

#: Unit footprint corners in the box frame, as multiples of half length and half width.
#: Ordered front-left, front-right, rear-right, rear-left.
_UNIT_CORNERS = np.array(
    [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]],
    dtype=np.float64,
)


def bev_corners(
    position: ArrayLike,
    quaternion: ArrayLike,
    size: ArrayLike,
) -> NDArrayF64:
    """Return the footprint corners of each box, with shape ``(N, 4, 2)``.

    The corners are rotated by the box's full rotation and projected onto the xy plane.
    For the yaw-only rotations perception boxes carry, that is exactly the footprint.

    Args:
        position: Box centres, ``(N, 3)``.
        quaternion: Box rotations in ``xyzw`` order, ``(N, 4)``.
        size: Box sizes as ``(width, length, height)``, ``(N, 3)``.
    """
    position = np.asarray(position, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)

    if len(position) == 0:
        return np.empty((0, 4, 2), dtype=np.float64)

    half = 0.5 * np.stack((size[:, 1], size[:, 0]), axis=1)  # (length, width) / 2
    local = _UNIT_CORNERS[None, :, :] * half[:, None, :]  # (N, 4, 2)
    local_3d = np.concatenate((local, np.zeros((len(local), 4, 1))), axis=2)

    matrices = Rotation.from_quat(quaternion).as_matrix()  # (N, 3, 3)
    rotated = np.einsum("nij,nkj->nki", matrices, local_3d)
    return rotated[:, :, :2] + position[:, None, :2]


def bev_area(size: ArrayLike) -> NDArrayF64:
    """Return the footprint area of each box, with shape ``(N,)``."""
    size = np.asarray(size, dtype=np.float64)
    return size[:, 0] * size[:, 1]


def volume(size: ArrayLike) -> NDArrayF64:
    """Return the volume of each box, with shape ``(N,)``."""
    size = np.asarray(size, dtype=np.float64)
    return size[:, 0] * size[:, 1] * size[:, 2]


def _as_polygons(corners: NDArrayF64) -> np.ndarray:
    """Turn ``(N, 4, 2)`` corners into an ``(N,)`` array of shapely polygons."""
    if len(corners) == 0:
        return np.empty(0, dtype=object)
    closed = np.concatenate((corners, corners[:, :1, :]), axis=1)
    return shapely.polygons(closed)


def pairwise_bev_intersection_area(
    est_corners: NDArrayF64,
    gt_corners: NDArrayF64,
) -> NDArrayF64:
    """Return the footprint intersection area of every pair, with shape ``(N, M)``.

    The footprints are rotated rectangles, so this is a polygon clip rather than an
    interval overlap. Shapely does it elementwise over whole arrays, which keeps the
    pairwise matrix out of a Python loop.
    """
    if len(est_corners) == 0 or len(gt_corners) == 0:
        return np.empty((len(est_corners), len(gt_corners)), dtype=np.float64)

    est_polygons = _as_polygons(est_corners)
    gt_polygons = _as_polygons(gt_corners)
    intersections = shapely.intersection(est_polygons[:, None], gt_polygons[None, :])
    return np.asarray(shapely.area(intersections), dtype=np.float64)


def pairwise_height_intersection(
    est_position: ArrayLike,
    est_size: ArrayLike,
    gt_position: ArrayLike,
    gt_size: ArrayLike,
) -> NDArrayF64:
    """Return the vertical overlap of every pair, with shape ``(N, M)``."""
    est_position = np.asarray(est_position, dtype=np.float64)
    gt_position = np.asarray(gt_position, dtype=np.float64)
    est_half = 0.5 * np.asarray(est_size, dtype=np.float64)[:, 2]
    gt_half = 0.5 * np.asarray(gt_size, dtype=np.float64)[:, 2]

    est_low = (est_position[:, 2] - est_half)[:, None]
    est_high = (est_position[:, 2] + est_half)[:, None]
    gt_low = (gt_position[:, 2] - gt_half)[None, :]
    gt_high = (gt_position[:, 2] + gt_half)[None, :]

    return np.maximum(0.0, np.minimum(est_high, gt_high) - np.maximum(est_low, gt_low))


def _iou(intersection: NDArrayF64, est_measure: NDArrayF64, gt_measure: NDArrayF64) -> NDArrayF64:
    union = est_measure[:, None] + gt_measure[None, :] - intersection
    # A pair of degenerate boxes has no union; calling that IoU 0 keeps it unmatched
    # rather than propagating a nan through the assignment.
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def pairwise_bev_iou(
    est_position: ArrayLike,
    est_quaternion: ArrayLike,
    est_size: ArrayLike,
    gt_position: ArrayLike,
    gt_quaternion: ArrayLike,
    gt_size: ArrayLike,
) -> NDArrayF64:
    """Return the bird's-eye-view IoU of every pair, with shape ``(N, M)``."""
    est_corners = bev_corners(est_position, est_quaternion, est_size)
    gt_corners = bev_corners(gt_position, gt_quaternion, gt_size)
    intersection = pairwise_bev_intersection_area(est_corners, gt_corners)
    return _iou(intersection, bev_area(est_size), bev_area(gt_size))


def pairwise_volume_iou(
    est_position: ArrayLike,
    est_quaternion: ArrayLike,
    est_size: ArrayLike,
    gt_position: ArrayLike,
    gt_quaternion: ArrayLike,
    gt_size: ArrayLike,
) -> NDArrayF64:
    """Return the 3D IoU of every pair, with shape ``(N, M)``.

    The intersection is the footprint overlap times the vertical overlap, which is exact
    while both boxes are upright.
    """
    est_corners = bev_corners(est_position, est_quaternion, est_size)
    gt_corners = bev_corners(gt_position, gt_quaternion, gt_size)
    area = pairwise_bev_intersection_area(est_corners, gt_corners)
    height = pairwise_height_intersection(est_position, est_size, gt_position, gt_size)
    return _iou(area * height, volume(est_size), volume(gt_size))


def pairwise_roi_iou(est_roi: NDArrayI32 | ArrayLike, gt_roi: NDArrayI32 | ArrayLike) -> NDArrayF64:
    """Return the image-plane IoU of every ROI pair, with shape ``(N, M)``.

    ROIs are axis-aligned ``(x_min, y_min, height, width)``, so this is an interval
    overlap in each axis and needs no polygon clipping.
    """
    est_roi = np.asarray(est_roi, dtype=np.float64)
    gt_roi = np.asarray(gt_roi, dtype=np.float64)

    if len(est_roi) == 0 or len(gt_roi) == 0:
        return np.empty((len(est_roi), len(gt_roi)), dtype=np.float64)

    est_x0, est_y0 = est_roi[:, 0][:, None], est_roi[:, 1][:, None]
    est_x1 = est_x0 + est_roi[:, 3][:, None]
    est_y1 = est_y0 + est_roi[:, 2][:, None]

    gt_x0, gt_y0 = gt_roi[:, 0][None, :], gt_roi[:, 1][None, :]
    gt_x1 = gt_x0 + gt_roi[:, 3][None, :]
    gt_y1 = gt_y0 + gt_roi[:, 2][None, :]

    overlap_x = np.maximum(0.0, np.minimum(est_x1, gt_x1) - np.maximum(est_x0, gt_x0))
    overlap_y = np.maximum(0.0, np.minimum(est_y1, gt_y1) - np.maximum(est_y0, gt_y0))

    est_area = est_roi[:, 2] * est_roi[:, 3]
    gt_area = gt_roi[:, 2] * gt_roi[:, 3]
    return _iou(overlap_x * overlap_y, est_area, gt_area)


def canonical_bev_corners(corners: NDArrayF64) -> NDArrayF64:
    """Reorder each box's corners counter-clockwise about its own centroid.

    Footprint corner order depends on how a box was constructed. Putting every box into
    one canonical cycle is what lets two boxes' corners be compared by index.
    """
    if len(corners) == 0:
        return corners

    centroid = corners.mean(axis=1, keepdims=True)
    offset = corners - centroid
    angles = np.arctan2(offset[:, :, 1], offset[:, :, 0])
    order = np.argsort(angles, axis=1)
    return np.take_along_axis(corners, order[:, :, None], axis=1)


def _nearest_plane_order(gt_corners: NDArrayF64) -> NDArrayI64:
    """Return, per ground-truth box, its corners ordered by distance from the origin."""
    distance = np.linalg.norm(gt_corners, axis=2)
    return np.argsort(distance, axis=1)


def pairwise_plane_distance(
    est_position: ArrayLike,
    est_quaternion: ArrayLike,
    est_size: ArrayLike,
    gt_position: ArrayLike,
    gt_quaternion: ArrayLike,
    gt_size: ArrayLike,
) -> NDArrayF64:
    """Return the plane distance of every pair, with shape ``(N, M)``.

    The plane distance compares the *nearest face* of the two boxes: it is the root mean
    square of the gaps between the two boxes' left and right corners on the face of the
    ground-truth box closest to the origin. Two boxes can therefore agree closely on the
    side the sensor sees while disagreeing about the far side, which is what makes this
    metric useful for a perception system that only observes the near face.

    Positions must be expressed in the frame the distance from the origin is measured in
    -- normally ``base_link``, which puts the ego at the origin.
    """
    est_corners = canonical_bev_corners(bev_corners(est_position, est_quaternion, est_size))
    gt_corners = canonical_bev_corners(bev_corners(gt_position, gt_quaternion, gt_size))

    num_est, num_gt = len(est_corners), len(gt_corners)
    if num_est == 0 or num_gt == 0:
        return np.empty((num_est, num_gt), dtype=np.float64)

    # Both cycles are canonical but may start at a different corner, so the alignment is
    # the rotation of the estimation's cycle that sits closest to the ground truth's.
    rolled = np.stack([np.roll(est_corners, shift, axis=1) for shift in range(4)])
    gap = np.linalg.norm(rolled[:, :, None, :, :] - gt_corners[None, None, :, :, :], axis=-1)
    best_shift = np.argmin(gap.sum(axis=-1), axis=0)  # (N, M)

    # `broadcast_to` only makes a view, so adding the ground-truth axis costs nothing
    # until `take_along_axis` picks the one shift each pair needs.
    candidates = np.broadcast_to(rolled[:, :, None, :, :], (4, num_est, num_gt, 4, 2))
    aligned = np.take_along_axis(
        candidates,
        np.broadcast_to(best_shift[None, :, :, None, None], (1, num_est, num_gt, 4, 2)),
        axis=0,
    )[0]  # (N, M, 4, 2)

    order = _nearest_plane_order(gt_corners)[:, :2]  # (M, 2)
    gt_face = np.take_along_axis(gt_corners, order[:, :, None], axis=1)  # (M, 2, 2)
    est_face = np.take_along_axis(
        aligned,
        np.broadcast_to(order[None, :, :, None], (num_est, num_gt, 2, 2)),
        axis=2,
    )  # (N, M, 2, 2)

    # Keep the two corners in a consistent left/right order, decided by the ground truth,
    # so the two gaps are measured between corresponding corners.
    cross = gt_face[:, 0, 0] * gt_face[:, 1, 1] - gt_face[:, 0, 1] * gt_face[:, 1, 0]
    swap = np.round(cross, 10) >= 0.0  # (M,)
    flip = np.where(swap[:, None], np.array([1, 0]), np.array([0, 1]))  # (M, 2)

    gt_face = np.take_along_axis(gt_face, flip[:, :, None], axis=1)
    est_face = np.take_along_axis(
        est_face,
        np.broadcast_to(flip[None, :, :, None], (num_est, num_gt, 2, 2)),
        axis=2,
    )

    gaps = np.linalg.norm(est_face - gt_face[None, :, :, :], axis=-1)  # (N, M, 2)
    return np.sqrt(0.5 * np.sum(gaps**2, axis=-1))
