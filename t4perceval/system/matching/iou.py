"""Matchers that score a pair by intersection over union: footprints, volumes, image regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define

from t4perceval import geometry
from t4perceval.descriptors import CLASS_ID, POSITION, QUATERNION, ROI, SIZE
from t4perceval.system.matching.base import BOX_3D, MatchingSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayF64


@define(slots=True)
class IoUBEVMatchingSystem(MatchingSystem):
    """Match 3D boxes by the IoU of their footprints.

    This is the 2D IoU the original package computed for 3D tasks -- its
    ``iou_2d_thresholds`` -- measured on the rotated footprint rather than an
    axis-aligned box. For image-plane IoU of 2D detections, use
    :class:`IoURoiMatchingSystem`.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = BOX_3D
    MATCHING_NAME: ClassVar[str] = "iou_bev"
    HIGHER_IS_BETTER: ClassVar[bool] = True
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_bev_iou(
            est_view.component(POSITION).values,
            est_view.component(QUATERNION).values,
            est_view.component(SIZE).values,
            gt_view.component(POSITION).values,
            gt_view.component(QUATERNION).values,
            gt_view.component(SIZE).values,
        )


@define(slots=True)
class IoU3DMatchingSystem(MatchingSystem):
    """Match 3D boxes by the IoU of their volumes."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = BOX_3D
    MATCHING_NAME: ClassVar[str] = "iou_3d"
    HIGHER_IS_BETTER: ClassVar[bool] = True
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_volume_iou(
            est_view.component(POSITION).values,
            est_view.component(QUATERNION).values,
            est_view.component(SIZE).values,
            gt_view.component(POSITION).values,
            gt_view.component(QUATERNION).values,
            gt_view.component(SIZE).values,
        )


@define(slots=True)
class IoURoiMatchingSystem(MatchingSystem):
    """Match 2D detections by the IoU of their image-plane regions.

    This is the ``iou_2d_thresholds`` of the original package's 2D tasks. It requires
    :data:`~t4perceval.descriptors.ROI` rather than a 3D box, which is why it is a
    separate system from :class:`IoUBEVMatchingSystem` instead of one class that inspects
    which components happen to be present.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (ROI, CLASS_ID)
    MATCHING_NAME: ClassVar[str] = "iou_roi"
    HIGHER_IS_BETTER: ClassVar[bool] = True
    DEFAULT_THRESHOLD: ClassVar[float] = 0.5

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_roi_iou(
            est_view.component(ROI).values,
            gt_view.component(ROI).values,
        )
