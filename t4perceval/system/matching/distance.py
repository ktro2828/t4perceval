"""Matchers that score a pair by a distance: between centres, or between facing planes."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define

from t4perceval import geometry
from t4perceval.descriptors import CLASS_ID, POSITION, QUATERNION, SIZE
from t4perceval.system.matching.base import BOX_3D, MatchingSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayF64


@define(slots=True)
class CenterDistanceMatchingSystem(MatchingSystem):
    """Match by the 3D distance between box centres."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION, CLASS_ID)
    MATCHING_NAME: ClassVar[str] = "center_distance"
    DEFAULT_THRESHOLD: ClassVar[float] = 1.0

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_center_distance(
            est_view.component(POSITION).values,
            gt_view.component(POSITION).values,
        )


@define(slots=True)
class CenterDistanceBEVMatchingSystem(MatchingSystem):
    """Match by the distance between box centres in the xy plane.

    Kept apart from :class:`CenterDistanceMatchingSystem` because a matching mode names
    the entity a metric later reads, so the two must not share a target. This is the mode
    the original package's ``center_distance_bev_thresholds`` configured.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION, CLASS_ID)
    MATCHING_NAME: ClassVar[str] = "center_distance_bev"
    DEFAULT_THRESHOLD: ClassVar[float] = 1.0

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_bev_center_distance(
            est_view.component(POSITION).values,
            gt_view.component(POSITION).values,
        )


@define(slots=True)
class PlaneDistanceMatchingSystem(MatchingSystem):
    """Match by the distance between the boxes' nearest faces.

    Two boxes can agree closely on the face the sensor observes while disagreeing about
    the far side, which is why this mode exists alongside centre distance: it scores what
    the perception system could actually see. See
    :func:`t4perceval.geometry.pairwise_plane_distance`.

    Positions must be expressed in the frame the distance from the origin is measured in
    -- normally ``base_link``, which puts the ego at the origin.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = BOX_3D
    MATCHING_NAME: ClassVar[str] = "plane_distance"
    DEFAULT_THRESHOLD: ClassVar[float] = 2.0

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        return geometry.pairwise_plane_distance(
            est_view.component(POSITION).values,
            est_view.component(QUATERNION).values,
            est_view.component(SIZE).values,
            gt_view.component(POSITION).values,
            gt_view.component(QUATERNION).values,
            gt_view.component(SIZE).values,
        )
