"""Filters on how well an object was observed: confidence, speed, points, visibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define, field

from t4perceval.component import VisibilityLevel
from t4perceval.descriptors import CONFIDENCE, NUM_POINTS, VELOCITY, VISIBILITY
from t4perceval.system.base import SystemContext
from t4perceval.system.filter.base import MaskSystem, check_range

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayBool


@define(slots=True)
class FilterByConfidenceSystem(MaskSystem):
    """Keep objects whose confidence is within ``[min, max]``.

    ``min_confidence`` is the original ``confidence_threshold``.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (CONFIDENCE,)
    FILTER_NAME: ClassVar[str] = "confidence"

    min_confidence: float = field(default=0.0, kw_only=True)
    max_confidence: float = field(default=1.0, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        for value, name in ((self.min_confidence, "min"), (self.max_confidence, "max")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}_confidence must be within [0, 1], got {value}")
        check_range(
            self.min_confidence,
            self.max_confidence,
            low_name="min_confidence",
            high_name="max_confidence",
        )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        confidence = view.component(CONFIDENCE).values
        return (confidence >= self.min_confidence) & (confidence <= self.max_confidence)


@define(slots=True)
class FilterBySpeedSystem(MaskSystem):
    """Keep objects whose speed -- the L2 norm of their velocity -- is within ``[min, max]``."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (VELOCITY,)
    FILTER_NAME: ClassVar[str] = "speed"

    min_speed: float = field(default=0.0, kw_only=True)
    max_speed: float = field(default=float("inf"), kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_speed < 0.0:
            raise ValueError(f"min_speed must be non-negative, got {self.min_speed}")
        check_range(self.min_speed, self.max_speed, low_name="min_speed", high_name="max_speed")

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        speed = view.component(VELOCITY).speed
        return (speed >= self.min_speed) & (speed <= self.max_speed)


@define(slots=True)
class FilterByNumPointsSystem(MaskSystem):
    """Keep objects whose box contains a number of points within ``[min, max]``.

    ``min_num_points`` is the original ``min_point_numbers``, which was a per-class list.
    Per-class thresholds are expressed here by composition -- see
    :class:`CombineMasksSystem`.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (NUM_POINTS,)
    FILTER_NAME: ClassVar[str] = "num_points"

    min_num_points: int = field(default=0, kw_only=True)
    max_num_points: int | None = field(default=None, kw_only=True)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_num_points < 0:
            raise ValueError(f"min_num_points must be non-negative, got {self.min_num_points}")
        if self.max_num_points is not None:
            check_range(
                self.min_num_points,
                self.max_num_points,
                low_name="min_num_points",
                high_name="max_num_points",
            )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        num_points = view.component(NUM_POINTS).values
        keep = num_points >= self.min_num_points
        if self.max_num_points is not None:
            keep &= num_points <= self.max_num_points
        return keep


@define(slots=True)
class FilterByVisibilitySystem(MaskSystem):
    """Keep objects at least as visible as ``min_visibility``.

    Objects annotated :attr:`VisibilityLevel.UNAVAILABLE` always pass, matching
    ``t4_devkit.filtering.FilterByVisibility``. Rejecting them instead would empty out
    every dataset that does not annotate visibility at all.
    """

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (VISIBILITY,)
    FILTER_NAME: ClassVar[str] = "visibility"

    min_visibility: VisibilityLevel = field(
        default=VisibilityLevel.NONE,
        converter=VisibilityLevel,
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.min_visibility is VisibilityLevel.UNAVAILABLE:
            raise ValueError(
                "min_visibility must be a comparable level, not UNAVAILABLE; "
                "UNAVAILABLE objects always pass this filter",
            )

    def keep(self, view: EntityView, ctx: SystemContext) -> NDArrayBool:
        del ctx
        visibility = view.component(VISIBILITY).values
        unavailable = visibility == int(VisibilityLevel.UNAVAILABLE)
        return unavailable | (visibility >= int(self.min_visibility))
