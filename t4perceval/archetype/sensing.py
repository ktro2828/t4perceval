"""The per-box outcome of the sensing evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from t4perceval.archetype._fields import component_field
from t4perceval.component import BatchNumPoints, BatchPosition3D, BatchSensingStatus, SensingStatus
from t4perceval.core.archetype import Archetype
from t4perceval.descriptors import NEAREST_POINT, NUM_POINTS, SENSING_STATUS

if TYPE_CHECKING:
    from typing_extensions import Self

    from t4perceval.typing import NDArrayBool

__all__ = ("SensingResults",)


@define(frozen=True, slots=True)
class SensingResults(Archetype):
    """What the sensor saw of each ground-truth box.

    One row per ground-truth box, **aligned with the objects entity** the system read: the
    rows share its partitioning, so the result is joined to its box by row index, as a mask
    is. That replaces ``DynamicObjectWithSensingResult``, which held the object and the
    inside points themselves and could not be stored.

    :attr:`num_points` is how many points of the evaluated cloud fell inside the box after
    the distance-dependent footprint scale; it uses the same descriptor as the annotation's
    point count, so :class:`~t4perceval.system.FilterByNumPointsSystem` works on this entity
    unchanged. :attr:`nearest_point` is the inside point closest to the ego, ``NaN`` when
    there is none. :attr:`sensing_status` is the verdict, a :class:`SensingStatus`.
    """

    num_points = component_field(NUM_POINTS, BatchNumPoints)
    sensing_status = component_field(SENSING_STATUS, BatchSensingStatus)
    nearest_point = component_field(NEAREST_POINT, BatchPosition3D)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        known = {int(status) for status in SensingStatus}
        unknown = set(np.unique(self.sensing_status.values).tolist()) - known
        if unknown:
            raise ValueError(f"sensing_status contains unknown values: {sorted(unknown)}")
        counts = self.num_points.values
        if (counts < 0).any():
            raise ValueError("num_points must be non-negative")
        missing = np.isnan(self.nearest_point.values).any(axis=1)
        if not np.array_equal(missing, counts == 0):
            raise ValueError("nearest_point must be NaN exactly where num_points is 0")

    @classmethod
    def empty(cls) -> Self:
        """Return a result with no rows."""
        return cls(
            num_points=np.empty(0, dtype=np.int32),
            sensing_status=np.empty(0, dtype=np.int8),
            nearest_point=np.empty((0, 3), dtype=np.float64),
        )

    def count(self, status: SensingStatus) -> int:
        """Return how many rows carry ``status``."""
        return int(np.count_nonzero(self.sensing_status.values == int(status)))

    @property
    def num_detected(self) -> int:
        return self.count(SensingStatus.DETECTED)

    @property
    def num_failed(self) -> int:
        return self.count(SensingStatus.FAILED)

    @property
    def num_warned(self) -> int:
        return self.count(SensingStatus.WARNING)

    @property
    def is_detected(self) -> NDArrayBool:
        """Return which rows are :attr:`SensingStatus.DETECTED`."""
        return self.sensing_status.values == int(SensingStatus.DETECTED)
