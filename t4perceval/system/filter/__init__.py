"""Filter systems.

A filter emits a boolean :data:`~t4perceval.descriptors.MASK` column instead of dropping
rows. The rows stay addressable, so a later stage can ask *why* an object was excluded,
and the filter itself becomes reviewable data rather than a side effect.

Every filter is only its predicate: :class:`MaskSystem` implements the query, the
validation, the empty-frame case and the chunk construction once.

Two conventions hold across the whole family:

* **Bounds are inclusive on both ends**, so a filter constructed with its default
  parameters passes every row. The original package compared strictly (``score >
  threshold``, ``abs(x) < max_x_position``), which makes ``min_distance=0.0`` reject an
  object at the origin; the boundary itself is measure-zero in float arithmetic, and a
  default that is a guaranteed no-op is worth more than matching it exactly.
* **A missing optional column is an error, not a pass.** ``t4_devkit`` lets a box with no
  velocity through its speed filter; here, filtering on a component the entity does not
  carry means the pipeline is wired wrong, and :func:`~t4perceval.system.base.require`
  says so.
"""

from __future__ import annotations

from t4perceval.system.filter.base import MaskSystem
from t4perceval.system.filter.identity import (
    FilterByInstanceSystem,
    FilterByLabelSystem,
    resolve_class_ids,
)
from t4perceval.system.filter.mask import ApplyMaskSystem, CombineMasksSystem, masked_view
from t4perceval.system.filter.point import (
    FilterPointsByCoverageSystem,
    FilterPointsByDistanceSystem,
    FilterPointsByMapSystem,
    FilterPointsByRegionSystem,
)
from t4perceval.system.filter.quality import (
    FilterByConfidenceSystem,
    FilterByNumPointsSystem,
    FilterBySpeedSystem,
    FilterByVisibilitySystem,
)
from t4perceval.system.filter.position import (
    FilterByDistanceSystem,
    FilterByMapSystem,
    FilterByRegionSystem,
)

__all__ = (
    "ApplyMaskSystem",
    "CombineMasksSystem",
    "FilterByConfidenceSystem",
    "FilterByDistanceSystem",
    "FilterByInstanceSystem",
    "FilterByLabelSystem",
    "FilterByNumPointsSystem",
    "FilterByMapSystem",
    "FilterByRegionSystem",
    "FilterBySpeedSystem",
    "FilterByVisibilitySystem",
    "FilterPointsByCoverageSystem",
    "FilterPointsByDistanceSystem",
    "FilterPointsByMapSystem",
    "FilterPointsByRegionSystem",
    "MaskSystem",
    "masked_view",
    "resolve_class_ids",
)
