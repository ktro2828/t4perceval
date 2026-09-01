"""Matching thresholds, uniform or per class.

The original ``evaluation_config_dict`` carried thresholds as lists aligned with
``target_labels`` -- ``center_distance_thresholds: [[1.0, 1.0, 1.0, 1.0]]`` -- which meant
the reader had to know the label order to know what a number applied to. Here a threshold
is either one number or an explicit class-keyed mapping, so it says what it applies to.

Matching systems already require :data:`~t4perceval.descriptors.CLASS_ID`, so supporting
per-class thresholds costs them no extra component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

import numpy as np
from attrs import define, field

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from typing_extensions import Self

    from t4perceval.label import LabelRegistry
    from t4perceval.typing import ArrayLike, NDArrayF64

__all__ = ("Thresholds", "ThresholdsLike")


def _as_by_class(
    value: Mapping[str | int, float] | Iterable[tuple[str | int, float]] | None,
) -> tuple[tuple[str | int, float], ...]:
    if value is None:
        return ()
    items = value.items() if hasattr(value, "items") else value
    return tuple((key, float(threshold)) for key, threshold in items)


@define(frozen=True, slots=True)
class Thresholds:
    """A matching threshold that may vary by class.

    Examples:
        >>> Thresholds.coerce(1.0).default
        1.0
        >>> per_class = Thresholds.coerce({"car": 2.0, "pedestrian": 0.5})
        >>> per_class.is_uniform
        False
    """

    default: float = field(converter=float)
    by_class: tuple[tuple[str | int, float], ...] = field(
        default=(),
        converter=_as_by_class,
        kw_only=True,
    )

    def __attrs_post_init__(self) -> None:
        for value in (self.default, *(threshold for _, threshold in self.by_class)):
            if not np.isfinite(value):
                raise ValueError(f"Threshold must be finite, got {value}")

    @classmethod
    def coerce(cls, value: ThresholdsLike, *, default: float | None = None) -> Self:
        """Accept a number, a class-keyed mapping, or an existing :class:`Thresholds`.

        Args:
            value: The threshold specification.
            default: Threshold for classes the mapping does not mention. Required when
                ``value`` is a mapping that should not fall back to "always feasible".
        """
        if isinstance(value, Thresholds):
            return value  # type: ignore[return-value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(float(value))

        by_class = _as_by_class(value)  # type: ignore[arg-type]
        if not by_class:
            raise ValueError(
                "A per-class threshold mapping must not be empty; pass a number instead",
            )
        if default is None:
            raise ValueError(
                "A per-class threshold mapping needs a default for the classes it omits; "
                "pass Thresholds(default, by_class=...) explicitly",
            )
        return cls(default, by_class=by_class)

    @property
    def is_uniform(self) -> bool:
        """Return whether the same threshold applies to every class."""
        return not self.by_class

    def resolve(
        self,
        class_ids: ArrayLike,
        labels: LabelRegistry | None = None,
    ) -> NDArrayF64:
        """Return the threshold for every row, with shape ``(len(class_ids),)``.

        Args:
            class_ids: The class-id column the thresholds apply to.
            labels: Registry used to resolve class *names*. Only needed when
                :attr:`by_class` is keyed by name.
        """
        ids = np.asarray(class_ids, dtype=np.int64)
        resolved = np.full(ids.shape, self.default, dtype=np.float64)

        names = [key for key, _ in self.by_class if isinstance(key, str)]
        if names and labels is None:
            raise ValueError(
                f"Per-class thresholds keyed by name {names} require a LabelRegistry; "
                "pass one as SystemContext(labels=...)",
            )

        for key, threshold in self.by_class:
            # An unknown name raises rather than silently leaving the default in place,
            # which would look like the per-class threshold had been applied.
            class_id = key if isinstance(key, int) else labels.class_id(key)  # type: ignore[union-attr]
            resolved[ids == class_id] = threshold

        return resolved


ThresholdsLike: TypeAlias = Union[float, "Mapping[str | int, float]", Thresholds]
