"""The verdict column of the sensing evaluation."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from attrs import define

from t4perceval.core.component import Component

__all__ = ("BatchSensingStatus", "SensingStatus")


class SensingStatus(IntEnum):
    """Outcome of looking for sensor points inside one ground-truth box."""

    DETECTED = 0
    """At least the required number of points lie inside the scaled box."""
    FAILED = 1
    """Fewer points than required lie inside the scaled box."""
    WARNING = 2
    """The object is annotated as fully occluded, so the count proves nothing either way."""


@define(frozen=True, slots=True)
class BatchSensingStatus(Component):
    """DETECTED / FAILED / WARNING verdict per ground-truth box, with shape ``(N,)``.

    Values are :class:`SensingStatus` members.
    """

    DTYPE = np.int8
