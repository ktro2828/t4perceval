from __future__ import annotations

import numpy as np
from attrs import define

from t4perceval.core.component import ANY, ColumnarComponent

__all__ = (
    "BatchModeConfidence",
    "BatchModeValid",
    "BatchTimeOffset",
    "BatchTimestepValid",
    "BatchWaypoints3D",
)


@define(frozen=True, slots=True)
class BatchWaypoints3D(ColumnarComponent):
    """Dense future waypoints with shape ``(N, M, T, 3)``.

    ``N`` objects, ``M`` trajectory modes, ``T`` timesteps. ``M`` and ``T`` are fixed
    within one chunk; shorter trajectories are padded and masked by
    :class:`BatchModeValid` / :class:`BatchTimestepValid` rather than stored ragged.

    Padding must still be finite -- an invalid timestep is marked in the validity mask,
    never encoded as ``NaN``.
    """

    SHAPE = (ANY, ANY, 3)
    REQUIRE_FINITE = True

    @property
    def num_modes(self) -> int:
        return int(self.values.shape[1])

    @property
    def num_timesteps(self) -> int:
        return int(self.values.shape[2])


@define(frozen=True, slots=True)
class BatchModeConfidence(ColumnarComponent):
    """Per-mode confidence with shape ``(N, M)``, constrained to ``[0, 1]``.

    Interpreted as the posterior probability of each mode. The sum over modes is *not*
    required to be 1: models frequently emit unnormalized scores, and normalizing here
    would silently change the metric.
    """

    SHAPE = (ANY,)
    VALUE_RANGE = (0.0, 1.0)

    @property
    def num_modes(self) -> int:
        return int(self.values.shape[1])


@define(frozen=True, slots=True)
class BatchModeValid(ColumnarComponent):
    """Which modes carry real data, with shape ``(N, M)``."""

    SHAPE = (ANY,)
    DTYPE = np.bool_


@define(frozen=True, slots=True)
class BatchTimestepValid(ColumnarComponent):
    """Which timesteps of which mode carry real data, with shape ``(N, M, T)``."""

    SHAPE = (ANY, ANY)
    DTYPE = np.bool_


@define(frozen=True, slots=True)
class BatchTimeOffset(ColumnarComponent):
    """Time offsets of the waypoint timesteps in nanoseconds, with shape ``(N, T)``.

    Offsets are relative to the frame the trajectory was predicted at. Because every row
    of a chunk shares the same time axis, this column is normally logged once as *static*
    data with ``N == 1``.
    """

    SHAPE = (ANY,)
    DTYPE = np.int64

    def __attrs_post_init__(self) -> None:
        if self.values.size == 0:
            return
        if np.any(self.values < 0):
            raise ValueError("BatchTimeOffset must be non-negative")
        if self.values.shape[1] > 1 and np.any(np.diff(self.values, axis=1) <= 0):
            raise ValueError("BatchTimeOffset must be strictly increasing")

    @property
    def num_timesteps(self) -> int:
        return int(self.values.shape[1])
