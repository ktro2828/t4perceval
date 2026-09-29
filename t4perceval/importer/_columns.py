"""Column scaffolding shared by every importer's converter.

A converter turns one source's objects into the raw arrays an archetype is built from.
What differs per source is *where* a value is read from -- a ``Box3D`` attribute, a
decoded message field -- and which conversions are not identity. What does not differ is
the array plumbing around that: stacking per-object values into a column, padding a
trajectory to a fixed shape, normalising quaternions, namespacing identities. That lives
here, so that a fix to one of them cannot leave the other source silently behind.

Nothing here imports an optional dependency; the helpers take arrays and plain values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from attrs import define

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.label import InstanceRegistry
    from t4perceval.typing import NDArrayBool, NDArrayF64, NDArrayI64

__all__ = (
    "NAN3",
    "Emit",
    "TrajectoryColumns",
    "column",
    "encode_instance_ids",
    "normalize_quaternions",
    "padded_trajectory",
    "resolve_emit",
    "select_kept",
    "wxyz_to_xyzw",
)

Emit: TypeAlias = Literal["auto", "always", "never"]
"""Whether an optional column is written.

``"auto"`` decides from the batch in hand. Callers importing a whole source must resolve
it **source-wide** and pass ``"always"`` or ``"never"``: ``concat_chunks`` rejects chunks
whose column sets differ, so a column present on one frame and absent on the next makes
``Store.range()`` raise over that source. :func:`resolve_emit` does the resolution.
"""

#: A missing 3-vector. NaN rather than zero: zero would be a claim, NaN is the absence
#: of one.
NAN3 = np.full(3, np.nan, dtype=np.float64)


def column(
    values: list[Any],
    count: int,
    row_shape: tuple[int, ...],
    dtype: Any,
) -> Any:
    """Stack per-object values into a column.

    An empty batch is allocated rather than inferred: ``np.asarray([])`` collapses to
    ``(0,)``, which fails the component's per-row shape check.
    """
    if count == 0:
        return np.empty((0, *row_shape), dtype=dtype)
    return np.asarray(values, dtype=dtype).reshape(count, *row_shape)


def select_kept(items: Sequence[Any], keep: NDArrayBool) -> tuple[list[Any], NDArrayI64]:
    """Return the items whose mask entry is set, and their indices.

    The indices are what a caller records as provenance: ``keep`` is indexed against
    the source's own object list, so they trace a row straight back to it.
    """
    kept = np.flatnonzero(keep).astype(np.int64, copy=False)
    return [items[index] for index in kept], kept


def resolve_emit(setting: Emit, present: bool) -> Emit:
    """Settle an ``"auto"`` emit setting for a whole source.

    ``"always"`` and ``"never"`` are returned as they are; ``"auto"`` becomes
    ``"always"`` when the source has the data and ``"never"`` otherwise.
    """
    if setting != "auto":
        return setting
    return "always" if present else "never"


def wxyz_to_xyzw(wxyz: NDArrayF64) -> NDArrayF64:
    """Reorder quaternions from ``wxyz`` to ``xyzw``, for one or a column of them.

    Both are four floats, so taking them verbatim yields a plausible rotation rather than
    an error -- a rear camera would read as unrotated instead of turned through 180
    degrees.
    """
    return wxyz[..., (1, 2, 3, 0)]


def normalize_quaternions(xyzw: NDArrayF64, *, what: str) -> NDArrayF64:
    """Return unit quaternions, rejecting a zero row.

    Args:
        xyzw: ``(N, 4)`` quaternions.
        what: What one row stands for in the error message, e.g. ``"Box"``.
    """
    if len(xyzw) == 0:
        return xyzw
    norms = np.linalg.norm(xyzw, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"{what} {int(np.argmin(norms))} has a zero quaternion")
    return xyzw / norms


def encode_instance_ids(
    instances: InstanceRegistry,
    names: Sequence[str],
    *,
    namespace: str,
) -> NDArrayI64:
    """Intern object identities, namespaced so sources cannot collide on one integer."""
    if namespace:
        names = [f"{namespace}/{name}" for name in names]
    return instances.encode(names)


@define(frozen=True, slots=True)
class TrajectoryColumns:
    """Dense trajectory columns with a fixed mode and timestep count."""

    waypoints: NDArrayF64
    """``(N, M, T, 3)``, always finite -- padding holds the last real position."""

    mode_confidence: NDArrayF64
    """``(N, M)`` in ``[0, 1]``."""

    mode_valid: NDArrayBool
    """``(N, M)``; ``False`` for a padded mode."""

    timestep_valid: NDArrayBool
    """``(N, M, T)``; ``False`` for a padded timestep."""

    time_offset: NDArrayI64
    """``(N, T)`` nanoseconds from the frame, non-negative and strictly increasing."""


def padded_trajectory(
    positions: NDArrayF64,
    *,
    num_modes: int,
    num_timesteps: int,
) -> TrajectoryColumns:
    """Return fully padded, fully masked trajectory columns for ``positions``.

    Every row is well formed before a single real waypoint is written, so a converter
    fills in only the rows its source has a future for and prediction rows still line up
    one-to-one with the tracking rows of the same frame. The arrays are writable; the
    caller fills them in place.
    """
    count = len(positions)
    shape = (count, num_modes, num_timesteps)

    # A row with no future holds station at its own centre. Zeros would teleport it to the
    # origin, which reads as a real -- and badly wrong -- prediction to anything that
    # forgets the mask. NaN is not an option: waypoints must be finite.
    waypoints = np.repeat(
        positions[:, None, None, :],
        num_modes * num_timesteps,
        axis=1,
    ).reshape(*shape, 3)

    mode_confidence = np.zeros((count, num_modes), dtype=np.float64)
    mode_valid = np.zeros((count, num_modes), dtype=np.bool_)
    timestep_valid = np.zeros(shape, dtype=np.bool_)

    # The default axis is already non-negative and strictly increasing, which is what the
    # time-offset column requires even of rows that carry no real future.
    time_offset = np.tile(np.arange(1, num_timesteps + 1, dtype=np.int64), (count, 1))

    return TrajectoryColumns(waypoints, mode_confidence, mode_valid, timestep_valid, time_offset)
