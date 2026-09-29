"""Plumbing shared by every importer's scene or topic loop.

An importer decides *when and where* data is recorded: which frames are taken, which
coordinate frame the whole selection is in, what the recording says about where it came
from. Those decisions are the same whichever source is being read, so they live here
rather than once per importer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypeVar

from t4perceval.recording import RecordingMetadata

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from t4perceval.recording import SourceInfo

__all__ = (
    "import_metadata",
    "narrow",
    "package_version",
    "pin_trajectory_shape",
    "single_frame_id",
)

T = TypeVar("T")


def package_version() -> str:
    """Return the installed package version, or an empty string when unavailable."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("t4perceval")
    except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
        return ""


def import_metadata(source: SourceInfo, *, frame_id: str | None) -> RecordingMetadata:
    """Return the metadata an importer stamps on a recording.

    The label fingerprint is left empty on purpose: ``Recording.of`` stamps it from the
    registry the recording is bound to, so it cannot disagree with the data.
    """
    return RecordingMetadata(
        t4perceval_version=package_version(),
        created_at_ns=time.time_ns(),
        sources=(source,),
        frame_id=frame_id,
    )


def narrow(
    items: Sequence[T],
    chosen: slice | Sequence[int] | None,
) -> tuple[tuple[int, T], ...]:
    """Narrow a sequence, keeping each item's position in the *full* sequence.

    So ``chosen=slice(10, 20)`` yields positions 10..19, and two selections of one source
    stay directly comparable instead of both starting at zero.
    """
    indexed = tuple(enumerate(items))
    if chosen is None:
        return indexed
    if isinstance(chosen, slice):
        return indexed[chosen]
    return tuple(indexed[index] for index in chosen)


def single_frame_id(frame_ids: Iterable[str], *, what: str) -> str | None:
    """Return the one coordinate frame a selection is in, rejecting a mixture.

    A chunk carries a single ``frame_id`` and ``concat_chunks`` refuses to join chunks
    that disagree, so picking one silently would break the source-wide query later, far
    from the cause.

    Args:
        frame_ids: The frame of every object or message in the selection.
        what: What the selection is, for the error message, e.g. ``"Scene"``.
    """
    seen = set(frame_ids)
    if len(seen) > 1:
        raise ValueError(f"{what} mixes coordinate frames: {sorted(seen)}")
    return seen.pop() if seen else None


def pin_trajectory_shape(
    fitted: tuple[int, int],
    *,
    num_modes: int | None,
    num_timesteps: int | None,
) -> tuple[int, int]:
    """Return the trajectory shape to use, honouring any pinned dimension."""
    return (
        num_modes if num_modes is not None else fitted[0],
        num_timesteps if num_timesteps is not None else fitted[1],
    )
