"""Pairing ground-truth and estimation frames by timestamp onto one ``FRAME`` axis.

Two recordings that were imported separately each number their own frames: a T4 scene by
sample index, a bag by message index. Both also carry ``TIMESTAMP``, and that is the only
axis they share. Nothing downstream checks the ``FRAME`` axes agree --
:class:`~t4perceval.system.matching.MatchingSystem` evaluates the *union* of both entities'
times, so a frame present on one side only scores as all-FP or all-FN, and indices that
happen to overlap numerically score as plausible garbage.

This module makes the pairing explicit and one-to-one, **driven by the reference**: every
ground-truth frame takes at most one estimation frame, the one nearest in time within a
tolerance, and the estimation recording's ``FRAME`` values are rewritten to the reference's.
Estimation frames nobody claimed leave the evaluated set. Reference frames nobody answered
stay by default and score as all-FN -- the fact that nothing was estimated there is part of
the result -- unless ``unmatched_reference="drop"`` asks for the incumbent's behaviour of not
evaluating them at all.

The incumbent (``perception_eval``) associates in the other direction: each estimation
message looks for a ground-truth frame within 75 ms and is skipped when none is there, so
the same ground-truth frame can be scored several times and, at 2 Hz ground truth against a
10 Hz stack, most messages are skipped. The 75 ms default is kept; the direction is not.

Alignment is a derivation over recordings, not a mutation: every function returns a new
:class:`~t4perceval.recording.Recording` that shares the untouched chunks, the label registry
and -- importantly for the instance-registry identity check in
:mod:`t4perceval.evaluation` -- the same :class:`~t4perceval.label.InstanceRegistry`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from attrs import cmp_using, define, evolve, field

from t4perceval.core.entity import as_entity_path
from t4perceval.core.store import Store
from t4perceval.core.timeline import FRAME, TIMESTAMP, TimeColumn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from numpy.typing import ArrayLike

    from t4perceval.core.chunk import Chunk
    from t4perceval.core.entity import EntityPath, EntityPathLike
    from t4perceval.recording import Recording
    from t4perceval.typing import NDArrayI64

__all__ = (
    "DEFAULT_TOLERANCE_NS",
    "AlignOptions",
    "Aligned",
    "FrameAlignment",
    "UnmatchedReference",
    "align_frames",
    "align_recordings",
    "drop_frames",
    "frame_times",
    "pair_times",
    "reindex_frames",
)

#: How far apart two frames may be and still be the same moment: 75 ms, the incumbent's
#: ``threshold_min_time`` of 75000 microseconds.
DEFAULT_TOLERANCE_NS: int = 75_000_000

UnmatchedReference: TypeAlias = Literal["keep", "drop"]
"""What to do with a reference frame no estimation frame was paired with.

``"keep"`` leaves it in place, where it scores as all-FN; ``"drop"`` removes it from the
reference recording, so the frame is not evaluated at all -- what the incumbent does.
"""

_TAG_PREFIX = "align."


def _as_i64(value: ArrayLike) -> NDArrayI64:
    array = np.array(value, dtype=np.int64).ravel()
    array.flags.writeable = False
    return array


# -- pairing --------------------------------------------------------------------------------


def pair_times(
    reference_ns: ArrayLike,
    query_ns: ArrayLike,
    *,
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
    offset_ns: int = 0,
) -> tuple[NDArrayI64, NDArrayI64]:
    """Pair two sets of times one-to-one, nearest first, within a tolerance.

    Args:
        reference_ns: Times of the reference frames, in any order.
        query_ns: Times of the query frames, in any order.
        tolerance_ns: Largest ``|query + offset - reference|`` accepted, inclusive.
        offset_ns: Added to every query time before comparing -- a known clock skew.

    Returns:
        ``(reference_index, query_index)``: indices into the two inputs, one pair per row,
        ordered by reference time.

    The rule is *nearest wins*: candidate pairs are considered by increasing distance, and a
    pair is accepted when neither side has been taken yet. Ties go to the earlier reference,
    then to the earlier query. The result is therefore the stable matching under distance
    preferences, which is what "the nearest frame within the tolerance" means once conflicts
    are resolved -- but it is not necessarily the largest possible matching: with references
    at 0 and 10, queries at 6 and 30 and a tolerance of 25, the pair (10, 6) is taken first
    and 0 is left without a partner at 30, whereas (0, 6) and (10, 30) would both have fit.

    Candidates are each side's two nearest neighbours on the other side (the bracketing
    times in sorted order), so the cost is ``O((N + M) log)`` rather than ``O(N M)``. Any pair
    this leaves out would skip a frame on *both* sides, which is never the nearest choice
    whenever the tolerance is below one stream's frame spacing -- always the case for 75 ms
    against a 10 Hz stack.
    """
    if tolerance_ns < 0:
        raise ValueError(f"tolerance_ns must be non-negative, got {tolerance_ns}")

    reference = np.asarray(reference_ns, dtype=np.int64).ravel()
    query = np.asarray(query_ns, dtype=np.int64).ravel() + np.int64(offset_ns)
    if reference.size == 0 or query.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    reference_order = np.argsort(reference, kind="stable")
    query_order = np.argsort(query, kind="stable")
    sorted_reference = reference[reference_order]
    sorted_query = query[query_order]

    ref_pos, query_pos = _bracketing(sorted_reference, sorted_query)
    query_pos_2, ref_pos_2 = _bracketing(sorted_query, sorted_reference)
    candidates = np.unique(
        np.stack(
            [
                np.concatenate([ref_pos, ref_pos_2]),
                np.concatenate([query_pos, query_pos_2]),
            ],
            axis=1,
        ),
        axis=0,
    )
    ref_pos, query_pos = candidates[:, 0], candidates[:, 1]
    distance = np.abs(sorted_query[query_pos] - sorted_reference[ref_pos])
    within = distance <= tolerance_ns
    ref_pos, query_pos, distance = ref_pos[within], query_pos[within], distance[within]

    order = np.lexsort((query_pos, ref_pos, distance))
    reference_taken = np.zeros(reference.size, dtype=np.bool_)
    query_taken = np.zeros(query.size, dtype=np.bool_)
    chosen_reference: list[int] = []
    chosen_query: list[int] = []
    for r, q in zip(ref_pos[order].tolist(), query_pos[order].tolist()):
        if reference_taken[r] or query_taken[q]:
            continue
        reference_taken[r] = query_taken[q] = True
        chosen_reference.append(r)
        chosen_query.append(q)

    by_reference = np.argsort(np.asarray(chosen_reference, dtype=np.int64), kind="stable")
    reference_index = reference_order[np.asarray(chosen_reference, dtype=np.int64)[by_reference]]
    query_index = query_order[np.asarray(chosen_query, dtype=np.int64)[by_reference]]
    return reference_index, query_index


def _bracketing(points: NDArrayI64, grid: NDArrayI64) -> tuple[NDArrayI64, NDArrayI64]:
    """Return ``(point_position, grid_position)`` for each point's two sorted neighbours."""
    right = np.searchsorted(grid, points, side="left")
    left = right - 1
    positions = np.arange(points.size, dtype=np.int64)
    point_pos = np.concatenate([positions, positions])
    grid_pos = np.concatenate([left, right])
    valid = (grid_pos >= 0) & (grid_pos < grid.size)
    return point_pos[valid], grid_pos[valid]


# -- reading frames off a recording ----------------------------------------------------------


def frame_times(recording: Recording, entity_path: EntityPathLike) -> tuple[NDArrayI64, NDArrayI64]:
    """Return one entity's distinct ``FRAME`` values and the ``TIMESTAMP`` each sits at.

    Raises:
        ValueError: When the entity has no frames, when a chunk carries one of the two
            timelines without the other, or when one frame is recorded at two different
            timestamps. The check is per entity: a camera entity legitimately stamps the same
            frame with its own capture time.
    """
    path = as_entity_path(entity_path)
    frames: list[NDArrayI64] = []
    times: list[NDArrayI64] = []
    for chunk in recording.chunks(path):
        frame_index = chunk.index(FRAME)
        time_index = chunk.index(TIMESTAMP)
        if frame_index is None and time_index is None:
            continue
        if frame_index is None or time_index is None:
            missing = TIMESTAMP if time_index is None else FRAME
            raise ValueError(
                f"{path} has a chunk on one of the two timelines only (missing "
                f"{missing.name!r}); alignment needs both FRAME and TIMESTAMP on every chunk",
            )
        frames.append(frame_index.times)
        times.append(time_index.times)
    if not frames:
        raise ValueError(f"{path} has no chunks on the FRAME timeline to align")

    frame = np.concatenate(frames)
    time = np.concatenate(times)
    order = np.lexsort((time, frame))
    frame, time = frame[order], time[order]
    first = np.concatenate(([True], frame[1:] != frame[:-1]))
    last = np.concatenate((first[1:], [True]))
    disagreeing = np.flatnonzero(time[first] != time[last])
    if disagreeing.size:
        index = int(np.flatnonzero(first)[disagreeing[0]])
        raise ValueError(
            f"{path} records frame {int(frame[index])} at two different timestamps "
            f"({int(time[index])} and {int(time[np.flatnonzero(last)[disagreeing[0]]])} ns)",
        )
    return frame[first], time[first]


# -- the alignment -------------------------------------------------------------------------


@define(frozen=True, slots=True)
class FrameAlignment:
    """Which query frame answers which reference frame, and which frames got no partner."""

    reference_frames: NDArrayI64 = field(converter=_as_i64, eq=cmp_using(eq=np.array_equal))
    """Paired reference ``FRAME`` values, ascending."""

    query_frames: NDArrayI64 = field(converter=_as_i64, eq=cmp_using(eq=np.array_equal))
    """The query ``FRAME`` paired with each reference frame."""

    reference_times: NDArrayI64 = field(converter=_as_i64, eq=cmp_using(eq=np.array_equal))
    query_times: NDArrayI64 = field(converter=_as_i64, eq=cmp_using(eq=np.array_equal))
    """Recorded query timestamps; the skew ``offset_ns`` is *not* applied here."""

    unmatched_reference_frames: NDArrayI64 = field(
        converter=_as_i64,
        eq=cmp_using(eq=np.array_equal),
    )
    unmatched_query_frames: NDArrayI64 = field(converter=_as_i64, eq=cmp_using(eq=np.array_equal))

    tolerance_ns: int = field(converter=int, kw_only=True)
    offset_ns: int = field(default=0, converter=int, kw_only=True)
    reference_path: str = field(default="", converter=str, kw_only=True)
    query_path: str = field(default="", converter=str, kw_only=True)

    def __attrs_post_init__(self) -> None:
        lengths = {
            len(self.reference_frames),
            len(self.query_frames),
            len(self.reference_times),
            len(self.query_times),
        }
        if len(lengths) != 1:
            raise ValueError(f"Paired arrays have mismatched lengths: {sorted(lengths)}")
        if self.tolerance_ns < 0:
            raise ValueError(f"tolerance_ns must be non-negative, got {self.tolerance_ns}")

    @property
    def num_pairs(self) -> int:
        return len(self.reference_frames)

    @property
    def deltas_ns(self) -> NDArrayI64:
        """``query_times - reference_times`` per pair; add ``offset_ns`` for the residual."""
        return self.query_times - self.reference_times

    def mapping(self) -> dict[int, int]:
        """Return query frame -> reference frame."""
        return dict(zip(self.query_frames.tolist(), self.reference_frames.tolist()))

    def describe(self) -> dict[str, str]:
        """Return the alignment as string pairs, the shape ``RecordingMetadata.tags`` takes."""
        tags = {
            f"{_TAG_PREFIX}tolerance_ns": str(self.tolerance_ns),
            f"{_TAG_PREFIX}offset_ns": str(self.offset_ns),
            f"{_TAG_PREFIX}pairs": str(self.num_pairs),
            f"{_TAG_PREFIX}unmatched_reference": str(len(self.unmatched_reference_frames)),
            f"{_TAG_PREFIX}unmatched_query": str(len(self.unmatched_query_frames)),
        }
        if self.reference_path:
            tags[f"{_TAG_PREFIX}reference_path"] = self.reference_path
        if self.query_path:
            tags[f"{_TAG_PREFIX}query_path"] = self.query_path
        return tags

    def __str__(self) -> str:
        text = (
            f"FrameAlignment: {self.num_pairs} pairs, "
            f"{len(self.unmatched_reference_frames)} unmatched reference, "
            f"{len(self.unmatched_query_frames)} unmatched query, "
            f"tolerance={self.tolerance_ns}ns, offset={self.offset_ns}ns"
        )
        if self.num_pairs:
            worst = int(np.abs(self.deltas_ns + self.offset_ns).max())
            text += f", max |delta|={worst}ns"
        return text


@define(frozen=True, slots=True)
class AlignOptions:
    """How :func:`align_recordings` pairs and prunes."""

    tolerance_ns: int = field(default=DEFAULT_TOLERANCE_NS, converter=int)
    offset_ns: int = field(default=0, converter=int)
    """Added to the query's timestamps before pairing -- a known clock skew."""

    unmatched_reference: UnmatchedReference = "keep"

    def __attrs_post_init__(self) -> None:
        if self.tolerance_ns < 0:
            raise ValueError(f"tolerance_ns must be non-negative, got {self.tolerance_ns}")
        if self.unmatched_reference not in ("keep", "drop"):
            raise ValueError(
                f"unmatched_reference must be 'keep' or 'drop', got {self.unmatched_reference!r}"
            )


@define(frozen=True, slots=True)
class Aligned:
    """The two recordings after alignment, and the pairing that produced them."""

    reference: Recording
    query: Recording
    alignment: FrameAlignment


def align_frames(
    reference: Recording,
    query: Recording,
    *,
    reference_path: EntityPathLike = "/ground_truth/objects",
    query_path: EntityPathLike = "/estimation/objects",
    tolerance_ns: int = DEFAULT_TOLERANCE_NS,
    offset_ns: int = 0,
) -> FrameAlignment:
    """Pair the frames of one entity in each recording by timestamp.

    Nothing is rewritten; see :func:`reindex_frames` for that step.
    """
    reference_frames, reference_times = frame_times(reference, reference_path)
    query_frames, query_times = frame_times(query, query_path)
    ref_idx, query_idx = pair_times(
        reference_times,
        query_times,
        tolerance_ns=tolerance_ns,
        offset_ns=offset_ns,
    )
    return FrameAlignment(
        reference_frames[ref_idx],
        query_frames[query_idx],
        reference_times[ref_idx],
        query_times[query_idx],
        np.setdiff1d(reference_frames, reference_frames[ref_idx]),
        np.setdiff1d(query_frames, query_frames[query_idx]),
        tolerance_ns=tolerance_ns,
        offset_ns=offset_ns,
        reference_path=str(as_entity_path(reference_path)),
        query_path=str(as_entity_path(query_path)),
    )


# -- deriving recordings -------------------------------------------------------------------


def reindex_frames(
    recording: Recording,
    alignment: FrameAlignment,
    *,
    entity_paths: Iterable[EntityPathLike] | None = None,
) -> Recording:
    """Return the query recording with its ``FRAME`` values rewritten to the reference's.

    Every partition whose frame was paired moves to the paired reference frame; partitions
    of unmatched frames are dropped. ``TIMESTAMP`` and any other index are untouched, so an
    evaluation on that axis still sees the original stamps. Static chunks and chunks that
    carry no ``FRAME`` (a bag's ``/tf`` samples) are copied as they are.

    Args:
        recording: The query recording.
        alignment: The pairing, from :func:`align_frames`.
        entity_paths: Which entities to rewrite; by default every entity with frames, since
            entities of one recording share the ``FRAME`` axis.
    """
    keys = np.asarray(alignment.query_frames, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    values = np.asarray(alignment.reference_frames, dtype=np.int64)[order]

    def rewrite(chunk: Chunk, index: TimeColumn) -> Chunk | None:
        position = np.searchsorted(keys, index.times)
        clipped = np.minimum(position, max(keys.size - 1, 0))
        hit = (
            (position < keys.size) & (keys[clipped] == index.times)
            if keys.size
            else np.zeros(
                len(index.times),
                dtype=np.bool_,
            )
        )
        if not hit.any():
            return None
        picked = chunk.select_partitions(hit)
        indexes = tuple(
            TimeColumn(FRAME, values[position[hit]]) if each.timeline == FRAME else each
            for each in picked.indexes
        )
        return evolve(picked, indexes=indexes)

    return _derive(recording, rewrite, entity_paths, tags=alignment.describe())


def drop_frames(
    recording: Recording,
    frames: ArrayLike,
    *,
    entity_paths: Iterable[EntityPathLike] | None = None,
) -> Recording:
    """Return the recording without the given ``FRAME`` values, on every entity with frames."""
    dropped = np.asarray(frames, dtype=np.int64).ravel()

    def rewrite(chunk: Chunk, index: TimeColumn) -> Chunk | None:
        keep = ~np.isin(index.times, dropped)
        if not keep.any():
            return None
        if keep.all():
            return chunk
        return chunk.select_partitions(keep)

    return _derive(recording, rewrite, entity_paths, tags={})


def align_recordings(
    reference: Recording,
    query: Recording,
    *,
    options: AlignOptions | None = None,
    reference_path: EntityPathLike = "/ground_truth/objects",
    query_path: EntityPathLike = "/estimation/objects",
) -> Aligned:
    """Pair the frames and derive the two recordings an evaluation should see.

    The query comes back re-indexed onto the reference's frames; the reference comes back
    unchanged, or without its unanswered frames under ``unmatched_reference="drop"``. Both
    carry the pairing in their metadata tags.
    """
    chosen = options if options is not None else AlignOptions()
    alignment = align_frames(
        reference,
        query,
        reference_path=reference_path,
        query_path=query_path,
        tolerance_ns=chosen.tolerance_ns,
        offset_ns=chosen.offset_ns,
    )
    tags = {
        **alignment.describe(),
        f"{_TAG_PREFIX}unmatched_reference_policy": chosen.unmatched_reference,
    }

    aligned_query = reindex_frames(query, alignment)
    aligned_reference = reference
    if chosen.unmatched_reference == "drop" and alignment.unmatched_reference_frames.size:
        aligned_reference = drop_frames(reference, alignment.unmatched_reference_frames)

    return Aligned(
        _with_tags(aligned_reference, tags),
        _with_tags(aligned_query, tags),
        alignment,
    )


# -- plumbing --------------------------------------------------------------------------------


def _derive(
    recording: Recording,
    rewrite: Callable[[Chunk, TimeColumn], Chunk | None],
    entity_paths: Iterable[EntityPathLike] | None,
    *,
    tags: Mapping[str, str],
) -> Recording:
    """Rebuild a recording, passing each FRAME-bearing chunk of the chosen entities through ``rewrite``."""
    chosen: set[EntityPath] | None = (
        {as_entity_path(path) for path in entity_paths} if entity_paths is not None else None
    )
    store = Store()
    for path in recording.entity_paths():
        for chunk in recording.static_chunks(path):
            store.send_chunk(chunk)
        for chunk in recording.chunks(path):
            index = chunk.index(FRAME)
            if index is None or (chosen is not None and path not in chosen):
                store.send_chunk(chunk)
                continue
            derived = rewrite(chunk, index)
            if derived is not None:
                store.send_chunk(derived)
    return _with_tags(evolve(recording, store=store), tags)


def _with_tags(recording: Recording, tags: Mapping[str, str]) -> Recording:
    """Return the recording with ``tags`` merged in; existing ``align.*`` keys are replaced."""
    if not tags:
        return recording
    merged = {
        key: value for key, value in recording.metadata.tags if not key.startswith(_TAG_PREFIX)
    }
    merged.update(tags)
    return recording.with_metadata(tags=tuple(merged.items()))
