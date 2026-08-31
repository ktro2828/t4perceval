from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from attrs import cmp_using, define, field

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from typing_extensions import Self

    from t4perceval.typing import ArrayLike, NDArrayBool, NDArrayI64

__all__ = (
    "FRAME",
    "TIMESTAMP",
    "TimeColumn",
    "TimeKind",
    "TimePoint",
    "TimeRange",
    "Timeline",
)


class TimeKind(Enum):
    """How the 64-bit integers on a timeline should be interpreted."""

    SEQUENCE = auto()
    """A monotonically increasing counter, such as a frame index."""
    TIMESTAMP = auto()
    """Nanoseconds since the Unix epoch."""
    DURATION = auto()
    """Elapsed nanoseconds relative to an unspecified origin."""


@define(frozen=True, slots=True)
class Timeline:
    """A named axis that data can be indexed along.

    Data may live on several timelines at once; each one is queried independently.
    """

    name: str = field(converter=str)
    kind: TimeKind = field(default=TimeKind.SEQUENCE)

    def __attrs_post_init__(self) -> None:
        if self.name == "":
            raise ValueError("Timeline.name must not be empty")

    def __str__(self) -> str:
        return self.name


#: Frame index within a scene.
FRAME = Timeline("frame", TimeKind.SEQUENCE)

#: Sensor timestamp in nanoseconds since the Unix epoch.
TIMESTAMP = Timeline("timestamp_ns", TimeKind.TIMESTAMP)


def _as_entries(
    value: Mapping[Timeline, int] | Iterable[tuple[Timeline, int]],
) -> tuple[tuple[Timeline, int], ...]:
    items = value.items() if hasattr(value, "items") else value
    entries = tuple((timeline, int(time)) for timeline, time in items)

    names = [timeline.name for timeline, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"TimePoint has duplicate timelines: {sorted(names)}")

    return tuple(sorted(entries, key=lambda entry: entry[0].name))


@define(frozen=True, slots=True)
class TimePoint:
    """A position on one or more timelines.

    Examples:
        >>> point = TimePoint.at(frame=3, timestamp_ns=1_624_164_470_849_887_000)
        >>> point[FRAME]
        3
    """

    entries: tuple[tuple[Timeline, int], ...] = field(converter=_as_entries)

    @classmethod
    def at(cls, *, frame: int | None = None, timestamp_ns: int | None = None) -> Self:
        """Build a time point on the built-in :data:`FRAME` / :data:`TIMESTAMP` timelines."""
        entries: list[tuple[Timeline, int]] = []
        if frame is not None:
            entries.append((FRAME, frame))
        if timestamp_ns is not None:
            entries.append((TIMESTAMP, timestamp_ns))
        if not entries:
            raise ValueError("TimePoint.at() requires at least one of frame or timestamp_ns")
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Timeline]:
        return (timeline for timeline, _ in self.entries)

    def __contains__(self, timeline: Timeline) -> bool:
        return any(known == timeline for known in self)

    def __getitem__(self, timeline: Timeline) -> int:
        for known, time in self.entries:
            if known == timeline:
                return time
        raise KeyError(f"TimePoint has no time on timeline {timeline.name!r}")

    def get(self, timeline: Timeline, default: int | None = None) -> int | None:
        """Return the time on ``timeline``, or ``default`` when absent."""
        for known, time in self.entries:
            if known == timeline:
                return time
        return default

    @property
    def timelines(self) -> tuple[Timeline, ...]:
        return tuple(timeline for timeline, _ in self.entries)


@define(frozen=True, slots=True)
class TimeRange:
    """A closed-by-default interval on a single timeline."""

    start: int = field(converter=int)
    end: int = field(converter=int)
    include_start: bool = field(default=True, kw_only=True)
    include_end: bool = field(default=True, kw_only=True)

    def __attrs_post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"TimeRange.end ({self.end}) must not precede start ({self.start})")

    @classmethod
    def everything(cls) -> Self:
        """Return a range covering every representable time."""
        info = np.iinfo(np.int64)
        return cls(int(info.min), int(info.max))

    @classmethod
    def single(cls, time: int) -> Self:
        """Return a range containing exactly one time."""
        return cls(time, time)

    def contains(self, times: ArrayLike) -> NDArrayBool:
        """Return an elementwise membership mask for ``times``."""
        array = np.asarray(times, dtype=np.int64)
        after_start = array >= self.start if self.include_start else array > self.start
        before_end = array <= self.end if self.include_end else array < self.end
        return after_start & before_end


@define(frozen=True, slots=True)
class TimeColumn:
    """An index column of a :class:`~t4perceval.core.chunk.Chunk`.

    Holds one time per *partition*, not per row: all rows of a partition were observed
    at the same point on the timeline.
    """

    timeline: Timeline
    times: NDArrayI64 = field(eq=cmp_using(eq=np.array_equal))

    @times.validator
    def _check_times(self, _attribute: object, value: NDArrayI64) -> None:
        if value.ndim != 1:
            raise ValueError(f"TimeColumn.times must have shape (P,), got {value.shape}")

    @times.default
    def _default_times(self) -> NDArrayI64:
        return np.empty(0, dtype=np.int64)

    def __attrs_post_init__(self) -> None:
        frozen = np.ascontiguousarray(self.times, dtype=np.int64)
        if frozen is self.times and frozen.flags.writeable:
            frozen = frozen.copy()
        if frozen.flags.writeable:
            frozen.flags.writeable = False
        object.__setattr__(self, "times", frozen)

    @classmethod
    def of(cls, timeline: Timeline, times: ArrayLike) -> Self:
        """Build a time column from anything array-like."""
        return cls(timeline, np.asarray(times, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.times.shape[0])

    def select_partitions(self, indices: NDArrayI64) -> Self:
        """Return a column keeping only the given partitions, in order."""
        return type(self)(self.timeline, self.times[indices])
