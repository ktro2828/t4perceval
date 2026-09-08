# Timeline

A **timeline** is a named integer axis. A chunk can be indexed along several at once, and a query
picks one.

```python
from t4perceval import FRAME, TIMESTAMP

str(FRAME)  # 'frame'
str(TIMESTAMP)  # 'timestamp_ns'
FRAME.kind  # TimeKind.SEQUENCE
TIMESTAMP.kind  # TimeKind.TIMESTAMP
```

## The two built-in axes

| Timeline    | Name           | Kind        | Unit                    | What it counts               |
| :---------- | :------------- | :---------- | :---------------------- | :--------------------------- |
| `FRAME`     | `frame`        | `SEQUENCE`  | index                   | keyframes, samples, messages |
| `TIMESTAMP` | `timestamp_ns` | `TIMESTAMP` | nanoseconds since epoch | when it actually happened    |

Most evaluation runs on `FRAME`: metrics are defined per frame, and a frame index is dense and
comparable. `TIMESTAMP` is what two independently imported recordings share, because their frame
numbering does not agree -- a T4 scene numbers its frames by sample, a bag by message.

You can define your own axis; `Timeline("scenario", TimeKind.SEQUENCE)` is an ordinary value.

## TimePoint

A `TimePoint` is a position on one or more timelines. It is what `Store.log` takes.

```python
from t4perceval import FRAME, TIMESTAMP, TimePoint

at = TimePoint.at(frame=0, timestamp_ns=1_624_164_470_849_887_000)
at[FRAME]  # 0
at[TIMESTAMP]  # 1624164470849887000
FRAME in at  # True
at.get(TIMESTAMP)  # 1624164470849887000
```

Logging with both axes is what lets the same chunk answer a frame query _and_ a timestamp query.
Importers always record both.

## TimeRange

A `TimeRange` is an inclusive-by-default interval, and it is what `Store.range` takes.

```python
from t4perceval import TimeRange

TimeRange.everything()  # the whole axis
TimeRange.single(3)  # exactly frame 3
TimeRange(0, 10)  # frames 0..10
TimeRange(0, 10, include_end=False)  # frames 0..9
```

Because every intermediate lives in the store, changing the range is the _only_ difference between
a per-frame answer and a whole-scene answer -- nothing is recomputed:

```python
scene = store.range(matcher.target, timeline=FRAME, time_range=TimeRange.everything())
frame = store.range(matcher.target, timeline=FRAME, time_range=TimeRange.single(0))
```

## Partitions keep frames apart

A range query concatenates several frames' rows into one view, but the partition boundaries survive.
`view.times(FRAME)` gives every row's time and `view.partition_ids()` gives the partition it came
from, which is how metrics that are defined per frame -- MOTA, ID switches -- do their per-frame
work over a single vectorized query.

## Aligning two recordings

Two recordings imported separately share only `TIMESTAMP`. Matching evaluates the union of both
`FRAME` sets, so unrelated indices would score as all-FP and all-FN frames rather than fail --
plausible numbers, not an error. [`t4perceval.align`](../recipes/align-frames.md) pairs the frames
first: every ground-truth frame takes the nearest estimation frame within a tolerance, one-to-one,
and the estimation's `FRAME` values are rewritten to the ground truth's.

```python
from t4perceval.align import AlignOptions
from t4perceval.evaluation import build_evaluation_store

setup = build_evaluation_store(
    ground_truth,
    estimation,
    align=AlignOptions(tolerance_ns=75_000_000),  # 75 ms
)
```

## Where to go next

- [Querying data](../user-guide/querying-data.md) -- range queries in practice.
- [Align frames by timestamp](../recipes/align-frames.md) -- the full recipe.
