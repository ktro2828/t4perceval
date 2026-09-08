# Querying data

Reading rows back out of a `Store` -- or out of a read-only [`Recording`](persistence.md), which
exposes the same query surface.

## The two queries

```python
from t4perceval import FRAME, TimeRange

# every partition inside a range, ordered by time
scene = store.range("/estimation/objects", timeline=FRAME, time_range=TimeRange.everything())

# the most recent partition at or before a time
frame = store.latest_at("/estimation/objects", timeline=FRAME, at=3)
```

Both return an `EntityView`: a lazy window onto the chunks that matched, which copies nothing until
you ask for a column.

Use `range` for anything aggregated over frames and for per-frame work you want vectorized; use
`latest_at` when you genuinely want "the state as of now", including a value logged in an earlier
frame.

## Restricting the range

```python
TimeRange.everything()  # the whole axis
TimeRange.single(3)  # exactly frame 3
TimeRange(0, 10)  # frames 0..10, inclusive
TimeRange(0, 10, include_end=False)  # frames 0..9
```

Because every intermediate is in the store, a per-frame answer and a whole-scene answer differ only
in the range -- nothing is recomputed.

## Restricting the columns

```python
from t4perceval.descriptors import CLASS_ID, POSITION

view = store.range(
    "/estimation/objects",
    timeline=FRAME,
    time_range=TimeRange.everything(),
    components=[POSITION, CLASS_ID],
)
```

Worth doing when a scene is large and you only need two columns.

## Reading a view

```python
from t4perceval import Detections3D
from t4perceval.descriptors import POSITION

len(view)  # rows in range
view.entity_path  # /estimation/objects
view.frame_id  # 'base_link', or None if unstated
view.descriptors  # every column present
view.has(POSITION)  # is this column here?

view.component(POSITION).values  # the raw (N, 3) array, read-only
view.materialize(Detections3D)  # back to the archetype, with validation
```

`component` returns `None` for a descriptor the entity does not carry; `materialize` raises if a
required one is missing.

## Keeping frames apart

A range query concatenates several frames into one view. The frame boundaries survive:

```python
view.times(FRAME)  # the time of every row
view.partition_ids()  # which partition (observation) each row came from
```

That is how a metric defined per frame does its per-frame work over one vectorized query.

## Narrowing a view

```python
import numpy as np

near = view.select(np.array([True, False, True]))  # a mask, or indices
```

`select` on a view is lazy -- it still refers to the original chunk. `select` on a component, an
archetype or a chunk produces independent data instead.

To narrow by a filter's mask, use [`masked_view`](filtering.md#reading-a-mask).

## Static columns

Static data is broadcast into a view, so a consumer does not have to know whether a column was
logged once or per frame. To read it directly:

```python
store.static("/tf/lidar")  # descriptor -> component
store.static_chunks("/tf/lidar")  # whole chunks, when the frame_id matters
store.static_frame_id("/tf/lidar")  # 'base_link'
```

## Inspecting a store

```python
store.entity_paths()  # every path with data, in insertion order
store.timelines()  # every timeline any chunk is indexed on
store.times("/estimation/objects", FRAME)  # the sorted, unique times of one entity
store.chunks("/estimation/objects")  # the raw chunks, in log order
```

Listing entity paths after a pipeline run is the quickest way to see what an evaluation produced:

```text
/estimation/objects
/estimation/objects/filter/distance
/estimation/objects/kept
/ground_truth/objects
/matching/center_distance/0
/metrics/ap/0
/metrics/map
```

## Reading results

Matching verdicts and metric values are ordinary entities, read the same way:

```python
from t4perceval import MatchResults, MetricValues

matches = store.range(
    "/matching/center_distance", timeline=FRAME, time_range=TimeRange.everything()
).materialize(MatchResults)
matches.num_tp, matches.num_fp, matches.num_fn

values = store.range("/metrics/map", timeline=FRAME, time_range=TimeRange.everything()).materialize(
    MetricValues
)
values.aggregate  # the row aggregated over classes and thresholds
values.of_class(labels.class_id("car"))
```

See [Metrics](metrics.md) for the `MetricValues` layout and
[Offline analysis](offline-analysis.md) for querying a finished run.

## Where to go next

- [Filtering](filtering.md) · [Matching](matching.md) · [Metrics](metrics.md)
- [Store](../concepts/store.md) -- what a chunk, a partition and a view actually are.
