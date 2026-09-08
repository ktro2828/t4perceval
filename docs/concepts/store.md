# Store

The `Store` is the mutable log everything else reads and writes. It says _what rows exist and when_;
it deliberately does not say what the integers in those rows mean -- pairing it with the registries
that do is [`Recording`](../user-guide/persistence.md)'s job.

```text
Store
 ├── temporal   entity path → [Chunk, Chunk, ...]   indexed along timelines
 └── static     entity path → [Chunk, Chunk, ...]   belongs to every point in time
```

It replaces the `Catalog → Scenario → Scene → List[PerceptionFrameResult]` nesting of the original
package. One frame is `latest_at`; a whole scene is `range`; anything shared by every frame is
`log_static`.

## Chunks

A **chunk** is one write. It holds:

| Field         | Meaning                                                      |
| :------------ | :----------------------------------------------------------- |
| `entity_path` | where it is filed                                            |
| `columns`     | descriptor → component, all of the same length               |
| `indexes`     | one `TimeColumn` per timeline, one time per **partition**    |
| `offsets`     | where each partition starts in the columns                   |
| `frame_id`    | the [coordinate frame](coordinate-system.md) the rows are in |
| `is_static`   | whether it belongs to every point in time                    |

A **partition** is one observation: the rows logged together at one time. `store.log` writes a chunk
with a single partition; a range query concatenates many chunks' partitions into one, keeping the
partition boundaries so a metric can still tell frames apart.

```python
from t4perceval import Detections3D, Store, TimePoint

store = Store()
store.log(
    "/estimation/objects",
    Detections3D(
        position=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
        size=[[1.9, 4.5, 1.6]] * 2,
        class_id=[0, 0],
        confidence=[0.9, 0.4],
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)

chunk = store.chunks("/estimation/objects")[0]
chunk.num_rows  # 2
chunk.num_partitions  # 1
chunk.frame_id  # 'base_link'
```

Chunks are frozen and their arrays read-only. Moving an entity between stores therefore copies
nothing -- the new store references the same objects.

## Temporal and static data

`log` writes temporal data. `log_static` writes data that belongs to **every** point on **every**
timeline: a sensor extrinsic, a trajectory's time axis, anything that does not change.

```python
from t4perceval import Transform3D

store.log_static(
    "/tf/lidar",
    Transform3D(
        translation=[1.2, 0.0, 1.8],
        rotation=[0.0, 0.0, 0.0, 1.0],
        child_frame_id="lidar",
    ),
    frame_id="base_link",
)
```

Two things follow:

- **Static takes precedence.** When a static and a temporal column share a descriptor, the static
  one wins in a view. A forgotten static write changes results rather than raising, which is why
  `build_evaluation_store` moves static chunks first.
- **Static keeps its `frame_id`.** Static is a statement about _time_, not about the kind of data.
  A fixed extrinsic states the frame it is expressed in as surely as an ego pose does.

Static rows do not surface through `latest_at` or `range` as partitions; read them with
`static()` (columns) or `static_chunks()` (whole chunks, when the frame matters).

## Querying

Two queries, both returning an [`EntityView`](#entityview):

```python
from t4perceval import FRAME, TimeRange

# The most recent partition at or before a time.
one = store.latest_at("/estimation/objects", timeline=FRAME, at=0)

# Every partition inside a range, ordered by time.
many = store.range("/estimation/objects", timeline=FRAME, time_range=TimeRange.everything())
```

`TimeRange.single(0)` narrows to one frame; `TimeRange.everything()` takes the lot. Both accept
`components=[...]` to restrict which columns come back. When several partitions share a time,
`latest_at` prefers the most recently logged one.

Inspection without a query:

```python
store.entity_paths()  # every path with data, in insertion order
store.timelines()  # every timeline any chunk is indexed on
store.times("/estimation/objects", FRAME)  # the sorted, unique times of one entity
store.static_frame_id("/tf/lidar")  # 'base_link'
```

## EntityView

An `EntityView` is a **lazy window**: it refers to the chunk it came from and copies nothing until a
column is asked for.

```python
view = store.range("/estimation/objects", timeline=FRAME, time_range=TimeRange.everything())

len(view)  # rows in range
view.frame_id  # 'base_link'
view.has(*Detections3D.required_descriptors())
view.component(POSITION)  # one column, or None
view.materialize(Detections3D)  # back to the archetype
view.times(FRAME)  # the time of every row
view.partition_ids()  # which partition each row came from
view.select(mask)  # a narrower view, still lazy
view.to_chunk()  # materialize into a chunk
```

Static columns are broadcast to the view's row count, so a consumer does not have to know whether a
column was logged once or per frame.

## Writing results back

A [system](evaluation-pipeline.md) writes with `send_chunk`, which appends to the entity's temporal
or static series as the chunk declares. `Pipeline.run` does this for you. That is what keeps a
filter's verdict and a matcher's score in the store instead of discarding them.

## Where to go next

- [Timeline](timeline.md) -- `FRAME`, `TIMESTAMP` and what a range means.
- [Logging data](../user-guide/logging-data.md) -- the how-to.
- [Querying data](../user-guide/querying-data.md) -- the how-to.
