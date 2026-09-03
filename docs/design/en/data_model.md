# Data Model Design

## Background

The original [`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation)
modelled one object as one Python object (`DynamicObject`). That shape causes several problems:

- `DynamicObject` holds position, orientation, shape, velocity, `tracked_*` and `predicted_*` as more
  than twenty fields on a single class. Whichever task you run, the fields it does not use sit there
  filled with `None`.
- Work happens by iterating a `List[DynamicObject]` in Python, so nothing vectorizes.
- The `EvaluationTask` enum leaks as `if` branches into config, matching, metrics and visualization.
- `Catalog → Scenario → Scene → PerceptionFrameResult` is a nest of Python lists, which makes
  cross-frame queries (CLEAR, HOTA, ADE) awkward to write.
- Intermediate products -- which rows a filter dropped, what a matcher scored -- are discarded and
  cannot be re-analysed.

`t4perceval` resolves this by implementing [rerun](https://github.com/rerun-io/rerun)'s data model
ourselves. Rerun itself is not a dependency; see "Relationship to Rerun" below.

## Layers

```text
                        ┌─────────────────────────────────────────┐
   t4perceval.system    │  System / Pipeline                      │  the "S" of ECS
                        │  filter · matching · metric · pass/fail │
                        └────────────────┬────────────────────────┘
                                         │ reads / writes Chunk
                        ┌────────────────▼────────────────────────┐
   t4perceval.core      │  Store          queried along timelines  │
                        │   ├ latest_at(entity, at)   → EntityView │
                        │   ├ range(entity, range)    → EntityView │
                        │   └ static                               │
                        ├──────────────────────────────────────────┤
                        │  Chunk    entity_path + indexes          │
                        │           + offsets + columns            │
                        ├──────────────────────────────────────────┤
                        │  Archetype   a bundle of components      │
                        ├──────────────────────────────────────────┤
                        │  Component   one NumPy column            │
                        │  ComponentDescriptor  column identity    │
                        │  EntityPath           stream address     │
                        │  Timeline             time axis          │
                        └──────────────────────────────────────────┘
                        ┌──────────────────────────────────────────┐
   t4perceval.transform │  Transform3D rows -> FrameGraph          │  frames as data
                        │  transform_edges · TransformResolver     │
                        └──────────────────────────────────────────┘
                        ┌──────────────────────────────────────────┐
   t4perceval.io        │  Chunk ↔ pyarrow.Table ↔ Parquet         │
                        └──────────────────────────────────────────┘
```

## EntityPath — what the data is about

A `/`-separated hierarchical path. Everything the original package expressed through types and
fields -- `frame_id`, the task enum, the estimation/ground-truth distinction -- becomes a difference
of path.

```python
EntityPath.parse("/estimation/objects")
EntityPath.parse("/ground_truth/objects")
EntityPath.parse("/estimation/objects") / "filter" / "distance"
```

### Conventions

| Path                                | Contents                                          |
| :---------------------------------- | :------------------------------------------------ |
| `/ground_truth/objects`             | Ground-truth objects                              |
| `/estimation/objects`               | Estimated objects                                 |
| `/estimation/objects/<channel>`     | Per-sensor-channel estimations                    |
| `/estimation/objects/filter/<name>` | A filter verdict about that entity (a mask)       |
| `/matching/<mode>`                  | Matching results (`center_distance`, `iou_3d`, …) |
| `/metrics/<name>`                   | Metric values (`map`, `clear`, …)                 |

Putting a filter verdict under the _source_ path is deliberate: one prefix query then finds an entity
together with every verdict recorded about it.

## ComponentDescriptor — column identity

```python
@define(frozen=True, slots=True)
class ComponentDescriptor:
    component: str  # identity is this field alone
    archetype: str | None  # eq=False; a provenance hint
    component_type: str | None  # eq=False; the component class name
```

**Key decision**: descriptor names are archetype-independent. `Detections3D` and
`Trackings3D` both expose their 3D centre as `POSITION` (that is, `"position"`).

```python
Detections3D.descriptor_of("position") == Trackings3D.descriptor_of("position")  # True
```

That is what lets a system declare `REQUIRES = (POSITION,)` and run against any entity carrying a 3D
position. Rerun qualifies its names by archetype (`Points3D:positions` vs `Boxes3D:positions`); for an
evaluation tool, requiring the same column _across_ archetypes is the more useful property, so we
deliberately differ here.

The canonical descriptors live in `t4perceval/descriptors.py`.

## Component — one column

`ColumnarComponent` implements the `values` field, `__len__`, `select()` and the Arrow round-trip
**once**. A subclass only declares its layout as class variables.

```python
@define(frozen=True, slots=True)
class BatchPosition3D(ColumnarComponent):
    SHAPE = (3,)  # per-row shape; () is scalar, ANY is inferred from the data
    DTYPE = np.float64  # every value is coerced to this dtype
    # VALUE_RANGE = (0.0, 1.0)   # optional inclusive bound
    # REQUIRE_FINITE = True      # optional; reject NaN / Inf
```

### Invariants

- `values` is always **read-only** (`values.flags.writeable is False`), consistent with the frozen
  dataclass around it.
- It never shares memory with a _writable_ array owned by the caller, so freezing can never surprise
  them.
- `select()` always returns independent data (fancy indexing copies). Lazy views are `EntityView`'s job.
- An empty input (`BatchPosition3D([])`) is accepted as zero rows. When `SHAPE` contains `ANY` the
  row shape is genuinely ambiguous, so use `empty()` there.

### The columns

| Module          | Component                                                                                                                               | Shape / dtype               |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------- |
| `vector.py`     | `BatchVector2D` / `BatchVector3D`                                                                                                       | `(2,)` / `(3,)` f64         |
| `geometry.py`   | `BatchPosition3D` `BatchPosition2D` `BatchQuaternion` `BatchVelocity` `BatchSize3D` `BatchSize2D`                                       | `BatchQuaternion` is `xyzw` |
| `scalar.py`     | `BatchClassId` (i32) `BatchConfidence` (f64, `[0,1]`) `BatchInstanceId` (i64) `BatchNumPoints` (i32) `BatchVisibility` (i8)             | `()`                        |
| `image.py`      | `BatchRoi` (i32, `(4,)`, `(x_min, y_min, height, width)`) `BatchPixel` (i32)                                                            |                             |
| `mask.py`       | `BatchMask` (bool)                                                                                                                      | `()`                        |
| `trajectory.py` | `BatchWaypoints3D` `(M,T,3)` `BatchModeConfidence` `(M,)` `BatchModeValid` `(M,)` `BatchTimestepValid` `(M,T)` `BatchTimeOffset` `(T,)` |                             |
| `matching.py`   | `BatchRowIndex` (i64) `BatchMatchingScore` (f64) `BatchMatchStatus` (i8)                                                                | `()`                        |

`BatchVector3D` does **not** inherit from `BatchVector2D`: a 3D vector is not a 2D vector, and this
package treats `isinstance` as a claim that must be true.

## Archetype — a bundle of components

```python
@define(frozen=True, slots=True)
class Trackings3D(Archetype):
    position = component_field(POSITION, BatchPosition3D)
    quaternion = component_field(QUATERNION, BatchQuaternion)
    size = component_field(SIZE, BatchSize3D)
    class_id = component_field(CLASS_ID, BatchClassId)
    confidence = component_field(CONFIDENCE, BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, kw_only=True)
    velocity = component_field(VELOCITY, BatchVelocity, optional=True, kw_only=True)
```

### Why inheritance was dropped

The archetypes used to form a chain, `BatchDetection3D → BatchTracking3D → BatchPrediction3D`. That
runs opposite to ECS composition:

- `select()` was duplicated as three near-identical bodies.
- Combinations such as "has a trajectory but no instance id" could not be expressed.
- `isinstance(tracking, Detections3D)` asserted "a tracking _is a kind of_ detection", which is
  not what the data means.

Now `Trackings3D` re-declares the box components explicitly. The descriptors are identical, so:

```python
tracking.has(*Detections3D.required_descriptors())  # True
isinstance(tracking, Detections3D)  # False
```

`has()` asks exactly the question a system asks through `REQUIRES`, and is the correct replacement for
the `isinstance` check.

### What the base provides

`select()` walks `attrs.fields()` and delegates to each component, so **no archetype implements it**.
`as_components()`, `from_components()`, `to_chunk()`, `from_chunk()`, `has()`, `descriptors()` and
`required_descriptors()` are likewise single implementations on the base.

### The archetypes

| Archetype                | Components                                                                                       |
| :----------------------- | :----------------------------------------------------------------------------------------------- |
| `Detections3D`           | position, quaternion, size, class_id, confidence, [velocity], [num_points], [visibility]         |
| `Detections2D`           | roi, class_id, confidence, [visibility]                                                          |
| `Trackings3D`            | Detection3D's columns + instance_id                                                              |
| `Trackings2D`            | Detection2D's columns + instance_id                                                              |
| `Predictions3D`          | Tracking3D's columns + waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset] |
| `Classifications2D`      | class_id, confidence, [instance_id]                                                              |
| `SemanticSegmentation2D` | pixel, class_id                                                                                  |
| `SemanticSegmentation3D` | point, class_id                                                                                  |
| `Trajectories3D`         | waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset]                        |
| `MatchResults`           | est_index, gt_index, matching_score, match_status                                                |

### Trajectory decisions

- `M` (modes) and `T` (timesteps) are **fixed within one instance**. Variable lengths are padded and
  masked by `BatchModeValid` / `BatchTimestepValid`.
- `mode_confidence` is defined as a per-mode posterior probability, but the sum over valid modes is
  **not** validated. Models that emit unnormalized scores are common, and normalizing here would
  silently change the metric. Only the `[0, 1]` range and finiteness are enforced.
- `waypoints` must be finite. An invalid timestep is expressed through `timestep_valid=False`, never
  as `NaN`.
- `time_offset` is an `(N, T)` column. When every row shares the axis, log it once as **static** data
  with `N == 1`.

## Timeline

```python
class TimeKind(Enum):
    SEQUENCE  # a monotonic counter, such as a frame index
    TIMESTAMP  # nanoseconds since the Unix epoch
    DURATION  # elapsed nanoseconds


FRAME = Timeline("frame", TimeKind.SEQUENCE)
TIMESTAMP = Timeline("timestamp_ns", TimeKind.TIMESTAMP)
```

One piece of data can sit on several timelines at once. `TimePoint.at(frame=3, timestamp_ns=...)`
records both, and a query picks whichever axis it wants.

### Dismantling `Header`

The old `Header(timestamp_ns, frame_id)` is gone.

- `timestamp_ns` → a value on the `TIMESTAMP` timeline (`Chunk.indexes`)
- `frame_id` → `Chunk.frame_id`, the coordinate frame every row of the chunk is expressed in

Archetypes therefore became pure bundles of components, with no header to carry around at
construction time.

> **Decided.** Rerun expresses coordinate frames through the entity-path hierarchy plus a
> `Transform3D` archetype (for example `/base_link/estimation/objects`). We keep
> `Chunk.frame_id` for perception data: an entity path answers _what the data is_, and folding
> the frame into it would make `/ground_truth/objects` unaddressable by a system that does not
> care which frame it is in.
>
> ~~A transform edge is the opposite case, and does live in the path:
> `/transforms/<parent>/<child>` with a `Transform3D` of `translation` and `rotation`. There the
> frame pair _is_ the identity of the data.~~
>
> **Reversed.** A transform now states its **parent** through `Chunk.frame_id` -- the same meaning
> that field has everywhere else -- and its **child** through a `child_frame_id` component, which is
> how ROS splits a `TransformStamped`. What decided it: a frame name had to be path-safe, so
> `/robot1/base_link` was inexpressible, and the frame graph could not be re-filed without being
> renamed. See "Transforms" below.

## Chunk — a column-oriented table

```python
@define(frozen=True, slots=True)
class Chunk:
    entity_path: EntityPath
    indexes: tuple[TimeColumn, ...]  # length P (partitions)
    offsets: NDArrayI64  # length P+1, row boundaries
    columns: dict[ComponentDescriptor, Component]  # each of length N = offsets[-1]
    frame_id: str | None = None
    is_static: bool = False
```

### Why a row is an object

A rerun chunk has one row per log call, with each cell holding a variable-length component batch.
`t4perceval` flattens instead: **one row is one object**, and `offsets` marks the frame boundaries.

The reason is that evaluation math is elementwise over objects -- distance matrices, IoU, TP/FP
verdicts -- and that maps directly onto NumPy over flat contiguous arrays, with no list offsets to
walk per frame. `offsets` still makes per-frame aggregation cheap (`partition(i)`, `partition_ids()`).

```text
offsets = [0, 2, 5]        indexes[frame].times = [0, 1]
           │  │  └── frame 1 is rows 2..4 (three objects)
           │  └───── frame 0 is rows 0..1 (two objects)
           └──────── always 0
```

### Invariants

- every `indexes` entry has length `num_partitions`
- every column has length `num_rows == offsets[-1]`
- `offsets[0] == 0` and offsets never decrease
- no duplicate timelines
- a chunk with `is_static=True` carries no index columns and has exactly one partition

### The `select()` contract

`Chunk.select()` preserves the partition structure, so a selection that **reorders rows across
partitions is rejected** (boolean masks and ascending index arrays always qualify). A partition may
become empty; its index entry is kept, so the time axis survives.

## Store — the mutable log

```python
store.send_chunk(chunk)
store.log(entity_path, archetype, at=TimePoint.at(frame=0), frame_id="base_link")
store.log_static(entity_path, archetype, frame_id="base_link")

store.latest_at(entity_path, timeline=FRAME, at=12)  # → EntityView
store.range(entity_path, timeline=FRAME, time_range=TimeRange(0, 99))  # → EntityView
```

This replaces the `Catalog → Scenario → Scene → List[PerceptionFrameResult]` nesting.

| Old structure         | New expression                             |
| :-------------------- | :----------------------------------------- |
| One frame             | `store.latest_at(...)`                     |
| One scene             | `store.range(..., TimeRange.everything())` |
| Shared by every frame | `store.log_static(...)`                    |

`static` is a statement about **time** -- "this does not depend on a timeline" -- not about the kind
of data. Anything may be logged either way, so a sensor calibration is a static `Transform3D` and an
ego pose is a temporal one, with no second archetype.

### Semantics (following Rerun)

- **Static data belongs to every timeline** and takes precedence over temporal data carrying the same
  descriptor. A one-row static column is broadcast across the view.
- A static write is kept as a **whole chunk**, so its `frame_id` survives -- `static_chunks()` and
  `static_frame_id()` reach it, while `static()` still returns just the columns, later writes winning
  per descriptor.
- A **static-only** entity therefore reads back as _zero rows_ through `latest_at` and `range`: a
  view is one temporal chunk plus a broadcast overlay, and there is no row count to broadcast to.
  That is deliberate. Surfacing static rows through a time query would invent objects in frames that
  have none, and hand index-less chunks to systems that ask a view for its times. Readers that want
  static rows ask for the chunk.
- `EntityView.frame_id` reports the **temporal** chunk's frame only. A static column's frame need not
  describe the rows -- a transform's frame is its edge's parent -- so letting it through would make
  an unrelated static column trip the cross-frame guard.
- `latest_at` returns the most recent partition at or **before** the given time. When several share
  that time, the most recently logged one wins.
- `range` orders partitions **by time** (ties keep log order).
- Chunks with different column sets may be logged to one entity. The error only appears when a
  `range` query actually spans incompatible chunks; `latest_at` reads a single chunk and always works.

## EntityView — the lazy window

```python
@define(frozen=True, slots=True)
class EntityView:
    chunk: Chunk
    indices: NDArrayI64  # normalized row indices into the chunk
    static: dict[ComponentDescriptor, Component]
```

`select()` only composes indices and **copies nothing**. Materialization happens in `component()`,
`materialize()` or `to_chunk()`.

```python
view.select(slice(None, None, 2)).select([1])  # no copy
view.component(POSITION)  # copies exactly one column
view.materialize(Detections3D)  # materializes as an archetype
```

### The copy/view contract

| API                   | Behaviour                                              |
| :-------------------- | :----------------------------------------------------- |
| `Component.select()`  | independent data (copy)                                |
| `Archetype.select()`  | independent data (copy)                                |
| `Chunk.select()`      | independent data (copy), partition structure preserved |
| `EntityView.select()` | index composition only (no copy)                       |

The three classes the TODO asked for -- `BatchDetection3DView`, `BatchTracking3DView`,
`BatchPrediction3DView` -- turned out to be unnecessary: the archetype is an argument to
`materialize()`, so one generic class covers all of them.

## LabelRegistry / InstanceRegistry — meaning for the integers

`BatchClassId` (i32) and `BatchInstanceId` (i64) are plain integer columns. The registries are the
only place that says what those integers mean.

```python
labels = LabelRegistry.from_names(["car", "truck", "pedestrian"])
labels.class_id("truck")  # 1
labels.encode(["car", "truck"])  # an i32 column for BatchClassId

merged = labels.merged({"vehicle": ["car", "truck"]})
merged.class_id("car") == merged.class_id("vehicle")  # True
```

This replaces `LabelConverter` together with `label_prefix`, `merge_similar_labels` and
`count_label_number`. Merging produces a **new registry** rather than setting a flag that match code
reads later, so the merge is visible in the data.

A registry is **metadata, not a column**: names are not numeric, and a registry describes a whole
recording rather than a row. It therefore travels as `SystemContext.labels` and in the Arrow schema
metadata -- the role Rerun gives to a static `AnnotationContext`.

`InstanceRegistry` interns string UUIDs into `BatchInstanceId` values and travels the same way, as
`SystemContext.instances`. Ids are stable for the lifetime of one registry, which is the property
tracking metrics need in order to compare identities across frames. `intern()` assigns an id to a UUID
it has not seen; `instance_id()` refuses to, and raises instead -- which is what filtering needs, so
that a mistyped UUID cannot quietly become a fresh identity.

## Transforms — coordinate frames as recorded data

A transform is an observation like any other, not hidden state owned by a service. One row is one
edge of the frame graph, split between the chunk and a column the way ROS splits a
`TransformStamped`:

| ROS `TransformStamped`  | `t4perceval`                 |
| :---------------------- | :--------------------------- |
| `header.frame_id`       | `Chunk.frame_id` -- parent   |
| `child_frame_id`        | `Transform3D.child_frame_id` |
| `transform.translation` | `Transform3D.translation`    |
| `transform.rotation`    | `Transform3D.rotation`       |

A row maps a point in the child frame into the parent: `p_parent = R p_child + t`.

The parent belongs on the chunk because `frame_id` already means "the frame these rows are expressed
in", which is exactly true of a transform's row.

### A transform is one edge, so its components are mono

`Transform3D` is the only archetype in the package whose components are **mono**
(`MonoComponent`). Every other one describes _N_ objects, so every column is a batch; a transform
describes one relationship, of which an entity holds exactly one per point in time. Its translation
is therefore a `(3,)` value and its child frame a `str`:

```python
pose = Transform3D(
    translation=[1.2, 0.0, 1.8], rotation=[0.0, 0.0, 0.0, 1.0], child_frame_id="lidar"
)
pose.translation.value  # array([1.2, 0. , 1.8])
pose.child_frame_id.name  # 'lidar'
```

There is no row to index into, and "what if it has three rows?" is not a question the type can be
asked -- it raises at construction.

**Mono is a boundary type; the store stays columnar.** `as_components()` widens a mono value into
its `BATCH` counterpart on the way into a chunk (`Position3D` to `BatchPosition3D`, `FrameId` to
`BatchFrameId`), and the archetype's field converter narrows it back. That division is load-bearing:
`Store.range` concatenates partitions, so a query spanning three samples of one edge returns a
three-row column -- which a type permitting exactly one row could not be. A chunk therefore always
holds batch columns:

```python
scene.range("/tf/base_link", timeline=FRAME, time_range=EVERYTHING).component(TRANSLATION)
# BatchPosition3D of 3 rows -- the ego's path

Transform3D.from_chunk(scene.static_chunks("/tf/LIDAR_TOP")[0]).translation.value
# array([0., 0., 2.]) -- one row narrows back to one value
```

Materializing a multi-row view as a `Transform3D` raises rather than silently taking the first row.
Read the columns for the series; materialize for the edge.

Neither frame is in the entity path. A path says where data is _filed_; a frame names a _node of the
graph_. Conflating them means a frame name has to be path-safe -- `/robot1/base_link` was
inexpressible -- and a tree cannot be re-filed without being renamed.

```python
store.log_static(  # a calibration: fixed
    "/tf/lidar",
    Transform3D(translation=[1.2, 0.0, 1.8], rotation=[0.0, 0.0, 0.0, 1.0], child_frame_id="lidar"),
    frame_id="base_link",
)
store.log(  # an ego pose: per frame
    "/tf/base_link",
    Transform3D(
        translation=[10.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0], child_frame_id="base_link"
    ),
    at=TimePoint.at(frame=1, timestamp_ns=...),
    frame_id="map",
)
```

### Static or temporal is a statement about time

Both of those are the same archetype, and neither is a special case. `static` means "does not depend
on a timeline", so a calibration is static and an ego pose is not, and a static write keeps its
`frame_id` -- which is what makes the edge interpretable without inventing a sample time for it.

The earlier design could not do this: static storage discarded everything but the columns, so a
fixed extrinsic had to be logged as a _temporal_ sample at the scene's first frame and rely on
`latest_at` reaching forward from it. The cost was that a windowed `range` starting after that sample
never saw it. A static edge has no window, so that cost is gone.

The remaining consequence is that static rows do not surface through `latest_at` or `range` (see
"Store"), so a fixed edge is read through `static_chunks()` rather than through a time query.

### `TRANSLATION` and `ROTATION` are their own descriptors

Not a reuse of `POSITION` and `QUATERNION`, so a system asking for a 3D position -- a distance
filter, say -- cannot be pointed at a transform entity and appear to work.

### Frame names are a text column

`FrameId` and its storage form `BatchFrameId` are the one non-numeric component in the model. The
rule that admits it: **text is admissible where there is one value per edge, not one per object.** Class and instance names are per object
and stay interned in a registry, carried as metadata; frame names are per edge, so a registry would
save a few hundred bytes while leaving `Chunk.frame_id` a string and the column an integer -- two
encodings of one concept, with nothing checking they agree -- and would have to be threaded through
`Store`, `SystemContext`, `Recording` and the Arrow schema.

The column is `object`-typed, never a fixed-width `<U*`: numpy truncates silently, so a long name
would become a different, shorter frame and two sensors could collapse into one. Its Arrow type is
pinned to `string` rather than inferred, because inference on an object array depends on the values
-- a zero-row column infers `null`, which the schema rejects for declaring every field non-nullable.

### Finding the graph again

```python
from t4perceval.transform import FrameGraph, TransformResolver, transform_edges

transform_edges(recording)  # -> (TransformEdge(parent, child, entity_path, is_static), ...)
FrameGraph.of(recording).frames()  # -> ("map", "base_link", "LIDAR_TOP", ...)
```

Discovery _reads_ the chunks -- the earlier design enumerated the graph from the list of entity paths
alone, which is no longer possible and was the price of frames being data. The read is small: a chunk
with no `child_frame_id` is skipped on a dict lookup, and one that has it holds one edge per row.

- A chunk without a `child_frame_id` column is **ignored**, so an unrelated entity filed nearby
  cannot break discovery.
- A chunk that names a child but states no `frame_id` **raises**: its parent is unknown, so the edge
  cannot be interpreted at all. This differs from `require_same_frame`, where an unstated frame is
  merely "no opinion" -- that rule is about comparing two things, this one about interpreting one.
- The same `(parent, child)` recorded in two places **raises**, because nothing could choose.

### Resolving a chain

`TransformResolver` is the interpretation step the store does not take. It walks the graph
breadth-first (fewest hops, since every composition compounds error), inverts an edge walked against
the direction it was recorded in -- exact, for a rigid transform -- and composes:

```python
resolver = TransformResolver.of(recording, timeline=FRAME)
resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=1)
# T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar
```

Static and temporal edges take part in one graph, exactly as ROS composes a latched transform with a
live one. A temporal edge picks its sample by `LookupPolicy`: `LATEST`, `EXACT`, `NEAREST` or
`INTERPOLATE` (linear translation, `Slerp` rotation). A static edge ignores the policy -- interpolating
something that never changes is a question with one answer, not an error. An unknown or unreachable
frame raises, so a missing calibration can never quietly resolve to identity.

It is **not** a `System`: a system returns chunks for a pipeline to file, whereas a lookup answers a
question and writes nothing. Materializing a _transformed entity_ is the system-shaped job, and it is
still blocked on a passthrough system being unable to declare the columns it carries.

### The frame tree of an imported scene

`log_scene_transforms` in `t4perceval.importer.t4` records two kinds of edge:

| Edge                     | Filed at        | How                     | Source              |
| :----------------------- | :-------------- | :---------------------- | :------------------ |
| `map -> base_link`       | `/tf/base_link` | one sample per keyframe | `ego_pose`          |
| `base_link -> <channel>` | `/tf/<channel>` | static                  | `calibrated_sensor` |

- Extrinsics come from `calibrated_sensor`, which holds one row per sensor, rather than from walking
  `sample_data` -- that would do work proportional to the scene to recover a handful of values. They
  are a property of the dataset's sensor set, so a channel this scene records no data for still
  appears, and the tree does not depend on which frames were imported.
- Two calibrations for one channel raise unless they place the sensor identically, and a calibration
  naming a sensor the `sensor` table does not list raises as well. Settling on one silently would put
  a wrong extrinsic into the frame tree, where nothing downstream could notice.
- The dataset stores rotations as `wxyz` and this package as `xyzw`. Both are four floats, so taking
  them verbatim yields a plausible rotation rather than an error; the reorder happens at this
  boundary, and nowhere else.
- Ego poses are recorded per **keyframe**, on both timelines, so a lookup works whether an evaluation
  runs on frame indices or on timestamps. Sub-frame ego motion is not represented, which is the
  resolution the annotations themselves have.

Still to come: a system that materializes a _transformed entity_. Until it exists, transforms are
recorded, discoverable and resolvable, but nothing rewrites an object chunk into another frame -- so
the system layer refuses to compare geometry across frames instead. See "Coordinate frames" in
[system.md](system.md).

## IO — Arrow and Parquet

```python
table = chunk_to_table(chunk, labels=labels)
chunk, labels = chunk_from_table(table)

write_parquet(chunk, path, labels=labels)
chunk, labels = read_parquet(path)
```

### Schema design

- One component becomes one Arrow field, named by `ComponentDescriptor.component`.
- Vectors are `fixed_size_list<T, W>`; `BatchWaypoints3D` nests them to express `(M, T, 3)`.
- Every field is `nullable=False`: components contain no nulls, and an optional component is
  expressed by the column's absence.
- Everything without a row dimension -- `entity_path`, `frame_id`, `is_static`, timeline names and
  kinds, index times, `offsets`, `labels` -- goes into the **schema metadata** (JSON under the
  `b"t4perceval"` key). It cannot be a column because its length differs.
- Component classes are recorded by class name and resolved by `t4perceval.io.registry`, so no module
  paths get baked into stored files.

## Relationship to Rerun

We adopt Rerun's data model as a **design**, without depending on the SDK.

| Concept                         | Rerun                                        | t4perceval                                           |
| :------------------------------ | :------------------------------------------- | :--------------------------------------------------- |
| Entity / EntityPath             | yes                                          | yes (our own)                                        |
| Component / ComponentDescriptor | yes                                          | yes, but descriptor names are archetype-independent  |
| Archetype                       | builder / convenience helper                 | a type: a component bundle with a generic `select()` |
| Chunk                           | row = log call, cell = variable-length batch | row = object, `offsets` = frame boundaries           |
| Timeline / static / latest-at   | yes                                          | yes                                                  |
| AnnotationContext               | a static component                           | `LabelRegistry` (metadata)                           |
| System                          | formalization stays inside the Rust viewer   | first class, in `t4perceval.system`                  |

### Why not depend on it

- Rerun 0.36 has removed the read-query API (`rr.dataframe`), so reading an `.rrd` back as evaluation
  input is not on a stable footing. An evaluation tool needs both directions.
- Evaluation math wants flat "row = object" columns, which is a different premise from Rerun's chunk
  layout.
- Depending on it would put Rerun's version churn directly in the path of reproducible evaluation
  results.

`rerun-sdk` is present in the virtualenv as a `t4-devkit` dependency, but `t4perceval` does not import
it.

## Dataloader design

The path from `t4_devkit.T4Devkit` to a `Chunk`. Implemented as `t4perceval.importer.t4`, against
the `tests/data/t4dataset` fixture; the design below is what shipped.

```text
T4Devkit(data_root, revision)
  ├ get_box3ds(sample_data_token, future_seconds=...)  → list[Box3D]
  └ get_box2ds(sample_data_token)                      → list[Box2D]
        │
        ├─ SemanticLabel.name  ──→ LabelRegistry.encode()    → BatchClassId
        ├─ Box3D.uuid          ──→ InstanceRegistry.encode()  → BatchInstanceId
        ├─ position / rotation / shape.size / velocity / num_points / visibility
        │                      ──→ BatchPosition3D / BatchQuaternion / BatchSize3D / …
        └─ Box3D.future (Future: timestamps (T,), confidences (M,), waypoints (M,T,3))
                               ──→ BatchWaypoints3D / BatchModeConfidence / BatchTimeOffset
        │
        └→ Detections3D / Trackings3D / Predictions3D
              .to_chunk(entity_path, at=TimePoint.at(frame=i, timestamp_ns=box.unix_time * 1000),
                        frame_id=box.frame_id)
```

Points to handle:

- An empty annotation becomes a zero-row batch, which every archetype already allows.
- Velocity is a NaN vector, not a missing value, when the devkit cannot estimate it. Whether to
  emit the column is therefore a **scene-wide** decision, never a per-frame one: `concat_chunks`
  rejects chunks with different column sets, so a column present on one frame and absent on the
  next makes `Store.range()` raise.
- `Box3D.unix_time` is in microseconds and must be converted for the nanosecond `TIMESTAMP` timeline.
- One scene is one `Store`, with frame indices on the `FRAME` timeline.
- The dependency on `t4_devkit` stays inside the dataloader module; `t4perceval.core` uses only the
  local aliases in `t4perceval/typing.py`.

## Resolved open questions

Conclusions for the points left open in `TODO.md`.

| Question                                   | Decision                                                                                                  |
| :----------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| Fix trajectory `M` / `T`?                  | Fixed within a chunk; variable lengths use validity masks                                                 |
| Validate that `mode_confidence` sums to 1? | No; only range and finiteness                                                                             |
| Validate finite `waypoints`?               | Yes; invalid timesteps are expressed by the mask                                                          |
| Make component arrays read-only?           | Yes                                                                                                       |
| Allow zero-object batches?                 | Yes, for every archetype                                                                                  |
| What does `Selection` accept?              | slice / int array / bool array / int list / bool list; negative, duplicate and reversed indices allowed   |
| How do categories map to `BatchClassId`?   | `LabelRegistry`, carried as static metadata                                                               |
| A view class per archetype?                | No; one generic `EntityView`                                                                              |
| Put the coordinate frame in the path?      | No. Perception data keeps `Chunk.frame_id`; a transform states its parent there and its child in a column |
| Carry a transform as static data?          | When it is time-invariant, yes -- `static` means "not on a timeline", and a static write keeps its frame  |
| Intern frame names into a registry?        | No; a text column, because a frame column is O(edges) rather than O(objects)                              |
