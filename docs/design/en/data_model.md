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

```
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
    component: str                    # identity is this field alone
    archetype: str | None             # eq=False; a provenance hint
    component_type: str | None        # eq=False; the component class name
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
    SHAPE = (3,)            # per-row shape; () is scalar, ANY is inferred from the data
    DTYPE = np.float64      # every value is coerced to this dtype
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
    position    = component_field(POSITION,    BatchPosition3D)
    quaternion  = component_field(QUATERNION,  BatchQuaternion)
    size        = component_field(SIZE,        BatchSize3D)
    class_id    = component_field(CLASS_ID,    BatchClassId)
    confidence  = component_field(CONFIDENCE,  BatchConfidence)
    instance_id = component_field(INSTANCE_ID, BatchInstanceId, kw_only=True)
    velocity    = component_field(VELOCITY,    BatchVelocity, optional=True, kw_only=True)
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
tracking.has(*Detections3D.required_descriptors())   # True
isinstance(tracking, Detections3D)                   # False
```

`has()` asks exactly the question a system asks through `REQUIRES`, and is the correct replacement for
the `isinstance` check.

### What the base provides

`select()` walks `attrs.fields()` and delegates to each component, so **no archetype implements it**.
`as_components()`, `from_components()`, `to_chunk()`, `from_chunk()`, `has()`, `descriptors()` and
`required_descriptors()` are likewise single implementations on the base.

### The archetypes

| Archetype                     | Components                                                                                       |
| :---------------------------- | :----------------------------------------------------------------------------------------------- |
| `Detections3D`            | position, quaternion, size, class_id, confidence, [velocity], [num_points], [visibility]         |
| `Detections2D`            | roi, class_id, confidence, [visibility]                                                          |
| `Trackings3D`             | Detection3D's columns + instance_id                                                              |
| `Trackings2D`             | Detection2D's columns + instance_id                                                              |
| `Predictions3D`           | Tracking3D's columns + waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset] |
| `Classifications2D`       | class_id, confidence, [instance_id]                                                              |
| `SemanticSegmentation2D` | pixel, class_id                                                                                  |
| `SemanticSegmentation3D` | point, class_id                                                                                  |
| `Trajectories3D`           | waypoints, mode_confidence, [mode_valid], [timestep_valid], [time_offset]                        |
| `MatchResults`            | est_index, gt_index, matching_score, match_status                                                |

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
    SEQUENCE   # a monotonic counter, such as a frame index
    TIMESTAMP  # nanoseconds since the Unix epoch
    DURATION   # elapsed nanoseconds

FRAME     = Timeline("frame", TimeKind.SEQUENCE)
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

> **Future option**: Rerun expresses coordinate frames through the entity-path hierarchy plus a
> `Transform3D` archetype (for example `/base_link/estimation/objects`). Worth revisiting when a
> transform system is introduced. For now `Chunk.frame_id` is what we use.

## Chunk — a column-oriented table

```python
@define(frozen=True, slots=True)
class Chunk:
    entity_path: EntityPath
    indexes: tuple[TimeColumn, ...]                    # length P (partitions)
    offsets: NDArrayI64                                # length P+1, row boundaries
    columns: dict[ComponentDescriptor, Component]      # each of length N = offsets[-1]
    frame_id: str | None = None
    is_static: bool = False
```

### Why a row is an object

A rerun chunk has one row per log call, with each cell holding a variable-length component batch.
`t4perceval` flattens instead: **one row is one object**, and `offsets` marks the frame boundaries.

The reason is that evaluation math is elementwise over objects -- distance matrices, IoU, TP/FP
verdicts -- and that maps directly onto NumPy over flat contiguous arrays, with no list offsets to
walk per frame. `offsets` still makes per-frame aggregation cheap (`partition(i)`, `partition_ids()`).

```
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
store.log_static(entity_path, archetype)

store.latest_at(entity_path, timeline=FRAME, at=12)                    # → EntityView
store.range(entity_path, timeline=FRAME, time_range=TimeRange(0, 99))  # → EntityView
```

This replaces the `Catalog → Scenario → Scene → List[PerceptionFrameResult]` nesting.

| Old structure         | New expression                             |
| :-------------------- | :----------------------------------------- |
| One frame             | `store.latest_at(...)`                     |
| One scene             | `store.range(..., TimeRange.everything())` |
| Shared by every frame | `store.log_static(...)`                    |

### Semantics (following Rerun)

- **Static data belongs to every timeline** and takes precedence over temporal data carrying the same
  descriptor. A one-row static column is broadcast across the view.
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
    indices: NDArrayI64                              # normalized row indices into the chunk
    static: dict[ComponentDescriptor, Component]
```

`select()` only composes indices and **copies nothing**. Materialization happens in `component()`,
`materialize()` or `to_chunk()`.

```python
view.select(slice(None, None, 2)).select([1])   # no copy
view.component(POSITION)                        # copies exactly one column
view.materialize(Detections3D)              # materializes as an archetype
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
labels.class_id("truck")            # 1
labels.encode(["car", "truck"])     # an i32 column for BatchClassId

merged = labels.merged({"vehicle": ["car", "truck"]})
merged.class_id("car") == merged.class_id("vehicle")   # True
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

## Dataloader design (a later step)

The path from `t4_devkit.T4Devkit` to a `Chunk`. Implementation waits until a minimal T4 dataset
fixture is available.

```
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

| Question                                   | Decision                                                                                                |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| Fix trajectory `M` / `T`?                  | Fixed within a chunk; variable lengths use validity masks                                               |
| Validate that `mode_confidence` sums to 1? | No; only range and finiteness                                                                           |
| Validate finite `waypoints`?               | Yes; invalid timesteps are expressed by the mask                                                        |
| Make component arrays read-only?           | Yes                                                                                                     |
| Allow zero-object batches?                 | Yes, for every archetype                                                                                |
| What does `Selection` accept?              | slice / int array / bool array / int list / bool list; negative, duplicate and reversed indices allowed |
| How do categories map to `BatchClassId`?   | `LabelRegistry`, carried as static metadata                                                             |
| A view class per archetype?                | No; one generic `EntityView`                                                                            |
