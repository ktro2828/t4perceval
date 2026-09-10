# Roadmap

What is next, and what is already done. The data model and the system layer are designed and
implemented; see the [design documents](./design/) for the design itself,
[Architecture](./architecture.md) for the layer map, and
[Design decisions](./design-decisions/index.md) for the individual calls.

## Direction

The next development phase prioritizes **architectural completeness and design stabilization**
over metric correctness and feature breadth. Stabilize the lower-level abstractions first, then
build task-specific and metric-specific functionality on top of them.

```text
Persistent Recording
        ↓
System / Transform Semantics
        ↓
Task Generalization
        ↓
Metric Semantics
```

1. [P0: Persistent recording](#p0-persistent-recording)
2. [P1: Transform completion](#p1-transform-completion)
3. [P2: Segmentation support](#p2-segmentation-support)
4. [P3: Metric correctness](#p3-metric-correctness)
5. [P4: Additional metrics and productization](#p4-additional-metrics-and-productization)

The immediate objective is **semantic and architectural stabilization rather than feature
accumulation**. The first major completion milestone:

> A complete evaluation can be stored as a `.t4eval` recording, reopened in another process,
> queried with the same `Store` semantics, transformed between coordinate frames, and used for
> multiple perception evaluation tasks without changing the core data model.

Once persistent recording, transform/system semantics and segmentation all work cleanly on the
current architecture, the core design is mature enough to freeze. Metric correctness and additional
metrics then evolve on top of a stable foundation.

## Ground rules

- A component that holds several rows is prefixed with `Batch`; an archetype gets a meaning-based
  plural name.
- Descriptor names are archetype-independent (`POSITION` is `"position"` in both Detection3D and
  Tracking3D).
- `Component.select()` / `Archetype.select()` / `Chunk.select()` produce independent data. Lazy
  views are `EntityView`'s job.
- The NumPy arrays inside a component are read-only.
- A batch of zero objects is allowed in every archetype.

## P0: Persistent recording

### Goal

Define a complete, persistent representation of an evaluation recording. `Recording` is already the
container for evaluation data:

```python
Recording(
    store=store,
    labels=labels,
    instances=instances,
    metadata=metadata,
)
```

The next step makes that object fully serializable and recoverable:

```python
write_recording(recording, "result.t4eval")

recording = read_recording("result.t4eval")
```

Design: [persistent-recordings.md](./persistent-recordings.md) and
[ADR 0004](./design-decisions/0004-persistent-recording.md).

### Why this comes first

Persistence is not merely an I/O feature. Implementing it forces us to state exactly:

- What constitutes a `Recording`
- What belongs in the `Store` and what belongs in `Recording` metadata
- Which state must survive serialization, and which is derived and can be recomputed
- How components and schemas are identified
- How insertion order is preserved
- How static and temporal data are represented
- How format evolution and versioning work

That is the persistent boundary of the t4perceval data model, and `.t4eval` is its canonical
artifact.

### Proposed first format

A directory-based format is enough for the first version:

```text
result.t4eval/
├── manifest.json
└── chunks/
    ├── 000000.parquet
    ├── 000001.parquet
    ├── 000002.parquet
    └── ...
```

The manifest maps chunks to their logical metadata rather than encoding `EntityPath` into the
filesystem hierarchy:

```json
{
  "format_version": 1,
  "chunks": [
    {
      "id": 0,
      "entity_path": "/ground_truth/objects",
      "file": "chunks/000000.parquet"
    }
  ]
}
```

This keeps the storage layout independent of `EntityPath` semantics.

### Required work

- [x] Define the `.t4eval` format, and the directory and file layout a saved recording uses
- [x] Define the manifest schema
- [x] Add format versioning, carried by the format itself and separate from the chunk
      `SCHEMA_VERSION`
- [x] `InstanceRegistry.to_metadata()` / `from_metadata()` — still the one registry that cannot
      round-trip
- [x] Preserve the label registry
- [x] Preserve entity paths
- [x] Preserve timelines
- [x] Preserve partition offsets
- [x] Preserve `frame_id`, on temporal and static chunks alike
- [x] Preserve static chunks
- [x] Preserve temporal chunk insertion order
- [x] Preserve evaluation-derived entities (`/matching/*`, `/metrics/*`) alongside the raw inputs
- [x] `write_recording()`
- [x] `read_recording()`
- [x] Full round-trip tests

The invariant to test:

```python
restored = read_recording(write_recording(recording))

# Semantically equivalent
restored.store == recording.store
```

In particular `latest_at()` must produce identical results before and after serialization — which
is what makes chunk insertion order part of the format rather than an implementation detail.

## P1: Transform completion

### Goal

Complete the coordinate transformation architecture on top of the existing `FrameGraph` and
`TransformResolver`. Lookup is already solved; the missing piece is applying a resolved transform to
an entity and materializing the result back into the `Store`.

```text
source entity
      │
      ▼
TransformEntitySystem
      │
      ▼
transformed entity
```

```python
TransformEntitySystem(
    source="/estimation/objects",
    target="/estimation/objects_map",
    target_frame="map",
)
```

Transforms are recorded data plus an explicit interpretation step, never hidden state.

### Design questions to finalize

#### System passthrough contracts

A transforming or masking system preserves most of the source's components, and cannot enumerate
them. `REQUIRES` / `PROVIDES` should support that explicitly rather than treating an empty
`PROVIDES` as an implicit passthrough — declaring `()` makes `Pipeline` reject any consumer of the
target. `ApplyMaskSystem` already carries this wart.

```python
PROVIDES = PassthroughFrom(SOURCE)
```

The exact API can differ, but passthrough semantics should be first-class: `_validate` propagates
the source's contract.

- [x] Resolve the `PROVIDES` problem — a sentinel that propagates the source contract, fixing
      `ApplyMaskSystem` at the same time

#### Component transformation semantics

Define which components are frame-dependent and how each transforms:

```text
Translation3D   → rotate + translate
Rotation3D      → compose rotation
Velocity3D      → rotate
Acceleration3D  → rotate
Trajectory      → transform each state
Size3D          → unchanged
ClassId         → unchanged
Score           → unchanged
InstanceId      → unchanged
```

Driven by component semantics rather than hard-coded per archetype.

- [x] Define the per-component transform semantics, keyed on the component rather than the
      archetype

Decisions already taken: velocity is rotated only and never translated, and the docstring must say
that this ignores relative motion between the frames; every waypoint of a chunk uses that chunk's
own transform; `MASK` columns are dropped rather than carried, because a distance or region mask is
a claim about the source frame.

#### Materialization

A transformed result is written as a new entity rather than mutating the source:

```text
/estimation/objects
        ↓
TransformEntitySystem
        ↓
/estimation/objects_map
```

This preserves provenance and fits the immutable, log-oriented `Store` model. It is also forced
rather than stylistic: `range()` refuses to concatenate chunks in different frames.

- [x] `TransformEntitySystem` — the plumbing that writes the resolver's answer back as an entity

### Expected outcome

Recordings in different coordinate frames become composable:

```text
Recording A
base_link
     │
     ├── transform ──→ common frame
     │
Recording B
map
     │
     └── transform ──→ common frame
                           │
                           ▼
                        matching
                           │
                           ▼
                         metrics
```

This validates the System architecture beyond evaluation-specific operations.

- [x] Validate cross-frame composition of two recordings end to end
- [x] A worked example of bringing two recordings into one coordinate frame
      ([recipe](../recipes/align-frames.md))

## P2: Segmentation support

### Goal

Complete segmentation evaluation, and use it as a test of whether the architecture generalizes
beyond object-based evaluation.

Detection, tracking and prediction operate on collections of objects and therefore use matching.
Segmentation does not:

```text
Ground Truth
     │
     ├──────────┐
     │          ▼
     │   Segmentation Metrics
     │          ▲
     └──────────┘
Estimation
```

An element-wise comparison needs no object matcher, which makes segmentation a useful architectural
stress test of the evaluation pipeline.

### Required work

- [x] Segmentation confusion statistics — `SegmentationConfusionMatrixSystem`
- [x] Per-class IoU — `SegmentationIoUSystem`, `/metrics/segmentation/iou`
- [x] Mean IoU — the `ALL_CLASSES` row of the same table, the family convention
- [x] Accuracy, where appropriate — class accuracy with its mean, and pixel accuracy as its own single-row entity
- [x] Label handling — the registry decides the class axis; names resolve through `SystemContext.labels`
- [x] Point order — a 3D model does not promise the ground truth's point order, so the metrics compare the `POINT` coordinates row by row and refuse a permuted cloud; `AlignPointsSystem` reorders the estimation by nearest neighbour as an explicit, inspectable stage
- [x] Invalid / ignored label handling — `UNKNOWN_CLASS_ID` ground truth is excluded by default, `ignore=` excludes more; an ignored class leaves the axis ("void" semantics); an unregistered ground-truth id raises
- [x] The component and archetype design for the metric outputs — the existing `MetricValues` and `ConfusionMatrix`; no new archetype was needed

Output entities:

```text
/metrics/segmentation/confusion_matrix
/metrics/segmentation/iou
/metrics/segmentation/accuracy
/metrics/segmentation/pixel_accuracy
```

### Design questions

How much is shared between `SemanticSegmentation2D` and `SemanticSegmentation3D`? The evaluation
logic should depend on semantic components rather than dimensionality wherever possible: if both
expose `CLASS_ID`, one metric implementation can operate on both.

- [x] Decide the 2D/3D sharing boundary — everything: both archetypes expose `CLASS_ID` and the row is the element, so one implementation serves both. `SemanticSegmentation2D` became `class_id` alone (the row is the pixel) with a static `IMAGE_SIZE`; `pixel` was removed
- [x] Segmentation metric documentation, and an
      [evaluation page](../evaluation/segmentation-3d.md) that stops saying "not implemented yet"

### Architectural purpose

This phase answers one question:

> Is t4perceval fundamentally a perception evaluation framework, or has the architecture
> accidentally become an object-matching framework?

If segmentation fits without special-casing the core abstractions, the ECS/System design has
generalized successfully.

### Follow-ups: loading segmentation data

The metrics and archetypes are done; nothing imports segmentation data yet. Every item below
needs **sensor data** -- the point-cloud or image files themselves, not just annotation tables --
which the importers have deliberately not read so far (see the T4 importer's deferrals under
Shipped). The vendored fixture `tests/data/t4dataset` _is_ a lidarseg dataset, so the first item can
be validated on real data immediately.

- [ ] **T4 lidarseg import.** Read the `lidarseg` table through `t4_devkit` (`filename` per
      `sample_data`), load the referenced point cloud and its per-point label file, and log a
      `SemanticSegmentation3D` per keyframe at `/ground_truth/points` with `frame_id` from the lidar
      channel and both `FRAME` and `TIMESTAMP` indexes, like the boxes. Decide the entity path
      convention next to `/ground_truth/objects` (`objects` vs `points`) and expose it as an
      `ImportOptions` field.
  - [ ] Map the dataset's label indices onto the `LabelRegistry` by name, with the dataset's
        ignore/noise index becoming `UNKNOWN_CLASS_ID` so the metrics exclude it by default. The
        registry is an _input_, shared with the box import, as everywhere else.
  - [ ] Point order and count must match the cloud the labels were written for; validate the label
        file's length against the point cloud and raise on a mismatch rather than truncate.
  - [ ] Keep `t4_devkit` and the point-cloud reader inside the importer module (the existing
        `import t4perceval` never-loads-it test must still hold). Check what the devkit offers for
        reading `.pcd.bin` before adding a dependency.
  - [ ] Decide whether the raw cloud without labels also lands as an entity (`/sensor/<channel>`);
        `POINT` on the segmentation entity already carries the geometry the metrics and
        `TransformEntitySystem` need, so this is only worth it for visualization.
- [ ] **Estimation side for lidarseg.** A model's output arrives as a per-point label array over a
      cloud that may be re-ordered or cropped; document (and if a common file format emerges,
      import) how to log it as `SemanticSegmentation3D`, and point at `AlignPointsSystem` /
      `FilterByCoverageSystem` for the order and coverage questions.
- [ ] **Rosbag `PointCloud2` with a label field.** Decode a segmentation topic (a cloud whose fields
      include a per-point class, as Autoware's lidar segmentation nodes publish) into
      `SemanticSegmentation3D` on `TIMESTAMP`, reusing the `mcap` decoding path and the
      `AUTOWARE_CLASS_NAMES`-style enum → name → registry-id mapping. Bags are large and clouds are
      dense, so this is also where the per-frame `range()` cost of the segmentation metrics gets
      measured on real sizes.
- [ ] **2D label images.** Import a camera segmentation mask as `SemanticSegmentation2D.from_label_map`
      with the static `IMAGE_SIZE`, `frame_id` = camera channel, from whatever the dataset provides
      (T4 has no 2D segmentation table today; a bag `Image` topic with class ids is the likely
      source). Confirm on a real image that a 2M-pixel entity per frame is acceptable in memory and
      in `.t4eval` size, or decide that 2D needs run-length / dense-image storage after all.
- [ ] **End-to-end check on the fixture.** Once the T4 import exists, extend `examples/evaluate_t4.py`
      to score a synthetic estimation against the fixture's real lidarseg labels -- the example
      currently prints that the dataset has no boxes and stops.

## P3: Metric correctness

### Goal

Once the architecture is stable, align the metric implementations with their canonical definitions
and fix the metric semantics. `benchmarks/compare.py` runs the metric systems and `perception_eval`
1.3.6 on the same synthetic scenes and reports speed, memory and per-metric agreement; the
divergences are classified in [metric-divergences.md](./metric-divergences.md).

### Guiding principle

The canonical implementation follows the actual metric or benchmark definition rather than
preserving legacy behaviour for compatibility alone:

```text
Official / canonical definition
             │
             ▼
t4perceval implementation
             │
             ▼
optional compatibility behavior
```

rather than:

```text
legacy perception_eval behavior
             │
             ▼
t4perceval behavior
```

### Suggested order

1. **Prediction validity and time offsets.** The prediction metrics do not consult `MODE_VALID` /
   `TIMESTEP_VALID` / `TIME_OFFSET`, so padding leaks into ADE / FDE / MissRate and the time axis is
   aligned by index.
2. **APH.** Heading similarity is treated as the true-positive count itself, which also changes the
   denominator of recall; Waymo uses the ordinary true-positive count for recall.
3. **AP confidence-order association.** Association is a globally optimal Hungarian assignment
   rather than confidence order; nuScenes matches greedily in descending confidence.
4. **CLEAR ID-switch semantics.** An identity change across a missing frame is counted as an ID
   switch (`_count_switches()` only ever receives true-positive rows).
5. **MOTA semantics.** MOTA is clamped to 0 by `max(0.0, ...)`, while CLEAR permits negative
   values. Under class-agnostic matching, a pair whose classes disagree becomes neither a true
   positive nor a false positive.
6. **Classification and miss-rate cleanup.** `accuracy` is defined as `TP / (TP + FP + FN)`
   (Jaccard/IoU), which does not match its name; MissRate is the fraction of all mode × timestep
   distances over the tolerance, which differs from the per-object definition.
7. **Comprehensive regression tests**, pinned to the canonical definitions rather than to the
   current output.

- [ ] Prediction validity masks and time offsets
- [ ] APH
- [ ] AP confidence-order association
- [ ] CLEAR ID-switch semantics
- [ ] MOTA semantics
- [ ] Classification and miss-rate definitions
- [ ] Regression tests over all of the above

After this phase, metric semantics can be considered stable.

## P4: Additional metrics and productization

Lower-priority functionality, once the core design has stabilized. None of it should drive changes
to the core architecture unless it exposes a genuine abstraction problem.

### Additional metrics

- [ ] `HotaSystem`
- [ ] `PassFailSystem`, including the critical-object verdict
- [ ] Additional segmentation metrics
- [ ] Task-specific aggregate metrics

### Visualization

Visualization comes after persistent recordings, and t4perceval should not become a visualization
framework: provide a thin integration layer over an existing tool.

```text
evaluation
    │
    ▼
result.t4eval
    │
    ▼
read_recording()
    │
    ▼
Store queries
    │
    ▼
RerunViewer / other visualization
```

- [ ] A visualization layer whose input is a query against the store, integrating
      `t4_devkit.viewer.RerunViewer` rather than implementing a viewer
- [ ] [OPTIONAL] Check whether the analyses corresponding to the original repository's
      `perception_analyzer3d` / `eda_tool` / `field_analyzer` can be rewritten as store queries

### Quality and release

Before the core API is called stable:

- [ ] Python version CI matrix (3.10 and later)
- [ ] Pyright or Mypy in CI alongside Ruff
- [ ] Execute the documentation examples in CI, so a signature change breaks the build rather than
      the reader
- [ ] Large-scene `Store` benchmarks — `Store.range()` currently builds a small chunk per partition
      and concatenates them
- [ ] Decide whether `Store` queries need a per-chunk cache, after measuring at real data scale
- [ ] Persistence compatibility tests across format versions
- [ ] A versioning policy
- [ ] A changelog, recording the migration to the Rerun-based data model
- [ ] The first versioned release

## Non-goals for now

Do not redesign already-stable concepts unless the work above exposes a concrete problem. In
particular, do not proactively redesign `Store`, `Archetype`, `Component`, `EntityPath`, `FrameId`,
`FrameGraph`, the importer architecture, or the core ECS-like data model.

Also postpone the high-level offline-analysis abstractions sketched in
[persistent-recordings.md](./persistent-recordings.md) §4:

```python
analysis.errors(...)
analysis.object_history(...)
analysis.filter_failures(...)
```

until actual usage patterns emerge. For now:

```python
recording = read_recording("result.t4eval")

recording.store.range(...)
recording.store.latest_at(...)
```

remains the fundamental offline-analysis interface. Convenience APIs come later, for the queries
that prove common.

## Deferred follow-ups

Real but unscheduled; none of them blocks a phase above.

- [ ] `scan_t4_labels`-style discovery without opening the whole dataset, if load time becomes a
      problem on real scenes.
- [ ] Decide whether a camera `channel_3d` should ever be allowed. It is rejected today because
      `get_sample_data` silently drops boxes outside the image.
- [ ] Ground-truth interpolation to the estimation stamp (the incumbent's
      `interpolate_ground_truth`: position / velocity lerp, orientation slerp, shape copied), and an
      order-preserving assignment for the rare case where nearest-wins leaves a pair on the table.
- [ ] Rosbag decoding is pure Python: a 60 s Autoware bag imports a detection topic in ~10 s and a
      `PredictedObjects` topic (up to 100 poses per path) in ~60 s. If that matters, batch `/tf`
      samples per child into one multi-partition chunk and decode with a CDR reader that returns
      arrays.
- [ ] Sub-frame ego motion. Only keyframe poses are imported today, so a lookup between frames has
      no finer sample to find.

## Shipped

### Core and system layer

- [x] `core` — `EntityPath` / `ComponentDescriptor` / `ColumnarComponent` / `Archetype` /
      `Timeline` / `Chunk` / `Store` / `EntityView` / `normalize_selection`
- [x] Move archetypes from inheritance to composition. `select()` has one implementation, on the base
- [x] Dismantle `Header` (`TimePoint` + `Chunk.frame_id`)
- [x] Promote `Trajectories3D` from a component to an archetype and split out its columns
- [x] Add `BatchModeValid` / `BatchTimestepValid` / `BatchNumPoints` / `BatchVisibility` /
      `BatchRoi` / `BatchMask` and the matching components
- [x] `LabelRegistry` / `InstanceRegistry` (the category ↔ `BatchClassId` correspondence)
- [x] Arrow IO as public API. Nested vectors are fixed-size lists, non-row-wise information lives in
      schema metadata. Parquet round-trip verified
- [x] Add `pyarrow` as a direct dependency
- [x] System protocol / `SystemContext` / `Pipeline` (order validation)
- [x] The filter systems — a shared `MaskSystem` base plus 8 kinds
      (`FilterByDistance` / `Region` / `Label` / `Confidence` / `Instance` / `Speed` /
      `NumPoints` / `Visibility`)
- [x] `CombineMasksSystem` (`mode="all"` / `"any"`)
- [x] `masked_view()` — a lazy view of the rows that passed a mask
- [x] `SystemContext.instances` and `InstanceRegistry.instance_id()` (a reference that does not
      intern)
- [x] The matching systems — a shared `MatchingSystem` base plus 6 modes
      (`CenterDistance` / `CenterDistanceBEV` / `PlaneDistance` / `IoUBEV` / `IoU3D` / `IoURoi`)
- [x] `t4perceval.geometry` — vectorized box geometry (footprint vertices, BEV/3D IoU, ROI IoU,
      plane distance). `shapely` added as a direct dependency
- [x] Per-class thresholds — matching takes `Thresholds(default, by_class=...)`, keyed by the
      ground-truth class. Filters express the same thing by composition through
      `CombineMasksSystem`, because not every filter system requires `CLASS_ID`
- [x] Design documents (ja / en) — data_model / system / migration
- [x] `README.md` states the purpose, the supported tasks, and a usage example
- [x] Update the `pyproject.toml` description
- [x] Documentation organized by user intent — getting started, concepts, user guide, evaluation
      tasks, recipes, the component and archetype catalogues, a generated API reference, and this
      development section

### T4 dataset importer

Done. `t4perceval.importer.t4`, plus the `Recording` boundary the importers converge on.

- [x] Implement an importer on top of `t4_devkit.T4Devkit`. See "Dataloader design" in
      [data_model.md](./design/en/data_model.md).
  - [x] Load by dataset root / revision.
  - [x] Narrow by scene, sample and sensor channel.
  - [x] Convert `Box3D` into `Detections3D` / `Trackings3D` / `Predictions3D` and `Box2D`
        into `Detections2D` / `Trackings2D`. One extraction, one projection per archetype:
        the 3D archetypes are a nested superset chain over the same annotation rows.
  - [x] Convert `Box3D.unix_time` (μs) to the `TIMESTAMP` timeline (ns).
  - [x] Handle empty annotations, missing velocity and invalid sample data.
  - [x] Keep the dependency on `t4_devkit` inside the importer module — enforced by a test
        asserting `import t4perceval` never loads it, and by making it an optional extra.
- [x] Prepare a minimal T4 dataset fixture and validate against the real data format.
      `tests/data/t4dataset`, 84 KB of annotation tables vendored from `t4-devkit`.
- [x] `Recording` / `RecordingMetadata` — a log bound to the registries that encoded it.
- [x] `t4perceval.evaluation` — materializing recordings into a runnable store, with the
      class-id, coordinate-frame and instance-registry agreement checks.
- [x] `t4perceval.reconcile` — expressing one registry's class ids in another's.

Deferred deliberately: segmentation (no system consumed `POINT`, and
`SemanticSegmentation2D` carried no image width — resolved in [P2](#p2-segmentation-support)) and
sensor data.

### MCAP / ROS bag importer

The second source. Its output converges on the same `Recording`, but it shares no
converters with T4: `DetectedObjects` / `TrackedObjects` / `PredictedObjects` are three
distinct message schemas, unlike the T4 3D archetypes.

- [x] `t4perceval.importer.rosbag`, decoding through the pure-Python `mcap` +
      `mcap-ros2-support` (the `rosbag` extra). MCAP is self-describing, so an Autoware bag
      decodes from its own embedded schemas — no ROS install, no Autoware message packages,
      and no version pinning against a project that releases independently. Only `ros2msg`
      schemas decode; a `ros2idl` bag (some AWSIM recordings) is refused with a message
      naming the encoding.
- [x] `AUTOWARE_CLASS_NAMES`: `ObjectClassification` is a `uint8` enum, so the mapping goes
      `enum -> canonical name -> registry id` in two visible stages rather than baking
      enum-to-class-id directly. The name -> id step moved to `importer/_labels.py`, shared
      with T4.
- [x] Topic-to-entity-path mapping. A topic names a message source, an entity path names a
      semantic location; they are not the same concept. One recording holds one topic at
      `/estimation/objects`; the topic is provenance (`SourceInfo.topic`).
- [x] `t4perceval.align` — associate ground-truth and estimation frames by nearest
      timestamp within a tolerance, one-to-one, producing a shared `FRAME` index. Needed
      because matching takes the _union_ of the two time sets, so mismatched stamps yield
      all-FP + all-FN frames instead of an error. Decided: **ground-truth driven** — each
      reference frame takes its nearest estimation frame within 75 ms (the incumbent's
      `threshold_min_time`), conflicts go to the smaller distance, leftover estimation frames
      leave the evaluated set; unmatched reference frames stay as all-FN by default
      (`unmatched_reference="drop"` for the incumbent's skip); `offset_ns` for clock skew.
      The greedy nearest pairing is the stable matching, not the largest one.
      `build_evaluation_store(..., align=AlignOptions())` runs it and tags the setup.
- [x] Decided: `BatchConfidence` is the top classification's probability by default (what a
      detector's score becomes, and what driving_log_replayer used), with
      `ImportOptions(confidence="existence" | "product")` as alternatives; an object with no
      classification falls back to `existence_probability`. `Shape.dimensions` `(x, y, z)`
      maps to `BatchSize3D` `(y, x, z)`; a `CYLINDER` is `(x, x, z)`; a `POLYGON` footprint
      is dropped (no component). A body-frame twist is rotated into `header.frame_id` so
      `BatchVelocity` shares `position`'s frame.
- [x] `/tf` samples are logged on `TIMESTAMP` only -- no frame index exists between messages
      -- and `/tf_static` with `log_static`; a child on both topics, or with two parents, is
      an error because a child is filed under one entity. Default `tf_scope="selection"`
      keeps the samples spanning the imported messages plus one bracket sample either side.

### Metric systems

- [x] `MeanAveragePrecisionSystem` (mAP / APH) — `AveragePrecisionSystem` /
      `AveragePrecisionHeadingSystem` / `MeanAveragePrecisionSystem`
- [x] `ClearSystem` (MOTA / MOTP / IDSwitch)
- [x] `PathDisplacementSystem` (ADE / FDE / MissRate)
- [x] `ClassificationSystem` (accuracy / precision / recall / F1)
- [x] `ConfusionMatrixSystem` (the between-class confusion matrix)
- [x] Decide the `/metrics/*` chunk schema — a scalar metric is fixed at the four `MetricValues`
      columns (`class_id` / `threshold` / `value` / `support`) and the metric's name is carried by
      the entity path (`/metrics/<name>`). A metric with structure defines its own archetype
      (`ConfusionMatrix`) and reuses the same source wiring

Correctness against the official definitions is [P3](#p3-metric-correctness).

### Coordinate transforms

- [x] ~~Decide how to carry a transform. Not `HomogeneousMatrix`, and not static data:
      `Transform3D` (`translation` + `rotation`) at `/transforms/<parent>/<child>`, logged as
      ordinary temporal samples, because a static-only entity reads back as zero rows.~~
- [x] ~~Revisit the coordinate frame in the `EntityPath` hierarchy. A transform _edge_ lives in
      the path, because there the frame pair is the identity of the data.~~
- [x] **Reversed, both of the above.** A transform is ROS-shaped: parent in `Chunk.frame_id`,
      child in a `child_frame_id` component, neither in the entity path. The path version
      could not express a frame name containing `/`, and tied the graph to where it was
      filed. `static` now means only "not on a timeline", so a calibration is `log_static`
      and an ego pose is `log` -- same archetype. Recorded in
      [data_model.md](./design/en/data_model.md); the old claims are struck through there
      rather than deleted.
- [x] Mono components. `Transform3D` describes one relationship, not N objects, so its
      fields are `Position3D` / `Quaternion` / `FrameId` -- values, not columns, with
      `.value` / `.name` accessors and no row to index. `MonoComponent` is a boundary type:
      `Archetype.as_components()` widens it into its `BATCH` counterpart, because
      `Store.range` concatenates partitions and a type permitting exactly one row could not
      be the stored column. Materializing a multi-row view as a `Transform3D` raises instead
      of taking the first row.
- [x] Make static data retain its chunk. `Store._static` holds whole chunks, so a static write
      keeps its `frame_id` (and offsets, and `is_static`); `static_chunks()` /
      `static_frame_id()` reach them while `static()` still returns folded columns. Static rows
      deliberately do **not** surface through `latest_at` / `range` -- that would invent rows in
      empty frames and hand index-less chunks to systems that ask a view for its times -- and
      `EntityView.frame_id` still reports the temporal chunk's frame only. This is also a
      prerequisite for `write_recording`, which would otherwise have nothing to write.
- [x] Refuse to compare geometry across frames. `require_same_frame()` in
      `t4perceval/system/base.py`, called from the matching base and from `MatchJoin`, so
      every matcher and every geometric metric is covered once. Two _different stated_ frames
      raise; an unstated frame is not a disagreement. Opt out per system with
      `check_frames=False`.
- [x] Import the frame tree. The T4 importer records `map -> base_link` per keyframe from
      `ego_pose` at `/tf/base_link`, and a **static** `base_link -> <channel>` per sensor from
      `calibrated_sensor` at `/tf/<channel>` -- one row per sensor, so the tree costs nothing
      per frame and does not depend on which frames were imported. Two calibrations for one
      channel raise unless they agree, and one naming a sensor the `sensor` table does not list
      raises as well.
- [x] Find the graph from the data. `transform_edges()` / `FrameGraph` read `(chunk.frame_id,
child_frame_id)` out of the chunks, so nothing parses an entity path. A chunk without the
      column is ignored; one that names a child but states no `frame_id` raises, as does the
      same `(parent, child)` recorded twice.
- [x] `TransformResolver` — breadth-first traversal, exact inversion of an edge walked
      backwards, composition, and `LookupPolicy.LATEST` / `EXACT` / `NEAREST` / `INTERPOLATE`
      (`Slerp` for rotation, lerp for translation). Static edges ignore the policy. Returns a
      one-row `Transform3D`, not a matrix, so it can be logged straight back. Not a `System`:
      it answers a question rather than producing chunks.

Applying a transform to an entity is [P1](#p1-transform-completion).

### Offline analysis

- [x] Decide the object that gets persisted. `Recording` is it — a store plus the label and
      instance registries plus `RecordingMetadata`. `EvaluationRecording` is dropped from the
      design: an evaluation recording is a `Recording` whose store also holds `/matching/*`
      and `/metrics/*`, which is a difference in content, not in type.

Writing and reading one back is [P0](#p0-persistent-recording).
