# TODO

The data model and the system layer are designed and implemented. See [docs/design/](../design/)
for the design itself.

## Ground rules

- A component that holds several rows is prefixed with `Batch`; an archetype gets a meaning-based
  plural name.
- Descriptor names are archetype-independent (`POSITION` is `"position"` in both Detection3D and
  Tracking3D).
- `Component.select()` / `Archetype.select()` / `Chunk.select()` produce independent data. Lazy
  views are `EntityView`'s job.
- The NumPy arrays inside a component are read-only.
- A batch of zero objects is allowed in every archetype.

## Done

- [x] `core` — `EntityPath` / `ComponentDescriptor` / `ColumnarComponent` / `Archetype` /
      `Timeline` / `Chunk` / `Store` / `EntityView` / `normalize_selection`
- [x] Move archetypes from inheritance to composition. `select()` has one implementation, on the base
- [x] Dismantle `Header` (`TimePoint` + `Chunk.frame_id`)
- [x] Promote `Trajectories3D` from a component to an archetype and split out its columns
- [x] Add `BatchModeValid` / `BatchTimestepValid` / `BatchNumPoints` / `BatchVisibility` /
      `BatchRoi` / `BatchPixel` / `BatchMask` and the matching components
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
- [x] `README.md` states the purpose, the data shapes and a usage example
- [x] Update the `pyproject.toml` description

## P0: dataloader (or data importer)

Done. `t4perceval.importer.t4`, plus the `Recording` boundary the importers converge on.

- [x] Implement an importer on top of `t4_devkit.T4Devkit`. See "Dataloader design" in
      [data_model.md](../design/en/data_model.md).
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

Deferred deliberately: segmentation (no system consumes `PIXEL` / `POINT`, and
`SemanticSegmentation2D` carries no image width) and sensor data.

### Follow-ups

- [ ] `scan_t4_labels`-style discovery without opening the whole dataset, if load time
      becomes a problem on real scenes.
- [ ] Decide whether a camera `channel_3d` should ever be allowed. It is rejected today
      because `get_sample_data` silently drops boxes outside the image.

## P1: MCAP / ROS bag importer

The second source. Its output converges on the same `Recording`, but it shares no
converters with T4: `DetectedObjects` / `TrackedObjects` / `PredictedObjects` are three
distinct message schemas, unlike the T4 3D archetypes.

- [ ] `t4perceval.importer.rosbag`, decoding through the pure-Python `mcap` +
      `mcap-ros2-support` (the `rosbag` extra). MCAP is self-describing, so an Autoware bag
      decodes from its own embedded schemas — no ROS install, no Autoware message packages,
      and no version pinning against a project that releases independently.
- [ ] `AUTOWARE_CLASS_NAMES`: `ObjectClassification` is a `uint8` enum, so the mapping goes
      `enum -> canonical name -> registry id` in two visible stages rather than baking
      enum-to-class-id directly.
- [ ] Topic-to-entity-path mapping. A topic names a message source, an entity path names a
      semantic location; they are not the same concept.
- [ ] `t4perceval.align` — associate ground-truth and estimation frames by nearest
      timestamp within a tolerance, one-to-one, producing a shared `FRAME` index. Needed
      because matching takes the *union* of the two time sets, so mismatched stamps yield
      all-FP + all-FN frames instead of an error.
- [ ] Decide what `existence_probability` versus the per-class probability means for
      `BatchConfidence`, and how `Shape.dimensions` (x=length, y=width, z=height) maps onto
      `BatchSize3D` (width, length, height).

## P1: metric systems

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
- [ ] `HotaSystem` / `PassFailSystem` (including the critical-object verdict) — on hold
- [ ] Address the findings in [metrics.md](./metrics.md). The implementations differ from the
      official benchmark definitions as follows (all pass the current tests, but compatibility is
      unverified)
  - [ ] The prediction metrics do not consult `MODE_VALID` / `TIMESTEP_VALID` / `TIME_OFFSET`
        (padding leaks into ADE / FDE / MissRate, and the time axis is aligned by index)
  - [ ] APH treats heading similarity as the true-positive count itself, which also changes the
        denominator of recall (Waymo uses the ordinary true-positive count for recall)
  - [ ] AP association is a globally optimal Hungarian assignment rather than confidence order
        (nuScenes matches greedily in descending confidence)
  - [ ] An identity change across a missing frame is counted as an ID switch
        (`_count_switches()` only ever receives true-positive rows)
  - [ ] MOTA is clamped to 0 by `max(0.0, ...)` (CLEAR permits negative values). Under
        class-agnostic matching, a pair whose classes disagree becomes neither a true positive nor
        a false positive
  - [ ] `accuracy` is defined as `TP / (TP + FP + FN)` (Jaccard/IoU), which does not match its name;
        MissRate is the fraction of all mode × timestep distances over the tolerance, which differs
        from the per-object definition

## P1: coordinate transforms

Transforms are recorded data plus an explicit interpretation step, never hidden state.

- [x] Decide how to carry a transform. Not `HomogeneousMatrix`, and not static data:
      `Transform3D` (`translation` + `rotation`) at `/transforms/<parent>/<child>`, logged as
      ordinary temporal samples. Static data carries no `frame_id` and reads back as zero
      rows on an entity with no temporal partition, so a fixed extrinsic is one sample that
      `latest_at` reaches forward from.
- [x] Revisit the coordinate frame in the `EntityPath` hierarchy. Perception data keeps
      `Chunk.frame_id`; a transform *edge* lives in the path, because there the frame pair is
      the identity of the data. Recorded in [data_model.md](../design/en/data_model.md).
- [x] Refuse to compare geometry across frames. `require_same_frame()` in
      `t4perceval/system/base.py`, called from the matching base and from `MatchJoin`, so
      every matcher and every geometric metric is covered once. Two *different stated* frames
      raise; an unstated frame is not a disagreement. Opt out per system with
      `check_frames=False`.
- [x] Import the frame tree. The T4 importer records `map -> base_link` per keyframe from
      `ego_pose` and a fixed `base_link -> <channel>` per sensor from `calibrated_sensor`.
- [ ] `TransformResolver` — graph traversal, inversion, composition, and a lookup policy of
      `latest` / `exact` / `nearest` / `interpolate` (`Slerp` for rotation, lerp for
      translation; both are in the scipy already depended on).
- [ ] `TransformSystem` — materialize a transformed entity. Decisions already taken: velocity
      is rotated only and never translated, and the docstring must say that ignores relative
      motion between the frames; every waypoint of a chunk uses that chunk's own transform;
      `MASK` columns are dropped rather than carried, because a distance or region mask is a
      claim about the source frame. It writes a *separate* entity — `range()` refuses to
      concatenate chunks in different frames, so this is forced rather than stylistic.
  - [ ] Resolve the `PROVIDES` problem first: a passthrough system cannot enumerate the
        columns it carries, and declaring `()` makes `Pipeline` reject any consumer of its
        target. `ApplyMaskSystem` already has this wart. A sentinel that makes `_validate`
        propagate the source's contract would fix both.
- [ ] Sub-frame ego motion. Only keyframe poses are imported today, so a lookup between
      frames has no finer sample to find.

## P1: offline analysis

- [x] Decide the object that gets persisted. `Recording` is it — a store plus the label and
      instance registries plus `RecordingMetadata`. `EvaluationRecording` is dropped from the
      design: an evaluation recording is a `Recording` whose store also holds `/matching/*`
      and `/metrics/*`, which is a difference in content, not in type.
- [ ] Persist a whole `Recording` so results can be analyzed and visualized later.
  - [ ] `write_recording` / `read_recording`, following [offline_analysis.md](./offline_analysis.md).
  - [ ] Decide the directory and file layout a saved recording uses.
  - [ ] `InstanceRegistry.to_metadata` / `from_metadata` — still the one registry that
        cannot round-trip.

## P2: visualization

- [ ] Design a visualization layer whose input is a query against the store.
  - [ ] Decide between Rerun as an optional output sink and a matplotlib implementation of our own.
        →> use `t4_devkit.viewer.RerunViewer`.
  - [ ] [OPTIONAL] Check whether the analyses corresponding to the original repository's
        `perception_analyzer3d` / `eda_tool` / `field_analyzer` can be rewritten as store queries.

## P2: quality

- [ ] Run Pyright or Mypy in CI alongside Ruff.
- [ ] Set up a test matrix for Python 3.10 and later.
- [ ] Decide whether `Store` queries need a per-chunk cache, after measuring at real data scale.
- [ ] Measure `Store.range()` performance on a large scene (it currently builds a small chunk per
      partition and concatenates them).
- [ ] Add a changelog and record the migration to the Rerun-based data model.
