# Migration Guide

How `autoware_perception_evaluation` (`perception_eval`) maps onto `t4perceval`.

## Types

| Old (`perception_eval`)                                    | New (`t4perceval`)                                                                                                                                | Notes                                                                   |
| :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------- |
| `DynamicObject`                                            | one row of `Detections3D` / `Trackings3D` / `Predictions3D`                                                                                       | a 20+ field class split into per-task component bundles                 |
| `DynamicObject2D`                                          | one row of `Detections2D` / `Trackings2D` / `Classifications2D`                                                                                   | `roi` becomes `BatchRoi`                                                |
| `Shape` / `ShapeType`                                      | `BatchSize3D`                                                                                                                                     | footprint computation moves into the matching systems                   |
| `FrameGroundTruth`                                         | one partition of `/ground_truth/objects`                                                                                                          | `unix_time` → the `TIMESTAMP` timeline                                  |
| `FrameGroundTruth.raw_data`                                | a separate entity path (`/sensor/<channel>`)                                                                                                      | not implemented yet                                                     |
| `FrameID` enum                                             | `Chunk.frame_id`, a plain string                                                                                                                  | systems compare it; they never branch on it                             |
| `HomogeneousMatrix` / a transform passed as an argument    | `Transform3D` rows plus `TransformResolver.lookup()`                                                                                              | recorded data, not state; parent is `Chunk.frame_id`, child is a column |
| `Catalog` / `Scenario` / `Scene`                           | `Store` plus a `TimeRange` on a timeline                                                                                                          | the list nesting is gone                                                |
| `PerceptionFrameResult`                                    | `store.latest_at(...)` plus the `/matching/*` chunk                                                                                               | no per-frame recomputation                                              |
| `DynamicObjectWithPerceptionResult`                        | `MatchResults`                                                                                                                                    | row indices, not references; storable and re-analysable                 |
| `MatchingMode` enum                                        | the matching system's type (`CenterDistance`, `CenterDistanceBEV`, `PlaneDistance`, `IoUBEV`, `IoU3D`, `IoURoi`) plus the `/matching/<mode>` path | all six modes are implemented                                           |
| `MatchingMethod` (`CenterDistanceMatching`, …)             | a `BatchMatchingScore` column                                                                                                                     | the entity path carries what the value means                            |
| `EvaluationTask` enum                                      | the set of components present plus the `Pipeline`                                                                                                 | the branching disappears                                                |
| `PerceptionEvaluationConfig`                               | each system's parameters plus `LabelRegistry`                                                                                                     |                                                                         |
| `evaluation_config_dict`                                   | a dataclass per system                                                                                                                            | validated at construction time                                          |
| threshold lists such as `center_distance_thresholds`       | `Thresholds(default, by_class=...)`                                                                                                               | keyed by class, so it says what it applies to                           |
| `MetricsScoreConfig`                                       | each metric system's parameters                                                                                                                   |                                                                         |
| `MetricsScore`                                             | a `/metrics/*` chunk                                                                                                                              |                                                                         |
| `LabelConverter` / `label_prefix` / `merge_similar_labels` | `LabelRegistry` / `LabelRegistry.merged()`                                                                                                        | the merge is visible in the data                                        |
| `Visibility`                                               | `BatchVisibility` plus `VisibilityLevel`                                                                                                          | an ordered integer instead of a string enum                             |
| the `objects_filter` functions                             | the `Filter*System` family                                                                                                                        | emits a `BatchMask` instead of dropping rows                            |
| the `object_matching` functions                            | the `*MatchingSystem` family                                                                                                                      | the geometry is vectorized in `t4perceval.geometry`                     |
| `PassFailResult` / `CriticalObjectFilterConfig`            | `PassFailSystem` plus critical-object filter systems                                                                                              |                                                                         |
| `PerceptionEvaluationManager`                              | `Pipeline` plus `SystemContext`                                                                                                                   |                                                                         |
| `perception_analyzer3d` / `visualization/`                 | queries against the store (a visualization layer comes later)                                                                                     | possible precisely because intermediates are kept                       |
| `class_to_dict` + `json.dump`                              | `write_parquet` / `chunk_to_table`                                                                                                                | dtypes and shapes are pinned by the schema                              |

## Concepts

| Old concept                                 | New concept                                                |
| :------------------------------------------ | :--------------------------------------------------------- |
| a field on an object                        | a component (a column)                                     |
| a class per task                            | an archetype (a bundle of components)                      |
| estimation vs ground truth (argument order) | the entity path (`/estimation/...` vs `/ground_truth/...`) |
| `frame_id` (coordinate frame)               | `Chunk.frame_id`                                           |
| an ego pose or a sensor extrinsic           | a `Transform3D` row: `frame_id` -> `child_frame_id`        |
| `unix_time`                                 | a value on the `TIMESTAMP` timeline                        |
| a frame number                              | a value on the `FRAME` timeline                            |
| walking a `List[FrameResult]`               | `store.range(...)`                                         |
| intermediate results (discarded)            | chunks under `/` (kept)                                    |

## Code

### Building objects

```python
# Before
obj = DynamicObject(
    unix_time=1624164470849887,
    frame_id=FrameID.BASE_LINK,
    position=(1.0, 2.0, 3.0),
    orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
    shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=(1.0, 4.0, 2.0)),
    velocity=(1.0, 0.0, 0.0),
    semantic_score=0.9,
    semantic_label=Label(AutowareLabel.CAR, "car"),
    uuid="c28556c19064ad491ff1dc438a38a3a7",
)
objects = [obj, ...]
```

```python
# After -- every object in a frame becomes one columnar batch
labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian", "motorbike"])
instances = InstanceRegistry()

detections = Trackings3D(
    position=[[1.0, 2.0, 3.0], ...],
    quaternion=[[0.0, 0.0, 0.0, 1.0], ...],  # xyzw
    size=[[1.0, 4.0, 2.0], ...],  # width, length, height
    class_id=labels.encode(["car", ...]),
    confidence=[0.9, ...],
    instance_id=instances.encode(["c28556c1...", ...]),
    velocity=[[1.0, 0.0, 0.0], ...],
)

store.log(
    "/estimation/objects",
    detections,
    at=TimePoint.at(frame=0, timestamp_ns=1_624_164_470_849_887_000),
    frame_id="base_link",
)
```

Note that `orientation` changes from a `wxyz` `Quaternion` to an **`xyzw`-ordered array**;
`BatchQuaternion` follows SciPy's convention.

### Running an evaluation

```python
# Before
manager = PerceptionEvaluationManager(evaluation_config=config)
for frame in frames:
    manager.add_frame_result(
        unix_time=frame.unix_time,
        ground_truth_now_frame=frame.gt,
        estimated_objects=frame.est,
        critical_object_filter_config=critical_config,
        frame_pass_fail_config=pass_fail_config,
    )
scene_score = manager.get_scene_result()
```

```python
# After
pipeline = Pipeline(
    [
        FilterByDistanceSystem.on("/ground_truth/objects", max_distance=102.4),
        CenterDistanceMatchingSystem.between(
            "/estimation/objects",
            "/ground_truth/objects",
            threshold=1.0,
        ),
    ]
)
pipeline.run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

# The whole scene
scene = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.everything(),
).materialize(MatchResults)
scene.num_tp, scene.num_fp, scene.num_fn

# A single frame -- from the same store, with no recomputation
frame_1 = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.single(1),
).materialize(MatchResults)
```

### Filtering objects

```python
# Before -- the rows vanish, so why they vanished does not survive
filtered = filter_objects(objects, is_gt=True, max_distance=102.4, target_labels=labels)
```

```python
# After -- each filter's verdict stays, as a child of the entity
src = "/ground_truth/objects"
region = FilterByRegionSystem.symmetric(src, max_xy=(102.4, 102.4))
label = FilterByLabelSystem.on(src, labels=["car", "bicycle", "pedestrian", "motorbike"])
points = FilterByNumPointsSystem.on(src, min_num_points=5)
keep = CombineMasksSystem.of(
    [region.target, label.target, points.target],
    f"{src}/filter/keep",
    mode="all",
)

ctx = SystemContext(store, FRAME, labels=labels, instances=instances)
Pipeline([region, label, points, keep]).run(ctx, TimeRange.everything())

# You can ask which filter dropped a row, one filter at a time
store.range(region.target, timeline=FRAME, time_range=TimeRange.everything()).component(MASK)

# And take only the rows that passed, as a lazy view
passed = masked_view(
    store,
    src,
    keep.target,
    timeline=FRAME,
    time_range=TimeRange.everything(),
).materialize(Detections3D)
```

### Type checks

```python
# Before
if isinstance(obj, DynamicObject) and obj.uuid is not None:
    ...  # treat it as tracking
```

```python
# After -- ask whether the component is there
if batch.has(INSTANCE_ID):
    ...
# "can this be treated as a detection?"
if batch.has(*Detections3D.required_descriptors()):
    ...
```

`isinstance(tracking, Detections3D)` is now **False**, because the inheritance was dropped. That is
intended; see "Why inheritance was dropped" in [data_model.md](data_model.md).

### Saving results

```python
# Before
dict_result = class_to_dict(manager.frame_results)
json.dump(dict_result, f)
```

```python
# After -- dtypes and shapes pinned by the schema, read back column-oriented
chunk = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.everything(),
).to_chunk()
write_parquet(chunk, "matching.parquet", labels=labels)

chunk, labels = read_parquet("matching.parquet")
result = MatchResults.from_chunk(chunk)
```

## Removed APIs

| Removed                                                                  | Replacement                                                                                                                 |
| :----------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| the `t4perceval.dataclass` package                                       | `t4perceval.core` / `t4perceval.component` / `t4perceval.archetype`                                                         |
| `Header(timestamp_ns, frame_id)`                                         | `TimePoint` (time) plus `Chunk.frame_id` (coordinate frame)                                                                 |
| the `BatchDetection3D → BatchTracking3D → BatchPrediction3D` inheritance | each archetype declares its components explicitly                                                                           |
| `BatchTrajectory3D` as a three-array component                           | promoted to an archetype; the columns split into `BatchWaypoints3D` / `BatchModeConfidence` / `BatchTimeOffset` and friends |
| `BatchTrajectory3D.positions` / `.confidences` / `.time_offsets_ns`      | `.waypoints` / `.mode_confidence` / `.time_offset`                                                                          |
| per-component `from_array()` / `as_array()`                              | kept, as base implementations on `ColumnarComponent`                                                                        |
