# t4perceval

Component-oriented perception evaluation for the [T4 dataset](https://github.com/tier4/t4-devkit).

A redesign of [`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation)
around [rerun](https://github.com/rerun-io/rerun)'s data model: data lives at an **entity path**, is made
of **component** columns, is bundled into **archetypes**, stored as **chunks**, indexed along
**timelines** inside a **store**, and transformed by **systems**.

The point of the redesign is that an evaluation task stops being a value to branch on. There is no
`EvaluationTask` enum and no single config dict; a task _is_ the set of components present plus the
pipeline you compose.

## Design documents

|              | 日本語                                           | English                                          |
| :----------- | :----------------------------------------------- | :----------------------------------------------- |
| Data model   | [ja/data_model.md](docs/design/ja/data_model.md) | [en/data_model.md](docs/design/en/data_model.md) |
| System layer | [ja/system.md](docs/design/ja/system.md)         | [en/system.md](docs/design/en/system.md)         |
| Migration    | [ja/migration.md](docs/design/ja/migration.md)   | [en/migration.md](docs/design/en/migration.md)   |
| Naming       | —                                                | [en/naming.md](docs/design/en/naming.md)         |

## Layers

```bash
t4perceval.system     System / Pipeline            the "S" of ECS
                      filter / matching            mask and pair up objects
t4perceval.core       Store / Chunk / Timeline     the mutable log
                      Archetype / Component        the data model
                      EntityPath / Descriptor      addressing
t4perceval.geometry   box corners / IoU / planes   vectorized, pairwise
t4perceval.transform  FrameGraph / resolver         coordinate frames as data
t4perceval.importer   T4 dataset / MCAP ROS bag    external formats in
t4perceval.io         Arrow / Parquet              persistence
t4perceval.label      LabelRegistry                meaning for the integer columns
t4perceval.recording  Recording                    a log plus what its integers mean
```

`importer` converts an external representation into this one; `io` moves an already-native
recording to and from storage. Reading a saved recording is `io`; reading a dataset is
`importer`.

## Data shapes

Every component is one column of `N` rows with a fixed per-row shape and dtype.

| Component                                           | Shape          | dtype          | Notes                                      |
| :-------------------------------------------------- | :------------- | :------------- | :----------------------------------------- |
| `BatchPosition3D` / `BatchVelocity` / `BatchSize3D` | `(N, 3)`       | `f64`          | `BatchSize3D` is `(width, length, height)` |
| `BatchPosition2D` / `BatchSize2D`                   | `(N, 2)`       | `f64`          |                                            |
| `BatchQuaternion`                                   | `(N, 4)`       | `f64`          | `xyzw` order (SciPy's convention)          |
| `BatchRoi`                                          | `(N, 4)`       | `i32`          | `(x_min, y_min, height, width)`            |
| `BatchClassId`                                      | `(N,)`         | `i32`          | meaning comes from `LabelRegistry`         |
| `BatchConfidence`                                   | `(N,)`         | `f64`          | constrained to `[0, 1]`                    |
| `BatchInstanceId`                                   | `(N,)`         | `i64`          | interned by `InstanceRegistry`             |
| `BatchNumPoints` / `BatchPixel`                     | `(N,)`         | `i32`          |                                            |
| `BatchVisibility` / `BatchMatchStatus`              | `(N,)`         | `i8`           | ordered enums                              |
| `BatchMask`                                         | `(N,)`         | `bool`         | a filter's verdict                         |
| `BatchWaypoints3D`                                  | `(N, M, T, 3)` | `f64`          | `M` modes, `T` timesteps                   |
| `BatchModeConfidence` / `BatchModeValid`            | `(N, M)`       | `f64` / `bool` |                                            |
| `BatchTimestepValid`                                | `(N, M, T)`    | `bool`         |                                            |
| `BatchTimeOffset`                                   | `(N, T)`       | `i64`          | nanoseconds, strictly increasing           |

Columns are always **read-only** and never share memory with a writable array you passed in.

## Usage

```python
import numpy as np

from t4perceval import (
    FRAME,
    Detections3D,
    MatchResults,
    InstanceRegistry,
    LabelRegistry,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.descriptors import MASK
from t4perceval.io import write_parquet
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    FilterByDistanceSystem,
    Pipeline,
    SystemContext,
)

labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian", "motorbike"])
instances = InstanceRegistry()
store = Store()

# One frame of objects is one columnar batch, not a list of objects.
store.log(
    "/ground_truth/objects",
    Detections3D(
        position=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
        size=[[1.9, 4.5, 1.6]] * 2,
        class_id=labels.encode(["car", "car"]),
        confidence=[1.0, 1.0],
    ),
    at=TimePoint.at(frame=0, timestamp_ns=1_624_164_470_849_887_000),
    frame_id="base_link",
)
store.log(
    "/estimation/objects",
    Detections3D(
        position=[[0.3, 0.0, 0.0], [50.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
        size=[[1.9, 4.5, 1.6]] * 2,
        class_id=labels.encode(["car", "car"]),
        confidence=[0.9, 0.4],
    ),
    at=TimePoint.at(frame=0, timestamp_ns=1_624_164_470_849_887_000),
    frame_id="base_link",
)

# The task is the pipeline. Each stage writes its result back into the store.
distance_filter = FilterByDistanceSystem.on("/estimation/objects", max_distance=40.0)
pipeline = Pipeline(
    [
        distance_filter,
        CenterDistanceMatchingSystem.between(
            "/estimation/objects",
            "/ground_truth/objects",
            threshold=1.0,
        ),
    ],
)
pipeline.run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

# Aggregate over a whole scene...
scene = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.everything(),
).materialize(MatchResults)
print(scene.num_tp, scene.num_fp, scene.num_fn)  # 1 1 1

# ...or ask about one frame, from the same store, with no recomputation.
frame_0 = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.single(0),
).materialize(MatchResults)
print(frame_0.num_tp, frame_0.num_fp, frame_0.num_fn)  # 1 1 1

# The filter's verdict is queryable data, not a discarded intermediate.
mask = store.range(
    distance_filter.target,
    timeline=FRAME,
    time_range=TimeRange.everything(),
).component(MASK)
print(mask.values)  # [ True False]

# Anything in the store persists with its dtypes and shapes pinned by the schema.
write_parquet(
    scene.to_chunk("/matching/center_distance", at=TimePoint.at(frame=0)), "matching.parquet"
)
```

### Importing a T4 dataset

Requires the `t4` extra (`pip install 't4perceval[t4]'`).

```python
from t4perceval.evaluation import build_evaluation_store
from t4perceval.importer.t4 import T4Importer
from t4perceval.system import Pipeline
from t4perceval.system.preset import average_precision_sweep

importer = T4Importer.open("tests/data/t4dataset")

# The registry is an input, not something each importer invents. Class ids are assigned in
# first-seen order, so two sources that each derive their own registry are both valid and
# silently incompatible -- and the disagreement shows up as plausible numbers, not an error.
labels = importer.label_registry()

ground_truth = importer.import_scene(labels=labels)  # -> Recording

# A Recording is read-only: `Pipeline.run` writes its results back into the store it reads
# from, so the entities an evaluation needs are materialized into a fresh one first. Only
# what you name moves, which is what keeps a saved result about the evaluation.
setup = build_evaluation_store(ground_truth, estimation)
Pipeline(
    average_precision_sweep("/estimation/objects", "/ground_truth/objects", thresholds=[1.0])
).run(setup.context(), TimeRange.everything())

result = setup.into_recording()  # inputs, matches and metrics, with provenance
```

### Importing a ROS bag

Requires the `rosbag` extra (`pip install 't4perceval[rosbag]'`). An MCAP bag is decoded from the
message definitions it embeds, so no ROS installation and no Autoware message packages are needed.

```python
from t4perceval import TIMESTAMP
from t4perceval.importer.rosbag import BagSelection, RosbagImporter
from t4perceval.transform import TransformResolver

importer = RosbagImporter.open("/data/bags/run_0")  # a directory of *.mcap, or one file
importer.topics()  # the DetectedObjects / TrackedObjects / PredictedObjects topics it holds

# Same rule as above: hand the ground-truth importer this very registry.
labels = importer.label_registry()  # the Autoware class enum, in enum order

estimation = importer.import_topic(  # -> Recording, one topic at a time
    labels=labels,
    selection=BagSelection(topic="/perception/object_recognition/tracking/objects"),
)

# `/tf_static` becomes static edges and `/tf` temporal ones -- on the TIMESTAMP timeline only,
# since a sample between two messages has no frame index. Look them up on that axis.
resolver = TransformResolver.of(estimation, timeline=TIMESTAMP)
```

The schema decides the archetype (`TrackedObjects` -> `Trackings3D`), `ObjectClassification` maps
enum -> canonical name -> registry id, `Shape.dimensions` `(x=length, y=width, z=height)` becomes
`BatchSize3D` `(width, length, height)`, a body-frame twist is rotated into the message frame, and
`BatchConfidence` is the top classification probability by default
(`ImportOptions(confidence="existence")` for `existence_probability`).

### Coordinate frames

A chunk states the frame its rows are in (`frame_id`), and a transform is recorded data like
everything else -- one edge of the frame graph, split the way ROS splits a `TransformStamped`: the
chunk's `frame_id` is the **parent**, `child_frame_id` is the **child**. Static (a calibration) and
temporal (an ego pose) are the same archetype; `static` says only that the value does not depend on
a timeline.

`Transform3D` is the one archetype whose components are **mono** -- it describes a single
relationship, not `N` objects -- so its fields are values rather than columns:

```python
pose = Transform3D(
    translation=[1.2, 0.0, 1.8], rotation=[0.0, 0.0, 0.0, 1.0], child_frame_id="lidar"
)
pose.translation.value  # array([1.2, 0. , 1.8])
pose.child_frame_id.name  # 'lidar'
```

Storage stays columnar: a mono value is widened into its `Batch*` counterpart on the way into a
chunk, so a range query over three ego samples still returns a three-row column.

```python
from t4perceval.transform import TransformResolver, transform_edges

# The T4 importer records the scene's tree: `map -> base_link` per keyframe from `ego_pose`,
# and a static `base_link -> <channel>` per sensor from `calibrated_sensor`.
print(sorted(edge.frames for edge in transform_edges(ground_truth)))
# [('base_link', 'CAM_BACK'), ('base_link', 'CAM_FRONT'), ('base_link', 'LIDAR_TOP'),
#  ('map', 'base_link')]

# Static and temporal edges compose in one graph:
#   T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar
resolver = TransformResolver.of(ground_truth, timeline=FRAME)
pose = resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=1)
print(pose.translation.value)  # [10.  0.  2.]
```

Edges are found by reading the chunks, not by parsing entity paths, so where a transform is filed is
a filing decision and a frame name may contain a `/`. A temporal edge picks its sample with a
`LookupPolicy` (`LATEST`, `EXACT`, `NEAREST`, `INTERPOLATE`); an unreachable frame raises rather than
quietly resolving to identity.

Nothing rewrites an entity's rows into another frame yet, so the system layer refuses to compare
geometry across frames instead of silently producing plausible numbers:

```text
ValueError: Cannot compare geometry across coordinate frames: /estimation/objects in 'base_link',
/ground_truth/objects in 'map'. Bring the inputs into one frame first.
```

Every matcher is covered through `MatchingSystem`, and every geometric metric through `MatchJoin`.
Only two _different, stated_ frames raise; an unstated frame is not a disagreement, and
`check_frames=False` opts out per system.

## Benchmark

`benchmarks/compare.py` feeds `autoware_perception_evaluation` (`perception_eval` 1.3.6) and
`t4perceval` the same synthetic scenes and compares both speed and the metric values themselves.
The two run in separate processes because `perception_eval` pins NumPy 1.

The full report is regenerated into [`benchmarks/results/latest.md`](benchmarks/results/latest.md) with:

```bash
uv run python benchmarks/compare.py --check
```

See [benchmarks/README.md](benchmarks/README.md) for the options, what the run does, and its prerequisites.

Scene phases run over 10 frames with 200 objects per frame (median of 5 runs after 2 warm-ups, one
pinned logical CPU); matching is a single frame.

| Workload (200 objects / frame)               | `perception_eval` | `t4perceval` |   Improvement |
| :------------------------------------------- | ----------------: | -----------: | ------------: |
| Data-model construction, one frame           |          6.035 ms |     0.069 ms |  87.2x faster |
| Center-distance matching, one frame          |        339.900 ms |     1.517 ms | 224.1x faster |
| Detection: mAP + mAPH at 4 thresholds        |       6360.713 ms |    94.074 ms |  67.6x faster |
| Tracking: CLEAR (MOTA / MOTP / ID switches)  |       3303.261 ms |    19.573 ms | 168.8x faster |
| Prediction: ADE / FDE / miss rate, top-k 1,3 |       3718.902 ms |    30.701 ms | 121.1x faster |
| Retained RSS, 20,000 est / 20,000 GT         |          83.7 MiB |      5.2 MiB | 16.0x smaller |

Numerical agreement is checked on two scenes. In the **unambiguous** scene every estimate has exactly
one feasible ground truth, so `perception_eval`'s greedy assignment and `t4perceval`'s linear-sum
assignment pick the same pairs: all 133 compared values (per-class and overall AP / APH / mAP / mAPH,
MOTA / MOTP / ID switches, ADE / FDE / miss rate) agree to within `1e-9` (`1e-8` for APH, which
`perception_eval` rounds). In the **dense** scene the two differ on 87 values, every one of which is
classified against a documented divergence -- Hungarian vs confidence-ordered greedy matching, the
previous-frame MOTP score, heading sign in APH, and how ID switches are counted -- see
[docs/TODOs/metrics.md](docs/TODOs/metrics.md). `--check` fails on any difference that is not.

## Development

```bash
uv sync
uv run pytest tests -q
ruff check t4perceval tests && ruff format --check t4perceval tests
```

## Status

Implemented: the data model (`core`), all component and archetype types, the store with timelines and
static data, lazy views, the label registries, Arrow/Parquet IO, the system protocol with `Pipeline`,
the full filter family (eight filters on a shared `MaskSystem` base, plus `CombineMasksSystem` and
`masked_view`), the full matching family (six modes on a shared `MatchingSystem` base, with per-class
`Thresholds` and vectorized geometry in `t4perceval.geometry`), the metric systems (mAP/APH, CLEAR,
ADE/FDE/MissRate, classification, confusion matrix), the T4 and MCAP/ROS bag importers with the
`Recording` boundary and `t4perceval.evaluation`, and coordinate transforms -- static and temporal
edges, frame-graph discovery from the data, composition through `TransformResolver`, and the
cross-frame guard.

Next: `HotaSystem` and pass/fail, a system that materializes a transformed entity, `t4perceval.align`
for pairing ground-truth and bag frames by timestamp, persisting a whole `Recording`, and a
visualization layer. See
[docs/design/en/system.md](docs/design/en/system.md) for where each of those fits on the protocol,
[docs/TODOs/design.md](docs/TODOs/design.md) for the current list, and
[docs/TODOs/metrics.md](docs/TODOs/metrics.md) for where the metric implementations differ from the
official benchmark definitions.
