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

## Layers

```
t4perceval.system     System / Pipeline            the "S" of ECS
                      filter / matching            mask and pair up objects
t4perceval.core       Store / Chunk / Timeline     the recording
                      Archetype / Component        the data model
                      EntityPath / Descriptor      addressing
t4perceval.geometry   box corners / IoU / planes   vectorized, pairwise
t4perceval.io         Arrow / Parquet              persistence
t4perceval.label      LabelRegistry                meaning for the integer columns
```

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
    BatchDetection3D,
    BatchMatchResult,
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
    BatchDetection3D(
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
    BatchDetection3D(
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
).materialize(BatchMatchResult)
print(scene.num_tp, scene.num_fp, scene.num_fn)   # 1 1 1

# ...or ask about one frame, from the same store, with no recomputation.
frame_0 = store.range(
    "/matching/center_distance",
    timeline=FRAME,
    time_range=TimeRange.single(0),
).materialize(BatchMatchResult)
print(frame_0.num_tp, frame_0.num_fp, frame_0.num_fn)  # 1 1 1

# The filter's verdict is queryable data, not a discarded intermediate.
mask = store.range(
    distance_filter.target,
    timeline=FRAME,
    time_range=TimeRange.everything(),
).component(MASK)
print(mask.values)   # [ True False]

# Anything in the store persists with its dtypes and shapes pinned by the schema.
write_parquet(scene.to_chunk("/matching/center_distance", at=TimePoint.at(frame=0)), "matching.parquet")
```

## Benchmark

The columnar matching path was compared with `autoware_perception_evaluation (a.k.a. perception_eval)` 1.3.6 on a
synthetic 3D detection frame. The timed region includes the complete center-distance score matrix
and one-to-one assignment, but excludes object generation and data loading. Estimation and ground
truth contain the same number of single-class objects and use a 1.0 m matching threshold.

Measurements were taken on an Intel Core i9-14900K with Python 3.10.12, pinned to one CPU. Each
number is the median of 15 runs after a warm-up, with garbage collection disabled inside the timed
region.

| Objects (est / gt) | `perception_eval` | `t4perceval` | Speedup |
| -----------------: | ----------------: | -----------: | ------: |
|            10 / 10 |          8.925 ms |     0.300 ms |   29.7x |
|            50 / 50 |         51.877 ms |     0.405 ms |  127.9x |
|          100 / 100 |        126.025 ms |     0.590 ms |  213.6x |
|          200 / 200 |        338.333 ms |     1.634 ms |  207.0x |

Retained-memory usage was measured separately from the process RSS after constructing 100,000
estimation and 100,000 ground-truth rows. Import-time memory was subtracted from each process.

| Implementation    | RSS increase | Relative usage |
| :---------------- | -----------: | -------------: |
| `perception_eval` |    136.1 MiB |           6.9x |
| `t4perceval`      |     19.8 MiB |           1.0x |

The results are a matching-kernel benchmark, not an end-to-end T4 dataset benchmark. The two
environments also used their supported dependency versions (NumPy 1.26.4 for the original package
and NumPy 2.2.6 for `t4perceval`). In addition, assignment semantics differ: the original package
uses greedy matching, while `t4perceval` uses a globally optimal linear-sum assignment. Treat the
numbers as evidence for the effect of the columnar implementation rather than strict drop-in
runtime equivalence.

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
`masked_view`), and the full matching family (six modes on a shared `MatchingSystem` base, with
per-class `Thresholds` and vectorized geometry in `t4perceval.geometry`).

Next: the metric systems (mAP/APH, CLEAR, HOTA, ADE/FDE, classification), pass/fail, the `t4_devkit`
dataloader, and a visualization layer. See
[docs/design/en/system.md](docs/design/en/system.md) for where each of those fits on the protocol, and
[TODO.md](TODO.md) for the current list.
