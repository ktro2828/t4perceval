# Prediction

Comparing predicted future trajectories against what actually happened.

## Overview

A prediction evaluation is a tracking evaluation whose objects also carry **multi-modal future
waypoints**. The current pose is matched as usual, and the metric then compares each estimate's
predicted futures against the ground-truth future.

```text
/ground_truth/objects ─┐
                       ├──▶ matcher (on the current pose) ──▶ PathDisplacementSystem
/estimation/objects  ──┘                                       └─▶ ADE / FDE / miss rate
```

## Inputs

### Ground truth

The ground truth is **single-mode**: the one future that happened.

```python
import numpy as np
from t4perceval import Predictions3D

Predictions3D(
    position=[[0.0, 0.0, 0.0]],
    quaternion=[[0.0, 0.0, 0.0, 1.0]],
    size=[[2.0, 4.0, 2.0]],
    class_id=labels.encode(["car"]),
    confidence=[1.0],
    instance_id=[1],
    waypoints=np.asarray([[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]]),  # (1, 1, 3, 3)
    mode_confidence=np.asarray([[1.0]]),  # (1, 1)
    time_offset=np.asarray([[100, 200, 300]]) * 1_000_000,  # (1, 3) ns: 0.1, 0.2, 0.3 s ahead
)
```

### Estimation

Same archetype with `M > 1` modes and a confidence per mode:

```python
Predictions3D(
    ...,
    waypoints=np.asarray(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],  # mode 0
                [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]],  # mode 1
            ]
        ]
    ),  # (1, 2, 3, 3)
    mode_confidence=np.asarray([[0.7, 0.3]]),  # (1, 2)
    time_offset=np.asarray([[100, 200, 300]]) * 1_000_000,  # (1, 3); both modes share it
)
```

The T4 importer produces this shape from `ImportOptions(kind_3d="predictions", future_seconds=3.0)`.

## Required components

```text
Predictions3D
├── position          required   the current 3D centre
├── quaternion        required
├── size              required
├── class_id          required
├── confidence        required
├── instance_id       required
├── waypoints         required   (N, M, T, 3) f64 -- objects, modes, timesteps, xyz
└── mode_confidence   required   (N, M)      f64 in [0, 1]
```

`M` and `T` are fixed within one instance; shorter trajectories are padded and masked.

## Optional components

| Component        | Shape       | Meaning                                       |
| :--------------- | :---------- | :-------------------------------------------- |
| `velocity`       | `(N, 3)`    | current velocity                              |
| `mode_valid`     | `(N, M)`    | which modes carry real data                   |
| `timestep_valid` | `(N, M, T)` | which timesteps of which mode carry real data |
| `time_offset`    | `(N, T)`    | nanoseconds from now, strictly increasing     |

`time_offset` is optional for the archetype but **required by `PathDisplacementSystem`** on both
sides: the metric compares trajectories in time, and without a time axis there is nothing to compare
them at. Both importers write it. Absent masks mean every mode and timestep is real.

### How the metric reads them

```text
ground truth   t:  0.1   0.2   0.3   0.4 (invalid)          scored at its valid times only
estimation     t:  0 ─────────── 0.5 ─────────── 1.0        interpolated at 0.1, 0.2, 0.3
                   ↑ current position anchors t = 0
```

- **Time, not index.** Each valid ground-truth step is scored against the prediction linearly
  interpolated at the same `time_offset`; the estimation's current `position` is the point at
  `t = 0`. A prediction sampled every 0.5 s and a ground truth sampled at the dataset's own rate are
  therefore compared correctly.
- **A prediction that ends early** is held at its last valid point beyond its horizon -- predicting
  less far ahead is penalised, not excused.
- **Estimation masks.** Only modes with `mode_valid` and at least one valid timestep compete for
  `top_k` and for `best_of_k`; invalid timesteps are interpolated across. An estimation with no
  valid mode is scored as standing still at its position.
- **Ground-truth masks.** A step is valid only if both its `timestep_valid` and its mode's
  `mode_valid` say so. Invalid steps are not scored, FDE is taken at the last valid one, and an
  object with no valid future step is left out -- its class can report `NaN` while `support` still
  counts it.

## Filtering

Everything in [Filtering](../user-guide/filtering.md) applies to the current pose. There is no
filter over the trajectory columns; write a [custom one](../recipes/custom-filter.md) over
`WAYPOINTS` if you need it.

## Matching

The matcher runs on the current pose, not on the trajectories:

```python
from t4perceval.system import CenterDistanceMatchingSystem

CenterDistanceMatchingSystem.between("/estimation/objects", "/ground_truth/objects", threshold=1.0)
```

Only pairs that are true positives **and** agree on class are scored by the metric.

## Metrics

```python
from t4perceval.system import PathDisplacementSystem

displacement = PathDisplacementSystem.on(
    matcher.target,
    "/estimation/objects",
    "/ground_truth/objects",
    top_k=3,
    miss_tolerance=2.0,
)
[str(t) for t in displacement.targets]
# ['/metrics/path_displacement/ade',
#  '/metrics/path_displacement/fde',
#  '/metrics/path_displacement/miss_rate']
```

| Parameter        | Default | Meaning                                                               |
| :--------------- | ------: | :-------------------------------------------------------------------- |
| `top_k`          |       3 | keep this many modes, highest `mode_confidence` first                 |
| `miss_tolerance` |     2.0 | a displacement at or above this counts as a miss                      |
| `best_of_k`      | `False` | report minADE_k / minFDE_k instead of the average over the kept modes |

Displacement is measured in **xy only**. What each metric reports, over the kept modes:

| Metric      | Definition                                                                         |
| :---------- | :--------------------------------------------------------------------------------- |
| `ade`       | per object, the mean over kept modes × valid steps; then the mean over objects     |
| `fde`       | per object, the mean over kept modes at the **last valid** step; then over objects |
| `miss_rate` | fraction of all scored (object, mode, step) distances ≥ `miss_tolerance`           |

With `best_of_k=True`, `ade` and `fde` are instead the **smallest** per-object ADE and,
independently, the smallest per-object FDE over the kept modes -- the usual minADE_k / minFDE_k, as
nuScenes and Waymo define them; `miss_rate` is then counted on the minADE mode. To score only the
most confident mode, use `top_k=1`.

## Complete example

```python
import numpy as np
from t4perceval import (
    FRAME,
    LabelRegistry,
    MetricValues,
    Predictions3D,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    PathDisplacementSystem,
    Pipeline,
    SystemContext,
)

labels = LabelRegistry.from_names(["car"])
store = Store()


def prediction(x, waypoints, confidences):
    return Predictions3D(
        position=[[x, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]],
        size=[[2.0, 4.0, 2.0]],
        class_id=labels.encode(["car"]),
        confidence=[0.9],
        instance_id=[1],
        waypoints=np.asarray(waypoints, dtype=np.float64)[None, ...],
        mode_confidence=np.asarray([confidences], dtype=np.float64),
        time_offset=[[100_000_000, 200_000_000, 300_000_000]],  # every 0.1 s
    )


straight = [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]

store.log(
    "/ground_truth/objects",
    prediction(0.0, straight, [1.0]),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)
store.log(
    "/estimation/objects",
    prediction(
        0.05,
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],  # right
            [[1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0]],  # 3 m off
        ],
        [0.6, 0.4],
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)

matcher = CenterDistanceMatchingSystem.between(
    "/estimation/objects", "/ground_truth/objects", threshold=1.0
)
displacement = PathDisplacementSystem.on(
    matcher.target,
    "/estimation/objects",
    "/ground_truth/objects",
    top_k=2,
    miss_tolerance=2.0,
    best_of_k=True,
)

Pipeline([matcher, displacement]).run(
    SystemContext(store, FRAME, labels=labels), TimeRange.everything()
)

for target in displacement.targets:
    values = store.range(target, timeline=FRAME, time_range=TimeRange.everything()).materialize(
        MetricValues
    )
    print(target.name, values.value.values)
```

```text
ade       [0.]
fde       [0.]
miss_rate [0.]
```

With `best_of_k=True` the better of the two modes is kept, so ADE and FDE are zero. Drop
`best_of_k` to average over both modes and watch the 3 m one pull the numbers up.

## Known divergences

The miss rate is the fraction of _all_ scored mode/timestep distances over the tolerance, where a
common forecasting definition reports the fraction of _objects_ whose selected trajectory misses at
the final step. See [Metric divergences](../development/metric-divergences.md).

## Where to go next

- [Tracking](tracking.md) -- the same inputs, without futures.
- [Predictions3D](../archetypes/prediction.md) · [Trajectories3D](../archetypes/trajectory.md)
