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

!!! warning "The validity masks are not applied yet"

    `PathDisplacementSystem` does not read `mode_valid`, `timestep_valid` or `time_offset`. Padded
    modes and timesteps therefore contribute to ADE, FDE and the miss rate, and two trajectories with
    different time axes are compared by array index. See
    [Metric divergences](../development/metric-divergences.md). Until that is fixed, produce
    trajectories that are genuinely dense and share a time axis.

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
# ['/metrics/displacement/ade', '/metrics/displacement/fde', '/metrics/displacement/miss_rate']
```

| Parameter        | Default | Meaning                                                                                                                         |
| :--------------- | ------: | :------------------------------------------------------------------------------------------------------------------------------ |
| `top_k`          |       3 | keep this many modes, highest `mode_confidence` first                                                                           |
| `miss_tolerance` |     2.0 | a displacement at or above this counts as a miss                                                                                |
| `kernel`         |  `None` | which mode to reduce to: `None` keeps all kept modes, `"highest"` takes the most confident, `"min"` the best, `"max"` the worst |

Displacement is measured in **xy only**. What each metric reports, over the kept modes:

| Metric      | Definition                                                            |
| :---------- | :-------------------------------------------------------------------- |
| `ade`       | mean displacement over every (object, mode, timestep)                 |
| `fde`       | mean displacement at the **final** timestep                           |
| `miss_rate` | fraction of all (object, mode, timestep) distances ≥ `miss_tolerance` |

`kernel="min"` with `top_k=k` is the usual "minADE_k" / "minFDE_k".

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
    kernel="min",
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

With `kernel="min"` the better of the two modes is kept, so ADE and FDE are zero. Drop `kernel` to
score both modes and watch the 3 m one dominate.

## Known divergences

Beyond the unapplied validity masks, the miss rate is the fraction of _all_ mode/timestep distances
over the tolerance, where a common forecasting definition reports the fraction of _objects_ whose
selected trajectory misses at the final step. See
[Metric divergences](../development/metric-divergences.md).

## Where to go next

- [Tracking](tracking.md) -- the same inputs, without futures.
- [Predictions3D](../archetypes/prediction.md) · [Trajectories3D](../archetypes/trajectory.md)
