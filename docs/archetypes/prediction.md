# Predictions3D

Tracked 3D boxes together with their predicted future trajectories.

## Schema

| Component         | Requirement | Shape          | dtype  | Description                               |
| :---------------- | :---------- | :------------- | :----- | :---------------------------------------- |
| `position`        | Required    | `(N, 3)`       | `f64`  | the **current** box centre                |
| `quaternion`      | Required    | `(N, 4)`       | `f64`  | orientation, `xyzw`                       |
| `size`            | Required    | `(N, 3)`       | `f64`  | `(width, length, height)`                 |
| `class_id`        | Required    | `(N,)`         | `i32`  |                                           |
| `confidence`      | Required    | `(N,)`         | `f64`  | in `[0, 1]`                               |
| `instance_id`     | Required    | `(N,)`         | `i64`  | keyword-only                              |
| `waypoints`       | Required    | `(N, M, T, 3)` | `f64`  | keyword-only; objects, modes, timesteps   |
| `mode_confidence` | Required    | `(N, M)`       | `f64`  | keyword-only; in `[0, 1]`                 |
| `velocity`        | Optional    | `(N, 3)`       | `f64`  |                                           |
| `mode_valid`      | Optional    | `(N, M)`       | `bool` | which modes carry real data               |
| `timestep_valid`  | Optional    | `(N, M, T)`    | `bool` | which timesteps of which mode are real    |
| `time_offset`     | Optional    | `(N, T)`       | `i64`  | nanoseconds from now, strictly increasing |

```python
prediction.num_modes  # M
prediction.num_timesteps  # T
```

`M` and `T` are fixed within one instance, and every trajectory column is checked against the
`waypoints` shape at construction. Shorter trajectories are padded and masked.

## Constructing

```python
import numpy as np
from t4perceval import Predictions3D

Predictions3D(
    position=[[0.0, 0.0, 0.0]],
    quaternion=[[0.0, 0.0, 0.0, 1.0]],
    size=[[2.0, 4.0, 2.0]],
    class_id=labels.encode(["car"]),
    confidence=[0.9],
    instance_id=[1],
    waypoints=np.asarray(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]],
            ]
        ]
    ),  # (1, 2, 3, 3)
    mode_confidence=np.asarray([[0.7, 0.3]]),  # (1, 2)
)
```

Ground truth is conventionally single-mode -- `M = 1`, the one future that happened.

## Composition

`Predictions3D` composes the box, instance **and** trajectory components rather than inheriting
from `Trackings3D`, so a system requiring only the box components, only the instance id, or only the
trajectory columns all apply directly:

```python
prediction.has(*Detections3D.required_descriptors())  # True
prediction.has(*Trackings3D.required_descriptors())  # True
prediction.has(*Trajectories3D.required_descriptors())  # True
```

## What it unlocks

Everything [`Trackings3D`](tracking.md#what-it-unlocks) unlocks, plus:

| Component present              | Enables                  |
| :----------------------------- | :----------------------- |
| `waypoints`, `mode_confidence` | `PathDisplacementSystem` |

!!! warning "The validity masks are not read yet"

    `PathDisplacementSystem` ignores `mode_valid`, `timestep_valid` and `time_offset`. See
    [Metric divergences](../development/metric-divergences.md).

## Where to go next

- [Prediction evaluation](../evaluation/prediction.md) -- the task.
- [Trajectories3D](trajectory.md) -- the same futures without a box.
