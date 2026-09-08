# Trajectories3D

Dense multi-modal future trajectories for `N` objects, without a box.

## Schema

| Component         | Requirement | Shape          | dtype  | Description                               |
| :---------------- | :---------- | :------------- | :----- | :---------------------------------------- |
| `waypoints`       | Required    | `(N, M, T, 3)` | `f64`  | objects, modes, timesteps, xyz            |
| `mode_confidence` | Required    | `(N, M)`       | `f64`  | in `[0, 1]`                               |
| `mode_valid`      | Optional    | `(N, M)`       | `bool` | which modes carry real data               |
| `timestep_valid`  | Optional    | `(N, M, T)`    | `bool` | which timesteps of which mode are real    |
| `time_offset`     | Optional    | `(N, T)`       | `i64`  | nanoseconds from now, strictly increasing |

Use this when the futures are the subject and the current pose is not -- otherwise use
[`Predictions3D`](prediction.md), which carries both.

## The shape

```text
waypoints[n, m, t] = the xyz of object n, in mode m, at timestep t

N  objects
M  modes         -- how many futures the model offered, fixed per instance
T  timesteps     -- how far ahead, fixed per instance
```

`M` and `T` are fixed within one instance; shorter trajectories are padded and masked by
`mode_valid` / `timestep_valid`. Every column is validated against the `waypoints` shape at
construction, so a mismatched mask raises rather than silently misaligning.

```python
trajectories.num_modes
trajectories.num_timesteps
```

## Building one

Row-at-a-time, for tests and adapters:

```python
import numpy as np
from t4perceval import Trajectories3D, TrajectoryMode3D

mode = TrajectoryMode3D(
    position=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    confidence=0.7,
    time_offset_ns=[500_000_000, 1_000_000_000],
)

trajectories = Trajectories3D.from_modes([[mode, other_mode], [mode2, other_mode2]])
trajectories.to_modes(0)  # every mode of object 0
```

`TrajectoryMode3D` is **not a component** -- it is a readable constructor. The columnar form is
`BatchWaypoints3D` and friends. `from_modes` requires every object to contribute the same number of
modes and every mode to share the same `time_offset_ns`.

Empty, with a pinned shape:

```python
Trajectories3D.empty(num_modes=6, num_timesteps=12)
```

## Validation rules

- `waypoints` must have at least one mode and one timestep.
- `mode_confidence` must agree with `waypoints` on `M`, and be finite and within `[0, 1]`.
- `mode_valid` must be `(M,)` per row, `timestep_valid` `(M, T)`, `time_offset` `(T,)`.
- `time_offset_ns` must be non-negative and strictly increasing.

## Where to go next

- [Predictions3D](prediction.md) -- these columns plus a box.
- [Prediction evaluation](../evaluation/prediction.md)
