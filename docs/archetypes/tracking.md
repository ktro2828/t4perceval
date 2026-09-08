# Trackings2D / Trackings3D

Detections carrying a persistent instance identifier.

## Trackings3D

| Component     | Requirement | Shape    | dtype | Description                       |
| :------------ | :---------- | :------- | :---- | :-------------------------------- |
| `position`    | Required    | `(N, 3)` | `f64` | box centre                        |
| `quaternion`  | Required    | `(N, 4)` | `f64` | orientation, `xyzw`               |
| `size`        | Required    | `(N, 3)` | `f64` | `(width, length, height)`         |
| `class_id`    | Required    | `(N,)`   | `i32` | semantic class                    |
| `confidence`  | Required    | `(N,)`   | `f64` | in `[0, 1]`                       |
| `instance_id` | Required    | `(N,)`   | `i64` | persistent identity, keyword-only |
| `velocity`    | Optional    | `(N, 3)` | `f64` |                                   |
| `num_points`  | Optional    | `(N,)`   | `i32` |                                   |
| `visibility`  | Optional    | `(N,)`   | `i8`  |                                   |

```python
from t4perceval import Trackings3D

Trackings3D(
    position=[[0.0, 0.0, 0.0]],
    quaternion=[[0.0, 0.0, 0.0, 1.0]],
    size=[[1.9, 4.5, 1.6]],
    class_id=labels.encode(["car"]),
    confidence=[0.9],
    instance_id=instances.encode(["7f3c...-a1"]),
)
```

## Trackings2D

| Component     | Requirement | Shape    | dtype | Description                     |
| :------------ | :---------- | :------- | :---- | :------------------------------ |
| `roi`         | Required    | `(N, 4)` | `i32` | `(x_min, y_min, height, width)` |
| `class_id`    | Required    | `(N,)`   | `i32` |                                 |
| `confidence`  | Required    | `(N,)`   | `f64` |                                 |
| `instance_id` | Required    | `(N,)`   | `i64` | keyword-only                    |
| `visibility`  | Optional    | `(N,)`   | `i8`  |                                 |

## The instance id

`instance_id` is what makes these tracking archetypes rather than detections. It is an `i64` interned
from a dataset UUID by an [`InstanceRegistry`](../components/object.md#batchinstanceid).

The two sides' id spaces are **independent**: a metric compares identity _continuity_, not id
equality, so the tracker's own integers are fine. When both sides are interned into one registry,
give them distinct namespaces -- `instance_namespace="gt"` and `"est"` is what the importers do.

## What it unlocks

Everything [`Detections3D`](detection-3d.md#what-it-unlocks) unlocks, plus:

| Component present | Enables                                 |
| :---------------- | :-------------------------------------- |
| `instance_id`     | `ClearSystem`, `FilterByInstanceSystem` |

## Relationships

The box components are **re-declared**, not inherited from `Detections3D`. They resolve to the same
descriptors, so `tracking.has(*Detections3D.required_descriptors())` is `True` and any system that
requires a detection's components runs unchanged.

`Predictions3D` extends `Trackings3D` the same way, with `waypoints` and `mode_confidence`.

## Where to go next

- [Tracking evaluation](../evaluation/tracking.md) -- the task.
- [Predictions3D](prediction.md)
