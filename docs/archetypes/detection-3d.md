# Detections3D

3D bounding boxes with a class and a confidence.

## Schema

| Component    | Requirement | Shape    | dtype | Description                      |
| :----------- | :---------- | :------- | :---- | :------------------------------- |
| `position`   | Required    | `(N, 3)` | `f64` | box centre, in the chunk's frame |
| `quaternion` | Required    | `(N, 4)` | `f64` | orientation, `xyzw`              |
| `size`       | Required    | `(N, 3)` | `f64` | `(width, length, height)`        |
| `class_id`   | Required    | `(N,)`   | `i32` | semantic class                   |
| `confidence` | Required    | `(N,)`   | `f64` | in `[0, 1]`                      |
| `velocity`   | Optional    | `(N, 3)` | `f64` | metres per second                |
| `num_points` | Optional    | `(N,)`   | `i32` | sensor points inside the box     |
| `visibility` | Optional    | `(N,)`   | `i8`  | occlusion level                  |

Descriptors: `POSITION`, `QUATERNION`, `SIZE`, `CLASS_ID`, `CONFIDENCE`, `VELOCITY`, `NUM_POINTS`,
`VISIBILITY`.

## Constructing

```python
from t4perceval import Detections3D

detections = Detections3D(
    position=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
    quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
    size=[[1.9, 4.5, 1.6]] * 2,
    class_id=labels.encode(["car", "car"]),
    confidence=[0.9, 0.4],
    velocity=[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # keyword-only, optional
)
```

The first five are positional; every optional field is keyword-only.

!!! warning "Two easy mistakes"

    `quaternion` is `xyzw` (SciPy / ROS), not `wxyz`. `size` is `(width, length, height)`, not
    Autoware's `(length, width, height)`.

## What it unlocks

| Component present | Enables                                                                        |
| :---------------- | :----------------------------------------------------------------------------- |
| `position`        | every 3D matcher, `FilterByDistanceSystem`, `FilterByRegionSystem`             |
| `quaternion`      | `PlaneDistance` / `IoUBEV` / `IoU3D` matchers, `AveragePrecisionHeadingSystem` |
| `size`            | `PlaneDistance` / `IoUBEV` / `IoU3D` matchers                                  |
| `class_id`        | every matcher, every metric, `FilterByLabelSystem`                             |
| `confidence`      | `AveragePrecisionSystem`, `FilterByConfidenceSystem`                           |
| `velocity`        | `FilterBySpeedSystem`                                                          |
| `num_points`      | `FilterByNumPointsSystem`                                                      |
| `visibility`      | `FilterByVisibilitySystem`                                                     |

## Relationships

`Trackings3D` and `Predictions3D` declare the same five required components under the same
descriptors, so anything that accepts a `Detections3D` accepts them too:

```python
tracking.has(*Detections3D.required_descriptors())  # True
prediction.has(*Detections3D.required_descriptors())  # True
```

The reverse does not hold: a `Detections3D` has no `instance_id`, so `ClearSystem` rejects it.

## Where to go next

- [Detection 3D evaluation](../evaluation/detection-3d.md) -- the task.
- [Trackings3D](tracking.md) · [Predictions3D](prediction.md)
