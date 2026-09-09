# Detection 3D

Matching 3D boxes against 3D boxes, and scoring them with average precision.

!!! note "Task, not type"

    `Detections3D` is an [archetype](../archetypes/detection-3d.md) -- a bundle of columns.
    *Detection 3D* is a task -- a pipeline you compose. The task runs against **any** entity
    carrying the required components, so `Trackings3D` and `Predictions3D` work here unchanged.

## Overview

```text
/ground_truth/objects ─┐
                       ├──▶ filters ──▶ matcher ──▶ AP / APH ──▶ mAP / mAPH
/estimation/objects  ──┘
```

## Inputs

### Ground truth

```python
store.log(
    "/ground_truth/objects",
    Detections3D(
        position=[[0.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]],
        size=[[1.9, 4.5, 1.6]],
        class_id=labels.encode(["car"]),
        confidence=[1.0],
        num_points=[42],  # optional, enables FilterByNumPoints
        visibility=[3],  # optional, enables FilterByVisibility
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)
```

### Estimation

Same archetype, with the model's own `confidence`. Both sides must be in the **same coordinate
frame** -- see [the cross-frame guard](../concepts/coordinate-system.md#the-cross-frame-guard).

## Required components

```text
Detections3D
├── position     required   3D centre, (N, 3) f64
├── quaternion   required   orientation, (N, 4) f64 xyzw
├── size         required   (width, length, height), (N, 3) f64
├── class_id     required   (N,) i32
└── confidence   required   (N,) f64 in [0, 1]
```

| Component    | Needed by                                                                |
| :----------- | :----------------------------------------------------------------------- |
| `position`   | every matcher, `FilterByDistance`, `FilterByRegion`                      |
| `quaternion` | `PlaneDistance` / `IoUBEV` / `IoU3D` matchers, `AveragePrecisionHeading` |
| `size`       | `PlaneDistance` / `IoUBEV` / `IoU3D` matchers                            |
| `class_id`   | every matcher, every metric                                              |
| `confidence` | `AveragePrecision` (it ranks by confidence), `FilterByConfidence`        |

## Optional components

| Component    | Enables                                   |
| :----------- | :---------------------------------------- |
| `velocity`   | `FilterBySpeedSystem`                     |
| `num_points` | `FilterByNumPointsSystem` (ground truth)  |
| `visibility` | `FilterByVisibilitySystem` (ground truth) |

## Filtering

Narrow both sides the same way, and **materialize** the result -- recall divides by the ground-truth
count, so the denominator must be the filtered set.

```python
from t4perceval.system import ApplyMaskSystem, FilterByDistanceSystem, Pipeline

narrow = []
for path in ("/ground_truth/objects", "/estimation/objects"):
    near = FilterByDistanceSystem.on(path, max_distance=50.0)
    narrow += [near, ApplyMaskSystem.of(path, near.target)]

Pipeline(narrow).run(ctx, scene)
```

See [Filtering](../user-guide/filtering.md) for the full family. The narrowing can run in the same
`Pipeline` as the evaluation, as the complete example below does.

## Matching

| Matcher                           | Score                 | Default threshold |
| :-------------------------------- | :-------------------- | ----------------: |
| `CenterDistanceMatchingSystem`    | 3D centre distance    |               1.0 |
| `CenterDistanceBEVMatchingSystem` | xy centre distance    |               1.0 |
| `PlaneDistanceMatchingSystem`     | nearest-face distance |               2.0 |
| `IoUBEVMatchingSystem`            | footprint IoU         |               0.5 |
| `IoU3DMatchingSystem`             | volume IoU            |               0.5 |

```python
from t4perceval.system import CenterDistanceMatchingSystem, Thresholds

CenterDistanceMatchingSystem.between(
    "/estimation/objects/kept",
    "/ground_truth/objects/kept",
    threshold=Thresholds(2.0, by_class=(("pedestrian", 0.5),)),
)
```

Pairs are chosen by a globally optimal linear-sum assignment, class-by-class unless
`class_agnostic=True`.

## Metrics

| Metric                          | Writes                      |
| :------------------------------ | :-------------------------- |
| `AveragePrecisionSystem`        | `/metrics/ap`               |
| `AveragePrecisionHeadingSystem` | `/metrics/aph`              |
| `MeanAveragePrecisionSystem`    | `/metrics/map`              |
| `ConfusionMatrixSystem`         | `/metrics/confusion_matrix` |

Each threshold needs its own matching run, so use the sweep rather than re-thresholding one result:

```python
from t4perceval.system import average_precision_sweep

average_precision_sweep(
    "/estimation/objects/kept",
    "/ground_truth/objects/kept",
    thresholds=[0.5, 1.0, 2.0, 4.0],
    heading=True,
)
```

## Complete example

```python
from t4perceval import FRAME, Detections3D, LabelRegistry, MetricValues, Store, TimePoint, TimeRange
from t4perceval.system import (
    ApplyMaskSystem,
    FilterByDistanceSystem,
    Pipeline,
    SystemContext,
    average_precision_sweep,
)

labels = LabelRegistry.from_names(["car", "pedestrian"])
store = Store()

for frame in range(3):
    store.log(
        "/ground_truth/objects",
        Detections3D(
            position=[[float(frame), 0.0, 0.0], [10.0, 5.0, 0.0], [80.0, 0.0, 0.0]],
            quaternion=[[0.0, 0.0, 0.0, 1.0]] * 3,
            size=[[1.9, 4.5, 1.6]] * 3,
            class_id=labels.encode(["car", "pedestrian", "car"]),
            confidence=[1.0, 1.0, 1.0],
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )
    store.log(
        "/estimation/objects",
        Detections3D(
            position=[[float(frame) + 0.3, 0.0, 0.0], [10.4, 5.0, 0.0], [30.0, 30.0, 0.0]],
            quaternion=[[0.0, 0.0, 0.0, 1.0]] * 3,
            size=[[1.9, 4.5, 1.6]] * 3,
            class_id=labels.encode(["car", "pedestrian", "car"]),
            confidence=[0.9, 0.7, 0.4],
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )

ctx = SystemContext(store, FRAME, labels=labels)
scene = TimeRange.everything()

narrow = []
for path in ("/ground_truth/objects", "/estimation/objects"):
    near = FilterByDistanceSystem.on(path, max_distance=50.0)
    narrow += [near, ApplyMaskSystem.of(path, near.target)]

Pipeline(
    narrow
    + average_precision_sweep(
        "/estimation/objects/kept",
        "/ground_truth/objects/kept",
        thresholds=[0.5, 1.0, 2.0],
    )
).run(ctx, scene)

m_ap = store.range("/metrics/map", timeline=FRAME, time_range=scene).materialize(MetricValues)
print("mAP:", round(m_ap.aggregate, 4))  # mAP: 0.9969
print("car:", round(m_ap.of_class(labels.class_id("car")), 4))  # car: 0.9938
```

## Known divergences

AP matching is a Hungarian assignment rather than confidence-ordered greedy matching, and APH's
denominator follows `autoware_perception_evaluation` rather than Waymo. Both are catalogued in
[Metric divergences](../development/metric-divergences.md).

## Where to go next

- [Detection 2D](detection-2d.md) · [Tracking](tracking.md) · [Prediction](prediction.md)
- [Evaluate a T4 dataset](../recipes/evaluate-t4-dataset.md)
