# Detection 2D

Matching image-plane regions against image-plane regions.

## Overview

Structurally identical to [Detection 3D](detection-3d.md): the geometry is a rectangle instead of a
box, so the matcher is `IoURoiMatchingSystem` and the required columns are `roi` and `class_id`.
Everything downstream -- AP, APH's non-heading part, the confusion matrix -- is the same code.

```text
/ground_truth/objects ─┐
                       ├──▶ filters ──▶ IoURoiMatchingSystem ──▶ AP ──▶ mAP
/estimation/objects  ──┘
```

## Inputs

### Ground truth

```python
store.log(
    "/ground_truth/objects",
    Detections2D(
        roi=[[100, 200, 50, 30]],  # (x_min, y_min, height, width)
        class_id=labels.encode(["car"]),
        confidence=[1.0],
        visibility=[3],  # optional
    ),
    at=TimePoint.at(frame=0),
    frame_id="CAM_FRONT",
)
```

`frame_id` is the **camera channel**, because a 2D box is only meaningful in one image. Two cameras
are two entities:

```text
/ground_truth/CAM_FRONT/objects   frame_id="CAM_FRONT"
/ground_truth/CAM_BACK/objects    frame_id="CAM_BACK"
```

The T4 importer files them that way; see
[`objects2d_path`](../user-guide/dataset-importers.md#t4-dataset).

### Estimation

Same archetype, with the model's own `confidence`.

## Required components

```text
Detections2D
├── roi          required   (x_min, y_min, height, width), (N, 4) i32
├── class_id     required   (N,) i32
└── confidence   required   (N,) f64 in [0, 1]
```

!!! warning "ROI order"

    `(x_min, y_min, height, width)` -- height before width, and an integer dtype. This is not
    `(x, y, w, h)`.

## Optional components

| Component    | Enables                                   |
| :----------- | :---------------------------------------- |
| `visibility` | `FilterByVisibilitySystem` (ground truth) |

`Trackings2D` adds `instance_id`, which is what a 2D tracking evaluation needs.

## Filtering

The geometric filters need `position`, which a 2D detection does not have, so only the
class-and-quality filters apply:

| Filter                                                | Works on 2D?                  |
| :---------------------------------------------------- | :---------------------------- |
| `FilterByLabelSystem`                                 | yes                           |
| `FilterByConfidenceSystem`                            | yes                           |
| `FilterByVisibilitySystem`                            | yes                           |
| `FilterByInstanceSystem`                              | yes, with `Trackings2D`       |
| `FilterByDistance` / `Region` / `Speed` / `NumPoints` | no -- they require 3D columns |

To bound ROI area or position, write a
[custom filter](../recipes/custom-filter.md) over `ROI`; `BatchRoi.area()` is already there.

## Matching

```python
from t4perceval.system import IoURoiMatchingSystem

IoURoiMatchingSystem.between(
    "/estimation/objects",
    "/ground_truth/objects",
    threshold=0.5,
)
```

Higher is better, default threshold `0.5`, target `/matching/iou_roi`.

## Metrics

`AveragePrecisionSystem`, `MeanAveragePrecisionSystem`, `ClassificationSystem` and
`ConfusionMatrixSystem` all apply unchanged -- none of them requires 3D geometry.
`AveragePrecisionHeadingSystem` does **not**: it needs `quaternion`.

## Complete example

```python
from t4perceval import FRAME, Detections2D, LabelRegistry, MetricValues, Store, TimePoint, TimeRange
from t4perceval.system import (
    AveragePrecisionSystem,
    IoURoiMatchingSystem,
    Pipeline,
    SystemContext,
)

labels = LabelRegistry.from_names(["car", "pedestrian"])
store = Store()

store.log(
    "/ground_truth/objects",
    Detections2D(roi=[[0, 0, 10, 10]], class_id=labels.encode(["car"]), confidence=[1.0]),
    at=TimePoint.at(frame=0),
    frame_id="CAM_FRONT",
)
store.log(
    "/estimation/objects",
    Detections2D(roi=[[1, 1, 10, 10]], class_id=labels.encode(["car"]), confidence=[0.8]),
    at=TimePoint.at(frame=0),
    frame_id="CAM_FRONT",
)

matcher = IoURoiMatchingSystem.between(
    "/estimation/objects", "/ground_truth/objects", threshold=0.5
)
ap = AveragePrecisionSystem.on(
    matcher.target, "/estimation/objects", "/ground_truth/objects", target="/metrics/ap_2d"
)

Pipeline([matcher, ap]).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

values = store.range(
    "/metrics/ap_2d", timeline=FRAME, time_range=TimeRange.everything()
).materialize(MetricValues)
values.value.values  # array([ 1., nan])  -- car, then pedestrian (no ground truth)
```

A class with no ground truth in range gets a `NaN` row with `support == 0` rather than being
dropped, so the result's shape does not depend on the scene.

## Where to go next

- [Detection 3D](detection-3d.md) -- the same task with boxes.
- [Detections2D](../archetypes/detection-2d.md) -- the archetype's schema.
