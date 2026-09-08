# Segmentation 3D

!!! warning "Partially supported"

    The archetypes exist and round-trip through the store and Parquet. **No matcher and no metric
    system reads them yet** -- there is no IoU-per-class or mIoU system in `t4perceval.system`. What
    you can do today is log, query and persist segmentation data, and compute metrics over the
    columns yourself.

    Segmentation metrics are on the [roadmap](../development/roadmap.md).

## Overview

Segmentation labels a _point_ or a _pixel_, not an object, so there is nothing to pair: the
estimation and the ground truth are already aligned element by element. That is why the matching
stage does not apply, and why a metric here is a per-class count rather than a set of verdicts.

```text
/ground_truth/points ─┐
                      ├──▶ (compare element-wise) ──▶ per-class IoU ──▶ mIoU
/estimation/points  ──┘
```

## Inputs

### 3D

```python
from t4perceval import SemanticSegmentation3D

store.log(
    "/ground_truth/points",
    SemanticSegmentation3D(
        point=[[1.0, 2.0, 0.3], [1.1, 2.0, 0.3]],  # (N, 3) f64
        class_id=labels.encode(["car", "car"]),  # (N,)   i32
    ),
    at=TimePoint.at(frame=0),
    frame_id="LIDAR_TOP",
)
```

### 2D

```python
from t4perceval import SemanticSegmentation2D

store.log(
    "/ground_truth/pixels",
    SemanticSegmentation2D(
        pixel=[10234, 10235],  # (N,) i32 flat pixel indices
        class_id=labels.encode(["road", "road"]),
    ),
    at=TimePoint.at(frame=0),
    frame_id="CAM_FRONT",
)
```

## Required components

```text
SemanticSegmentation3D          SemanticSegmentation2D
├── point      required         ├── pixel      required   flat pixel index, (N,) i32
└── class_id   required         └── class_id   required
```

Neither archetype has optional components. Note that `point` uses the `POINT` descriptor, not
`POSITION` -- a labelled point is not an object with a pose, and the separate name stops an
object filter from being pointed at a point cloud and appearing to work.

## Optional components

None.

## Filtering

`FilterByLabelSystem` works, because it only needs `class_id`. The geometric filters need
`POSITION` and so do **not** apply to `SemanticSegmentation3D` despite its `point` column -- by
design.

## Matching

Not applicable. The rows are already aligned.

## Metrics

Not implemented. In the meantime, per-class IoU over one frame is a short query:

```python
import numpy as np
from t4perceval import FRAME, TimeRange
from t4perceval.descriptors import CLASS_ID


def per_class_iou(store, estimation, ground_truth, class_ids, *, at):
    time_range = TimeRange.single(at)
    est = store.range(estimation, timeline=FRAME, time_range=time_range).component(CLASS_ID).values
    gt = store.range(ground_truth, timeline=FRAME, time_range=time_range).component(CLASS_ID).values
    if len(est) != len(gt):
        raise ValueError("estimation and ground truth must label the same elements")

    result = {}
    for class_id in class_ids:
        e, g = est == class_id, gt == class_id
        union = np.count_nonzero(e | g)
        result[class_id] = np.count_nonzero(e & g) / union if union else float("nan")
    return result
```

That assumes both entities enumerate the same points in the same order, which is the ordinary case
when the estimation is a per-point label over the ground truth's cloud. If they do not, associate
them yourself first -- the `point` column is there for it.

To produce a proper metric system instead, see
[Write a custom metric](../recipes/custom-metric.md); note that `MetricSystem`'s base assumes a
`MatchJoin`, so a segmentation metric implements the `System` protocol directly.

## Complete example

```python
from t4perceval import FRAME, LabelRegistry, SemanticSegmentation3D, Store, TimePoint

labels = LabelRegistry.from_names(["road", "car"])
store = Store()

store.log(
    "/ground_truth/points",
    SemanticSegmentation3D(
        point=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        class_id=labels.encode(["road", "car", "car"]),
    ),
    at=TimePoint.at(frame=0),
    frame_id="LIDAR_TOP",
)
store.log(
    "/estimation/points",
    SemanticSegmentation3D(
        point=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        class_id=labels.encode(["road", "car", "road"]),
    ),
    at=TimePoint.at(frame=0),
    frame_id="LIDAR_TOP",
)

per_class_iou(store, "/estimation/points", "/ground_truth/points", [0, 1], at=0)
# {0: 0.5, 1: 0.5}
```

## Where to go next

- [Roadmap](../development/roadmap.md) -- where segmentation metrics sit.
- [Extending systems](../development/extending-systems.md) -- implementing the `System` protocol directly.
