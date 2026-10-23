# Sensing

Checking that the sensor sees what the annotations say is there, and nothing where they say it is
empty.

## Overview

This is the `sensing` task of `perception_eval`. There is no estimation: a LiDAR sweep is judged
directly against the ground-truth boxes, in two areas.

- **Detection area** -- the inside of every box. Enough points must fall inside each box, once its
  footprint is scaled for its distance, or the object was not sensed.
- **Non-detection area** -- regions you declare empty, such as the lane ahead. No point may remain
  there once the points inside any box are discounted.

```text
/ground_truth/objects ─┬──▶ SensingDetectionSystem ────▶ <objects>/sensing            (SensingResults, one row per box)
/sensor/lidar ─────────┤                                      └──▶ SensingDetectionRateSystem ──▶ /metrics/sensing/detection_rate
                       └──▶ SensingNonDetectionSystem ─▶ <points>/filter/non_detection  (a mask: the offending points)
```

## Inputs

### Ground truth

The annotated boxes, as `Detections3D` or anything carrying its geometry. `visibility` is read when
present: a box annotated `VisibilityLevel.NONE` is reported as a warning rather than judged.

### Point cloud

The sweep to judge, logged by you as a [`PointCloud`](../archetypes/point-cloud.md) -- the dataset
importers do not load raw sensor data. Log it **in the same frame as the boxes**, normally
`base_link`: the scale is a function of the distance from the origin, and the nearest point is the
one closest to it. A cloud in the sensor frame is moved first with
[`TransformEntitySystem`](../user-guide/transforms.md). `perception_eval` expects the cloud with the
ground removed; that is still your pre-processing.

```python
from t4perceval import PointCloud, TimePoint

store.log("/sensor/lidar", PointCloud(point=xyz), at=TimePoint.at(frame=0), frame_id="base_link")
```

## Required components

```text
Detections3D                       PointCloud
├── position     required          └── point   required
├── quaternion   required
├── size         required
└── visibility   optional   NONE -> WARNING
```

`class_id` is needed only by `SensingDetectionRateSystem`.

## Filtering

`perception_eval` judges detection on the objects that survive `target_uuids`, but crops the
non-detection areas with **every** annotation, filtered or not. Here each system is pointed at the
entity it should read, so the same choice is two paths:

```python
from t4perceval.system import ApplyMaskSystem, FilterByInstanceSystem

GT = "/ground_truth/objects"
LIDAR = "/sensor/lidar"

targets = FilterByInstanceSystem.on(GT, instances=["1b40c087..."])
kept = ApplyMaskSystem.of(GT, targets.target)
sensing = SensingDetectionSystem.between(kept.target, LIDAR)  # judge only the targets
offending = SensingNonDetectionSystem.between(LIDAR, GT, areas=areas)  # discount every box
```

## Detection

```python
from t4perceval.system import SensingDetectionSystem

sensing = SensingDetectionSystem.between(GT, LIDAR, box_scale_0m=1.0, box_scale_100m=1.2)
str(sensing.target)
# '/ground_truth/objects/sensing'
```

| Parameter        | Default | Meaning                                                   |
| :--------------- | :------ | :-------------------------------------------------------- |
| `box_scale_0m`   | `1.0`   | footprint scale of a box at the origin                    |
| `box_scale_100m` | `1.0`   | footprint scale of a box 100 m away; linear in between    |
| `min_num_points` | `1`     | inside points a box needs to count as detected            |
| `check_frames`   | `True`  | refuse a cloud and boxes that state different coordinates |

The scale is `box_scale_0m + (box_scale_100m - box_scale_0m) / 100 * distance`, with the distance the
3D norm of the box centre. It multiplies width and length only; the height is never scaled. A point
on a face counts as inside.

Every box gets one row of [`SensingResults`](../archetypes/sensing.md), written with the objects'
partitioning so that row `i` of the result is the verdict on row `i` of the objects:

| Status     | When                                                          |
| :--------- | :------------------------------------------------------------ |
| `DETECTED` | at least `min_num_points` points inside the scaled box        |
| `FAILED`   | fewer                                                         |
| `WARNING`  | `visibility == NONE`, whatever the count: the box is occluded |

```python
results = store.range(sensing.target, timeline=FRAME, time_range=SCENE).materialize(SensingResults)
results.num_points.values  # inside points per box
results.is_detected  # bool per box
results.nearest_point.values  # (N, 3); NaN where nothing was inside
results.num_failed
```

The count is written under the `num_points` descriptor -- the same column the annotation carries,
with the same meaning -- so a filter on it re-judges the result at another threshold without
re-counting, and `ApplyMaskSystem` turns that into the boxes that were seen:

```python
from t4perceval.system import ApplyMaskSystem, FilterByNumPointsSystem

seen = FilterByNumPointsSystem.on(sensing.target, min_num_points=5)
seen_boxes = ApplyMaskSystem.of(GT, seen.target, name="seen")
```

A frame of boxes for which the cloud has **no frame at all** raises: a missing sweep is not a sweep
with no points. An empty sweep is, and fails every box. A frame with no boxes keeps its (empty)
partition.

## Non-detection

An area is an xy polygon between two heights. `perception_eval` listed the corners of its lower and
upper planes; `from_vertices` takes that list as is.

```python
from t4perceval.system import NonDetectionArea, SensingNonDetectionSystem

lane = NonDetectionArea([(0.0, 1.5), (10.0, 1.5), (10.0, -1.5), (0.0, -1.5)], z_min=0.0, z_max=4.0)
lane = NonDetectionArea.from_vertices(non_detection_areas[0])  # the original's 8-corner list

offending = SensingNonDetectionSystem.between(LIDAR, GT, areas=[lane], box_scale_100m=1.2)
str(offending.target)
# '/sensor/lidar/filter/non_detection'
```

`areas` takes one area, several, or `(polygon, z_min, z_max)` triples; the mask is their union. The
box scales are the same two parameters as above and should be given the same values.

The output is a mask over the cloud, but with the **opposite polarity of a filter: true marks an
offending point** -- one inside an area and inside no scaled box at its frame. The count of the mask
is the failure count, and `ApplyMaskSystem` materializes the failures, which is what the original's
`pointcloud_failed_non_detection` held:

```python
failures = ApplyMaskSystem.of(LIDAR, offending.target, name="non_detection")
left = store.range(failures.target, timeline=FRAME, time_range=SCENE).materialize(PointCloud)
```

A frame of the cloud for which the objects entity has nothing is judged with no boxes: every point in
an area offends. Annotations are complete by definition, so their absence means an empty road,
unlike a missing cloud.

## Metrics

`perception_eval` stops at the per-frame verdicts. `SensingDetectionRateSystem` adds the summary the
other tasks have: one `MetricValues` row per class and one for all classes, over the whole range.

```python
from t4perceval.system import SensingDetectionRateSystem

rate = SensingDetectionRateSystem.between(sensing.target, GT)
str(rate.target)
# '/metrics/sensing/detection_rate'
```

| Row        | Value                                             | Support             |
| :--------- | :------------------------------------------------ | :------------------ |
| a class    | `DETECTED / (DETECTED + FAILED)`, `NaN` when none | `DETECTED + FAILED` |
| all (`-1`) | the same over every class                         |                     |

Warnings are left out of both: they are statements about the annotation, not the sensor. The
`threshold` column is `NaN`; `min_num_points` is a property of the verdicts, recorded on the system
that made them.

## Complete example

Three boxes and one sweep. The near car is hit twice, the pedestrian is annotated as fully occluded,
and the far car gets one return just beyond its rear face, which the scale at 100 m still reaches.
One stray return sits on the lane ahead; the clutter below the lane's height band does not count.

```python
import numpy as np

from t4perceval import (
    FRAME,
    Detections3D,
    LabelRegistry,
    MetricValues,
    PointCloud,
    SensingResults,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import SensingStatus
from t4perceval.system import (
    ApplyMaskSystem,
    NonDetectionArea,
    Pipeline,
    SensingDetectionRateSystem,
    SensingDetectionSystem,
    SensingNonDetectionSystem,
    SystemContext,
)

GT = "/ground_truth/objects"
LIDAR = "/sensor/lidar"
labels = LabelRegistry.from_names(["car", "pedestrian"])
store = Store()

store.log(
    GT,
    Detections3D(
        position=[[10.0, 0.0, 0.0], [30.0, 5.0, 0.0], [100.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * 3,
        size=[[2.0, 4.0, 1.5], [0.8, 0.8, 1.8], [2.0, 4.0, 1.5]],
        class_id=labels.encode(["car", "pedestrian", "car"]),
        confidence=[1.0] * 3,
        visibility=[3, 0, 3],  # FULL, NONE, FULL
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)
store.log(
    LIDAR,
    PointCloud(
        point=[
            [8.2, 0.3, 0.5],
            [8.4, -0.4, 0.6],
            [8.3, 0.0, 0.9],  # the near car; the last is above it
            [97.7, 0.0, 0.2],  # 0.3 m behind the far car's rear face
            [5.0, 0.0, 0.5],  # on the lane ahead
            [6.0, 0.0, -0.3],  # below the lane's height band
        ]
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)

lane_ahead = NonDetectionArea(
    [(3.0, 1.5), (7.0, 1.5), (7.0, -1.5), (3.0, -1.5)], z_min=0.0, z_max=4.0
)
sensing = SensingDetectionSystem.between(GT, LIDAR, box_scale_100m=1.2, min_num_points=2)
offending = SensingNonDetectionSystem.between(LIDAR, GT, areas=[lane_ahead])
failures = ApplyMaskSystem.of(LIDAR, offending.target, name="non_detection")
rate = SensingDetectionRateSystem.between(sensing.target, GT)
Pipeline([sensing, offending, failures, rate]).run(
    SystemContext(store, FRAME, labels=labels), TimeRange.everything()
)

results = store.range(
    sensing.target, timeline=FRAME, time_range=TimeRange.everything()
).materialize(SensingResults)
for count, status, nearest in zip(
    results.num_points.values, results.sensing_status.values, results.nearest_point.values
):
    print(SensingStatus(status).name, count, np.round(nearest, 1).tolist())

left = store.range(failures.target, timeline=FRAME, time_range=TimeRange.everything())
print("left in the lane:", left.materialize(PointCloud).point.values.tolist())

values = store.range(rate.target, timeline=FRAME, time_range=TimeRange.everything()).materialize(
    MetricValues
)
print("detection rate", round(values.aggregate, 3), "over", int(values.support.values[-1]), "boxes")
```

```text
DETECTED 2 [8.2, 0.3, 0.5]
WARNING 0 [nan, nan, nan]
FAILED 1 [97.7, 0.0, 0.2]
left in the lane: [[5.0, 0.0, 0.5]]
detection rate 0.5 over 2 boxes
```

The far car's box spans 98 to 102 m unscaled; at `box_scale_100m=1.2` its half length is 2.4 m, so
the return at 97.7 m is inside, but one point is short of `min_num_points`. The pedestrian is a
warning and is not in the rate's denominator.

## Known divergences

A point on a box face is inside here; `perception_eval`'s crop is half-open in xy. A box whose
computed scale is zero or negative (`box_scale_100m` below `box_scale_0m`, far enough away) holds
nothing rather than cropping with an inverted footprint. See
[Metric divergences](metric-divergences.md#sensing-box-membership).

## Where to go next

- [SensingResults](../archetypes/sensing.md) -- the result's columns.
- [Sensing systems](../systems/sensing.md) -- parameters and targets.
- [Transforms](../user-guide/transforms.md) -- moving a sensor-frame cloud into `base_link`.
- [Segmentation](segmentation-3d.md) -- the other evaluation over point clouds.
