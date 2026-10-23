# Sensing

A point cloud judged against the ground-truth boxes, with no estimation involved. See
[Sensing evaluation](../evaluation/sensing.md) for the task. The three systems live where their
role puts them: the per-box judge in `t4perceval.system.sensing`, the mask in
`t4perceval.system.filter.sensing`, the rate in `t4perceval.system.metric.sensing`. All are
exported from `t4perceval.system`.

| System                       | Sources              | Requires                                   | Parameters (default)                                                                      | Writes                                              |
| :--------------------------- | :------------------- | :----------------------------------------- | :---------------------------------------------------------------------------------------- | :-------------------------------------------------- |
| `SensingDetectionSystem`     | `(objects, points)`  | `position`, `quaternion`, `size` / `point` | `box_scale_0m` (1.0), `box_scale_100m` (1.0), `min_num_points` (1), `check_frames` (True) | `SensingResults` at `<objects>/sensing`             |
| `SensingNonDetectionSystem`  | `(points, objects)`  | `point` / `position`, `quaternion`, `size` | `areas` (required), `box_scale_0m` (1.0), `box_scale_100m` (1.0), `check_frames` (True)   | a `mask` at `<points>/filter/non_detection`         |
| `SensingDetectionRateSystem` | `(sensing, objects)` | `sensing_status` / `class_id`              | --                                                                                        | `MetricValues` at `/metrics/sensing/detection_rate` |

All three build with `.between(first, second, target=...)`.

**`SensingDetectionSystem`** counts, for every box at every frame, the points inside the box once
its footprint is scaled by `box_scale_at(distance, box_scale_0m, box_scale_100m)` from
[`t4perceval.geometry`](../reference/api/geometry.md). The verdict is `DETECTED` at
`min_num_points` or more, `FAILED` below, and `WARNING` when the box carries
`visibility == NONE`. The result is row-aligned with the objects entity, like a mask. A frame of
boxes with no cloud frame raises.

**`SensingNonDetectionSystem`** is a `MaskSystem` with two sources, like
`FilterPointsByCoverageSystem`, and is **not a filter**: its mask is true for the points that
offend -- inside one of the `areas` and inside no scaled box -- so `BatchMask.num_selected` is the
failure count and `ApplyMaskSystem` yields the failures. `areas` are
[`NonDetectionArea`](../evaluation/sensing.md#non-detection) values: an xy polygon with a closed
height range, or `(polygon, z_min, z_max)` triples. A cloud frame with no boxes is judged with none.

**`SensingDetectionRateSystem`** reads a detection result and the objects it was computed over,
joined by row, and writes `DETECTED / (DETECTED + FAILED)` per registered class and for all classes,
with that denominator as `support`. Warnings count in neither.

## Where to go next

- [Sensing evaluation](../evaluation/sensing.md) -- inputs, parameters and a complete example.
- [SensingResults](../archetypes/sensing.md) -- the result archetype.
- [Filters](filters.md#point) -- the other systems over a cloud's `point` column.
