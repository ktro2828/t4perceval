# PointCloud

A set of 3D points in the chunk's frame, with nothing said about them.

## Schema

| Component | Requirement | Shape    | dtype | Description                     |
| :-------- | :---------- | :------- | :---- | :------------------------------ |
| `point`   | Required    | `(N, 3)` | `f64` | the point, in the chunk's frame |

```python
from t4perceval import PointCloud

store.log(
    "/ground_truth/points",
    PointCloud(point=[[1.0, 2.0, 0.3], [1.1, 2.0, 0.3]]),
    at=TimePoint.at(frame=0),
    frame_id="LIDAR_CONCAT",
)
```

## When to use it

For the cloud itself: a raw sensor sweep, a cropped or down-sampled copy of one, or the reference
cloud a coverage filter asks about. Once each point carries a class, log a
[`SemanticPointCloud`](segmentation.md) instead.

## Composition

`SemanticPointCloud` is `PointCloud` plus `class_id`, by composition rather than inheritance, as
`Trackings3D` is `Detections3D` plus `instance_id`:

```text
PointCloud  ──(+ class_id)──▶  SemanticPointCloud
```

Both resolve `point` to the same `POINT` descriptor, so a labelled cloud satisfies every requirement
a bare one does, and a view over a `SemanticPointCloud` entity materializes as a `PointCloud` when
only the geometry is wanted:

```python
from t4perceval import PointCloud, SemanticPointCloud

segmentation.has(*PointCloud.required_descriptors())  # True
view.materialize(PointCloud).point.values  # (N, 3), labels left behind
```

## POINT, not POSITION

`point` uses the **`POINT`** descriptor, deliberately not `POSITION`. A point is not an object
with a pose, and the separate name stops `FilterByDistanceSystem` -- or any other system asking for
a 3D object position -- from being pointed at a cloud and appearing to work. See
[the segmentation page](segmentation.md#point-not-position) for the full argument.

## What it unlocks

| Component present | Enables                                                                        |
| :---------------- | :----------------------------------------------------------------------------- |
| `point`           | `TransformEntitySystem`, and every `FilterPointsBy*` system, coverage included |

Without a `class_id` there is no class axis, so `FilterByLabelSystem` and the segmentation metrics
do not apply.

## Where to go next

- [SemanticImage / SemanticPointCloud](segmentation.md) -- the labelled cloud.
- [Filters](../systems/filters.md#point) -- the point filters.
- [Frames and points](../systems/frames.md) -- how a cloud moves between frames.
