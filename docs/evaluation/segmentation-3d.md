# Segmentation

Semantic segmentation labels a _point_ or a _pixel_, not an object, so there is nothing to pair:
estimation and ground truth are already aligned element by element. The matching stage does not
apply, and a metric here is a per-class count over aligned rows rather than a set of verdicts.

```text
/ground_truth/points ─┐   SegmentationIoUSystem ─────────────▶ /metrics/segmentation/iou
                      ├─▶                                       /metrics/segmentation/accuracy
/estimation/points  ──┘                                         /metrics/segmentation/pixel_accuracy
                          SegmentationConfusionMatrixSystem ──▶ /metrics/segmentation/confusion_matrix
```

## The row is the element

Neither archetype names its elements. Row `i` of the estimation is compared with row `i` of the
ground truth, at every frame -- so both must hold the same number of rows per frame and enumerate
the same elements in the same order: the pixels of one image, or the points of one cloud. That is
the ordinary case when a model labels the cloud the ground truth was annotated on, and it is exactly
how a filter's `mask` relates to the entity it was computed over.

A frame where the counts differ is an error, not a misalignment:

```text
ValueError: /estimation/points and /ground_truth/points must label the same elements, but hold
2 and 3 row(s) at frame=1
```

The check is made per frame, because a whole-range comparison would silently shift every later row
when one side lacked a frame.

## Point order

A label image has a fixed order -- the row is the pixel -- but a point cloud does not: a model, or the
node that publishes its output, does not promise to list the points in the order the ground truth
does. So when both entities carry `point`, the metrics also compare the coordinates row by row and
refuse a frame whose points differ by more than `point_tolerance` (default `1e-6`):

```text
ValueError: /estimation/points and /ground_truth/points describe different points at frame=0:
2 row(s) differ by up to 2. Bring the estimation into the ground truth's order first with
AlignPointsSystem, or pass point_tolerance=None if the rows are known to correspond
```

`AlignPointsSystem` is that stage. It pairs each ground-truth point with the estimation point at the
same location (nearest neighbour within `tolerance`) and rewrites the estimation -- every column it
carries -- in the ground truth's order, as a new entity beside the source:

```python
from t4perceval.system import AlignPointsSystem, Pipeline, SegmentationIoUSystem

align = AlignPointsSystem.between(
    "/estimation/points", "/ground_truth/points"
)  # -> /estimation/points/aligned
iou = SegmentationIoUSystem.between(align.target, "/ground_truth/points")
Pipeline([align, iou]).run(ctx, TimeRange.everything())
```

Estimation points that no ground-truth point claims are dropped -- a prediction on a point the
ground truth does not label cannot be scored. A ground-truth point with no estimation point within
`tolerance` raises, as do coincident points, which geometry cannot tell apart. Like a coordinate
transform, the correspondence is an explicit stage whose result stays in the store, not something
the metric does quietly; it is unrelated to [`t4perceval.align`](../recipes/align-frames.md), which
pairs _frames_ by timestamp.

### Ground truth the estimation does not cover

A cropped or downsampled output leaves ground-truth points with no prediction. Whether those are
misses or simply out of scope is an evaluation decision, so it is a stage you add rather than a
default: `FilterByCoverageSystem` masks the ground truth by whether a reference point lies within
`tolerance`, and the mask records exactly which points were left out.

```python
from t4perceval.system import ApplyMaskSystem, FilterByCoverageSystem

covered = FilterByCoverageSystem.between("/ground_truth/points", "/estimation/points")
gt_kept = ApplyMaskSystem.of("/ground_truth/points", covered.target)
align = AlignPointsSystem.between("/estimation/points", gt_kept.target)
iou = SegmentationIoUSystem.between(align.target, gt_kept.target)
Pipeline([covered, gt_kept, align, iou]).run(ctx, TimeRange.everything())
```

`support` then counts only the covered points, and `covered.target` says how many were dropped.

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
    frame_id="LIDAR_CONCAT",
)
```

`point` uses the `POINT` descriptor, not `POSITION`: a labelled point is not an object with a pose,
and the separate name stops an object filter from being pointed at a point cloud and appearing to
work. A coordinate transform still moves it as a point, so a cloud can be brought into another frame
with [`TransformEntitySystem`](../user-guide/transforms.md#expressing-an-entity-in-another-frame)
before it is scored.

### 2D

A label image is one class per pixel in row-major order. The image size is not a column -- it would
be the same value repeated per pixel -- but a one-row **static** component on the entity:

```python
from t4perceval import SemanticSegmentation2D
from t4perceval.component import BatchImageSize
from t4perceval.descriptors import IMAGE_SIZE

store.log(
    "/ground_truth/pixels",
    SemanticSegmentation2D.from_label_map(label_image),  # (H, W) -> N = H*W rows
    at=TimePoint.at(frame=0),
    frame_id="CAM_FRONT",
)
store.log_static_components("/ground_truth/pixels", {IMAGE_SIZE: BatchImageSize([[height, width]])})
```

`as_label_map(height, width)` restores the image. `frame_id` is the camera channel, as for every 2D
archetype.

## Required components

```text
SemanticSegmentation3D          SemanticSegmentation2D
├── point      required         └── class_id   required     one per pixel, row-major
└── class_id   required
                                static on the entity: IMAGE_SIZE  (height, width)
```

Neither archetype has optional components.

## Filtering

`FilterByLabelSystem` works, because it only needs `class_id`. The geometric filters need `POSITION`
and so do **not** apply to `SemanticSegmentation3D` despite its `point` column -- by design.

Applying the **same** mask to both entities keeps them aligned:

```python
drop = FilterByLabelSystem.on(GT, exclude=["road"])
gt_kept = ApplyMaskSystem.of(GT, drop.target)
est_kept = ApplyMaskSystem.of(EST, drop.target, target="/estimation/points/kept")
```

For excluding classes from the score, `ignore=` below is the shorter way.

## Matching

Not applicable. The rows are already aligned.

## Metrics

```python
from t4perceval.system import Pipeline, SegmentationConfusionMatrixSystem, SegmentationIoUSystem

iou = SegmentationIoUSystem.between("/estimation/points", "/ground_truth/points")
matrix = SegmentationConfusionMatrixSystem.between("/estimation/points", "/ground_truth/points")
Pipeline([iou, matrix]).run(ctx, TimeRange.everything())
```

Both systems depend on `class_id` alone, so the same implementation scores a 2D label image and a
3D point cloud. Everything is derived from one count matrix over `(ground-truth class, estimated
class)`, pooled over the frames in range.

### Reading the results

`SegmentationIoUSystem` writes three `MetricValues` tables under its target:

| Entity                                 | Per-class row                                               | Aggregate row (`class_id == ALL_CLASSES`)               |
| :------------------------------------- | :---------------------------------------------------------- | :------------------------------------------------------ |
| `/metrics/segmentation/iou`            | `TP / (TP + FP + FN)`; `NaN` when the union is empty        | the mean of the defined rows: **mIoU**                  |
| `/metrics/segmentation/accuracy`       | `TP / (TP + FN)` -- the class's elements labelled correctly | mean class accuracy                                     |
| `/metrics/segmentation/pixel_accuracy` | --                                                          | correctly labelled elements over all evaluated elements |

`support` is the number of evaluated ground-truth elements of the class; the aggregate row's
support is their sum. Every registered class gets a row, even at support `0`.

```python
iou = store.range("/metrics/segmentation/iou", timeline=FRAME, time_range=scene).materialize(
    MetricValues
)
iou.of_class(labels.class_id("car"))
iou.aggregate  # mIoU
```

`SegmentationConfusionMatrixSystem` writes a long-form
[`ConfusionMatrix`](../archetypes/metrics.md) with ground-truth classes down the rows and estimation
classes across the columns. The background **column** holds elements predicted as no known class --
an estimation of `UNKNOWN_CLASS_ID`, or of an ignored class. The background **row is always zero**:
a ground-truth element without a known class is ignored or rejected, never counted.

### Ignoring labels

Ground-truth rows labelled `UNKNOWN_CLASS_ID` (`-1`) are always left out. `ignore=` leaves out more,
by class name or id:

```python
SegmentationIoUSystem.between(EST, GT, ignore=["noise", "sky"])
```

An ignored class is _not evaluated_: its ground-truth rows are dropped, and it leaves the class axis
in every table, so a prediction of it counts as background -- a false negative for the true class.
This is the "void" semantics of the Cityscapes and mmsegmentation benchmarks. A ground-truth class
the registry does not know is malformed data and raises, naming the ids and the frame.

### Coordinate frames

Two entities that state different frames -- `CAM_FRONT` against `CAM_BACK`, say -- raise, exactly as
a matcher would; comparing two cameras' label maps element-wise is the mistake this catches.
`check_frames=False` opts out when the frames are known to coincide.

## Complete example

```python
from t4perceval import (
    FRAME,
    LabelRegistry,
    MetricValues,
    SemanticSegmentation3D,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.system import Pipeline, SegmentationIoUSystem, SystemContext

labels = LabelRegistry.from_names(["road", "car"])
store = Store()

store.log(
    "/ground_truth/points",
    SemanticSegmentation3D(
        point=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        class_id=labels.encode(["road", "car", "car"]),
    ),
    at=TimePoint.at(frame=0),
    frame_id="LIDAR_CONCAT",
)
store.log(
    "/estimation/points",
    SemanticSegmentation3D(
        point=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        class_id=labels.encode(["road", "car", "road"]),
    ),
    at=TimePoint.at(frame=0),
    frame_id="LIDAR_CONCAT",
)

iou = SegmentationIoUSystem.between("/estimation/points", "/ground_truth/points")
Pipeline([iou]).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

scene = TimeRange.everything()
result = store.range(iou.targets[0], timeline=FRAME, time_range=scene).materialize(MetricValues)
result.of_class(labels.class_id("road"))  # 0.5 -- one hit, one false positive
result.of_class(labels.class_id("car"))  # 0.5 -- one hit, one miss
result.aggregate  # 0.5 -- mIoU
```

## Where to go next

- [Segmentation archetypes](../archetypes/segmentation.md) -- the schemas.
- [Metrics](../user-guide/metrics.md#segmentation-iou-accuracy-confusion) -- the parameters.
- [Metric divergences](../development/metric-divergences.md) -- how an undefined IoU is treated.
