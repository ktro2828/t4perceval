# Geometry components

Positions, rotations, sizes, velocities, image-plane regions and label images.

## BatchPosition3D

`(N, 3)` `f64` -- a 3D centre, in metres, expressed in the chunk's
[`frame_id`](../concepts/coordinate-system.md).

```python
from t4perceval.component import BatchPosition3D

BatchPosition3D([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
```

Descriptor: `POSITION`. Required by every 3D matcher, `FilterByDistanceSystem`,
`FilterByRegionSystem`, `FilterByPolarGridSystem` and `FilterByMapSystem`.

`BatchPosition2D` is the `(N, 2)` counterpart.

## BatchQuaternion

`(N, 4)` `f64` -- unit quaternions in **`xyzw` order**, SciPy's convention.

```python
from t4perceval.component import BatchQuaternion

rotations = BatchQuaternion([[0.0, 0.0, 0.0, 1.0]])
rotations.as_rotation()  # scipy.spatial.transform.Rotation
rotations.yaw()  # (N,) yaw angle in radians
rotations.normalized()  # every quaternion rescaled to unit norm
```

Descriptor: `QUATERNION`. Required by the box matchers and by
`AveragePrecisionHeadingSystem`.

!!! warning "xyzw, not wxyz"

    ROS `geometry_msgs/Quaternion` is `x, y, z, w` and so is this. `pyquaternion` and many papers
    use `w, x, y, z`.

## BatchSize3D

`(N, 3)` `f64` in **`(width, length, height)`** order.

```python
from t4perceval.component import BatchSize3D

BatchSize3D([[1.9, 4.5, 1.6]])  # a car: 1.9 m wide, 4.5 m long, 1.6 m tall
```

Descriptor: `SIZE`. Required by `PlaneDistance`, `IoUBEV` and `IoU3D` matching.

!!! warning "width first"

    Autoware's `Shape.dimensions` is `(x=length, y=width, z=height)`. The ROS bag importer swaps
    them for you; if you build the column yourself, do the swap.

`BatchSize2D` is the `(N, 2)` counterpart.

## BatchVelocity

`(N, 3)` `f64` -- metres per second, in the chunk's frame.

```python
from t4perceval.component import BatchVelocity

BatchVelocity([[3.0, 0.0, 0.0]]).speed  # array([3.]) -- a property, the L2 norm per row
```

Descriptor: `VELOCITY`. Optional on every object archetype; required by
`FilterBySpeedSystem`.

## BatchRoi

`(N, 4)` `i32` -- an image-plane region as **`(x_min, y_min, height, width)`**.

```python
from t4perceval.component import BatchRoi

rois = BatchRoi([[100, 200, 50, 30]])
rois.area()  # array([1500]) -- height * width, in pixels
rois.x_min, rois.y_min, rois.width, rois.height  # each (N,)
rois.x_max, rois.y_max  # derived
```

Descriptor: `ROI`. Required by `IoURoiMatchingSystem`.

## BatchClassIdImage

`(N, H, W)` `i32` -- one `(H, W)` class-id image per row. `H` and `W` are wildcard dimensions,
inferred once per column, so every image in a column has the same resolution and a zero-sized side is
refused.

```python
images = BatchClassIdImage(np.zeros((2, 1080, 1920), dtype=np.int32))
images.height, images.width, images.num_pixels()  # ints: the same for every row
images.as_class_id()  # BatchClassId of N * H * W rows, image by image, each row-major
```

Descriptor: `CLASS_ID_IMAGE`. The archetype field is the **mono** `ClassIdImage`, written and read as
a bare `(H, W)` value, because [`SemanticSegmentation2D`](../archetypes/segmentation.md) holds one
image per point in time; storage widens it to a one-row `BatchClassIdImage`, and a `range` over
several frames reads back as a stack. The segmentation metrics consume `as_class_id()`.

## Vectorized geometry

The pairwise geometry the matchers use is public, so you can call it directly:

```python
from t4perceval.geometry import (
    bev_area,
    bev_corners,
    corner_displacements,
    heading_errors,
    pairwise_bev_iou,
    pairwise_plane_distance,
    pairwise_roi_iou,
    pairwise_volume_iou,
    volume,
)

bev_corners(position, quaternion, size)  # (N, 4, 2) footprint corners
corner_displacements(
    est_position, est_quaternion, est_size, gt_position, gt_quaternion, gt_size
)  # (N,) mean corner distance of aligned pairs
heading_errors(est_quaternion, gt_quaternion)  # (N,) absolute yaw error wrapped to [0, pi]
bev_area(size)  # (N,)
volume(size)  # (N,)

pairwise_bev_iou(
    est_position,
    est_quaternion,
    est_size,
    gt_position,
    gt_quaternion,
    gt_size,
)  # (N, M)
```

Every `pairwise_*` function returns an `(N, M)` matrix over the two inputs -- which is exactly the
shape a [custom matcher](../recipes/custom-matcher.md) has to produce.

## Where to go next

- [Object components](object.md) · [Metric components](metrics.md)
- [`t4perceval.geometry` API](../reference/api/geometry.md)
