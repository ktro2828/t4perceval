# SemanticSegmentation2D / SemanticSegmentation3D

A class per labelled pixel or point. Neither archetype names its elements: **the row is the
element**, and an estimation corresponds to a ground truth because both enumerate the same elements
in the same order. See [Segmentation](../evaluation/segmentation-3d.md) for the metrics that rest
on that.

## SemanticSegmentation3D

| Component  | Requirement | Shape    | dtype | Description                              |
| :--------- | :---------- | :------- | :---- | :--------------------------------------- |
| `point`    | Required    | `(N, 3)` | `f64` | the labelled point, in the chunk's frame |
| `class_id` | Required    | `(N,)`   | `i32` | semantic class                           |

```python
from t4perceval import SemanticSegmentation3D

SemanticSegmentation3D(
    point=[[1.0, 2.0, 0.3], [1.1, 2.0, 0.3]],
    class_id=labels.encode(["car", "car"]),
)
```

## SemanticSegmentation2D

| Component  | Requirement | Shape  | dtype | Description                   |
| :--------- | :---------- | :----- | :---- | :---------------------------- |
| `class_id` | Required    | `(N,)` | `i32` | semantic class, one per pixel |

The row **is** the pixel: row `i` is pixel `(i // width, i % width)` and an image has
`height * width` rows. The image size is not a column -- it would be the same pair repeated per pixel
-- but a one-row **static** [`BatchImageSize`](../components/geometry.md#batchimagesize) on the entity:

```python
from t4perceval import SemanticSegmentation2D
from t4perceval.component import BatchImageSize
from t4perceval.descriptors import IMAGE_SIZE

segmentation = SemanticSegmentation2D.from_label_map(label_image)  # (H, W) -> N = H*W
store.log(path, segmentation, at=TimePoint.at(frame=0), frame_id="CAM_FRONT")
store.log_static_components(path, {IMAGE_SIZE: BatchImageSize([[height, width]])})

segmentation.as_label_map(height, width)  # back to (H, W)
```

`EntityView.component(IMAGE_SIZE)` broadcasts the single row over every pixel of a view;
`store.static(path)[IMAGE_SIZE]` reads it un-broadcast. `frame_id` is the camera channel, as for
every 2D archetype.

## POINT, not POSITION

`SemanticSegmentation3D.point` uses the **`POINT`** descriptor, deliberately not `POSITION`. A
labelled point is not an object with a pose, and the separate name stops
`FilterByDistanceSystem` -- or any other system asking for a 3D object position -- from being
pointed at a point cloud and appearing to work. A coordinate transform still moves it as a point.

Both use `BatchPosition3D` as the underlying column type. That is the descriptor-versus-type
distinction again: the type says what the numbers look like, the descriptor says what they mean.

## Neither has optional components

`SemanticSegmentation3D` is exactly two columns and `SemanticSegmentation2D` exactly one -- like a
mask, a column whose meaning is fixed by the entity it sits on.

## Where to go next

- [Segmentation evaluation](../evaluation/segmentation-3d.md) -- IoU, accuracy and the confusion
  matrix.
- [Data model](../concepts/data-model.md#descriptors) -- why descriptors carry the meaning.
