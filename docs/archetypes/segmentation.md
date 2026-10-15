# SemanticImage / SemanticPointCloud

A class per labelled pixel or point. Neither archetype names its elements: an estimation
corresponds to a ground truth because both enumerate the same elements in the same order -- the
points of one cloud, one per row, or the pixels of one image, one `(H, W)` image per row. See
[Segmentation](../evaluation/segmentation-3d.md) for the metrics that rest on that.

## SemanticPointCloud

| Component  | Requirement | Shape    | dtype | Description                              |
| :--------- | :---------- | :------- | :---- | :--------------------------------------- |
| `point`    | Required    | `(N, 3)` | `f64` | the labelled point, in the chunk's frame |
| `class_id` | Required    | `(N,)`   | `i32` | semantic class                           |

This is [`PointCloud`](point-cloud.md) plus `class_id`, by composition: `point` resolves to the
same `POINT` descriptor, so every system that runs on a bare cloud runs on a labelled one.

```python
from t4perceval import SemanticPointCloud

SemanticPointCloud(
    point=[[1.0, 2.0, 0.3], [1.1, 2.0, 0.3]],
    class_id=labels.encode(["car", "car"]),
)
```

## SemanticImage

| Component        | Requirement | Shape         | dtype | Description                    |
| :--------------- | :---------- | :------------ | :---- | :----------------------------- |
| `class_id_image` | Required    | `(H, W)` mono | `i32` | one class per pixel, one image |

The one component is a **mono** [`ClassIdImage`](../components/geometry.md#batchclassidimage): an
entity holds one image per point in time, as it holds one transform, so `len()` is 1 and the image
is read back as a value, not a column. The image size is not logged separately -- it is the row shape
of the column.

```python
from t4perceval import SemanticImage

segmentation = SemanticImage(class_id_image=label_image)  # (H, W) int
store.log(path, segmentation, at=TimePoint.at(frame=0), frame_id="CAM_FRONT")

segmentation.class_id_image.value  # the (H, W) image
segmentation.class_id_image.height, segmentation.class_id_image.width
```

Every image on an entity shares one column, and a wildcard dimension is inferred once per column, so
the resolution is an invariant of the entity: a `range` that spans a resolution change raises in
`concat_chunks` (`column class_id_image has row shapes (1080, 1920) and (720, 1280)`) instead of
misaligning pixels. `latest_at(...).materialize(SemanticImage)` reads one image back; a
`range` over several frames is a stack of images, read as a column with
`view.component(CLASS_ID_IMAGE)` of shape `(frames, H, W)`, not as this archetype. `frame_id` is the
camera channel, as for every 2D archetype.

## POINT, not POSITION

`SemanticPointCloud.point` uses the **`POINT`** descriptor, deliberately not `POSITION`. A
labelled point is not an object with a pose, and the separate name stops
`FilterByDistanceSystem` -- or any other system asking for a 3D object position -- from being
pointed at a point cloud and appearing to work. A coordinate transform still moves it as a point,
and the same predicates over points are their own systems: `FilterPointsByDistanceSystem`,
`FilterPointsByRegionSystem`, `FilterPointsByPolarGridSystem` and `FilterPointsByMapSystem` in
[`t4perceval.system.filter.point`](../systems/filters.md#point).

Both use `BatchPosition3D` as the underlying column type. That is the descriptor-versus-type
distinction again: the type says what the numbers look like, the descriptor says what they mean.

## Neither has optional components

`SemanticPointCloud` is exactly two columns and `SemanticImage` exactly one mono image
-- like a mask, a component whose meaning is fixed by the entity it sits on.

## Where to go next

- [Segmentation evaluation](../evaluation/segmentation-3d.md) -- IoU, accuracy and the confusion
  matrix.
- [Data model](../concepts/data-model.md#descriptors) -- why descriptors carry the meaning.
