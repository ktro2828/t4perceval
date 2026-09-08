# SemanticSegmentation2D / SemanticSegmentation3D

A class per labelled pixel or point.

!!! warning "No metric system reads these yet"

    The archetypes exist and round-trip through the store and Parquet. Segmentation metrics are on
    the [roadmap](../development/roadmap.md); see
    [Segmentation 3D](../evaluation/segmentation-3d.md) for what you can do today.

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

| Component  | Requirement | Shape  | dtype | Description                     |
| :--------- | :---------- | :----- | :---- | :------------------------------ |
| `pixel`    | Required    | `(N,)` | `i32` | flat pixel index into the image |
| `class_id` | Required    | `(N,)` | `i32` | semantic class                  |

```python
from t4perceval import SemanticSegmentation2D

SemanticSegmentation2D(pixel=[10234, 10235], class_id=labels.encode(["road", "road"]))
```

The index is flat -- `row * width + column` -- so the image width has to be known out of band. Log
it as static data on the entity if you need it to travel with the rows.

## POINT, not POSITION

`SemanticSegmentation3D.point` uses the **`POINT`** descriptor, deliberately not `POSITION`. A
labelled point is not an object with a pose, and the separate name stops
`FilterByDistanceSystem` -- or any other system asking for a 3D object position -- from being
pointed at a point cloud and appearing to work.

Both use `BatchPosition3D` as the underlying column type. That is the descriptor-versus-type
distinction again: the type says what the numbers look like, the descriptor says what they mean.

## Neither has optional components

Both archetypes are exactly two columns.

## Where to go next

- [Segmentation 3D evaluation](../evaluation/segmentation-3d.md) -- what is and is not supported.
- [Data model](../concepts/data-model.md#descriptors) -- why descriptors carry the meaning.
