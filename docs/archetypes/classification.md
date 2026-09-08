# Classifications2D

A class and a confidence per object, without any geometry.

## Schema

| Component     | Requirement | Shape  | dtype | Description    |
| :------------ | :---------- | :----- | :---- | :------------- |
| `class_id`    | Required    | `(N,)` | `i32` | semantic class |
| `confidence`  | Required    | `(N,)` | `f64` | in `[0, 1]`    |
| `instance_id` | Optional    | `(N,)` | `i64` | keyword-only   |

```python
from t4perceval import Classifications2D

Classifications2D(
    class_id=labels.encode(["car", "pedestrian"]),
    confidence=[0.9, 0.7],
)
```

## When to use it

For a classifier whose association to the ground truth is already known -- a crop classifier, or a
re-labelling of boxes that are not themselves under test. Without geometry there is nothing for a
matcher to score, so the pairing has to come from somewhere else, normally the row order or an
`instance_id`.

If your objects do have geometry, use [`Detections2D`](detection-2d.md) or
[`Detections3D`](detection-3d.md) instead: `ClassificationSystem` and `ConfusionMatrixSystem` both
work off a `MatchResults`, which a geometric matcher produces.

## What it unlocks

| Component present | Enables                                                        |
| :---------------- | :------------------------------------------------------------- |
| `class_id`        | `FilterByLabelSystem`, and every metric's class axis           |
| `confidence`      | `FilterByConfidenceSystem`, `AveragePrecisionSystem`'s ranking |
| `instance_id`     | `FilterByInstanceSystem`                                       |

No matcher requires only these two columns, so a classification evaluation needs either a matcher
over richer inputs or a [custom system](../recipes/custom-metric.md).

## Where to go next

- [Metrics](../user-guide/metrics.md#classification-and-the-confusion-matrix)
- [ConfusionMatrix](metrics.md#confusionmatrix)
