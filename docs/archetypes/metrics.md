# MetricValues and ConfusionMatrix

The two shapes every metric result takes.

## MetricValues

One metric's values, broken down by class and threshold.

| Component      | Requirement | Shape  | dtype | Description                                |
| :------------- | :---------- | :----- | :---- | :----------------------------------------- |
| `class_id`     | Required    | `(N,)` | `i32` | the class, or `ALL_CLASSES` (`-1`)         |
| `threshold`    | Required    | `(N,)` | `f64` | the threshold, `NaN` when the row has none |
| `metric_value` | Required    | `(N,)` | `f64` | the number, `NaN` when undefined           |
| `support`      | Required    | `(N,)` | `i64` | how many ground-truth objects it rests on  |

**Every scalar metric shares this shape, and the metric's name is the entity path it is logged to.**
A reader can therefore treat `/metrics/ap` and `/metrics/clear/mota` identically, and comparing two
runs is one query rather than a bespoke traversal per metric family.

```python
from t4perceval import MetricValues

values = store.range("/metrics/map", timeline=FRAME, time_range=TimeRange.everything()).materialize(
    MetricValues
)

values.aggregate  # the ALL_CLASSES row with a NaN threshold
values.of_class(labels.class_id("car"))  # one class, when it has exactly one row
values.value.values  # the whole column
values.support.values

MetricValues.empty()
MetricValues.from_rows([(0, 1.0, 0.93, 120), (-1, float("nan"), 0.88, 340)])
```

`aggregate` and `of_class` both raise unless the row they want occurs **exactly once**, so a metric
with several thresholds per class must be read through the `threshold` column:

```python
rows = values.class_id.values == labels.class_id("car")
dict(zip(values.threshold.values[rows], values.value.values[rows]))
```

### NaN is an answer

A class with no ground truth in range gets a `NaN` value with `support == 0` rather than being
dropped, so the shape of a result does not depend on the scene.

## ConfusionMatrix

A long-form detection confusion matrix: one row per `(ground_truth_class_id, estimation_class_id,
count)` cell.

| Component               | Requirement | Shape  | dtype | Description                        |
| :---------------------- | :---------- | :----- | :---- | :--------------------------------- |
| `ground_truth_class_id` | Required    | `(N,)` | `i32` | the true class, or background      |
| `estimation_class_id`   | Required    | `(N,)` | `i32` | the predicted class, or background |
| `count`                 | Required    | `(N,)` | `i64` | how many times that cell occurred  |

`BACKGROUND_CLASS_ID` (`-2`) on the **estimation** axis is a false negative; the same value on the
**ground-truth** axis is a false positive. The background/background cell is present so the matrix
is square, and is always zero.

```python
from t4perceval import ConfusionMatrix

matrix = store.range(
    "/metrics/confusion_matrix", timeline=FRAME, time_range=TimeRange.everything()
).materialize(ConfusionMatrix)

matrix.as_matrix()  # dense, class ids ascending, then background
matrix.as_matrix(include_background=False)
matrix.as_matrix([0, 2, 1])  # a chosen axis order
matrix.at(ground_truth_class_id=0, estimation_class_id=1)

ConfusionMatrix.empty()
ConfusionMatrix.from_rows([(0, 0, 12), (0, 1, 3), (0, -2, 5)])
```

Duplicate cells are summed by `as_matrix`, so it is safe on concatenated chunks as well as on a
single metric result. Passing `class_ids` containing the background class, or a duplicate, raises.

### Why long-form

A dense matrix would have to fix its axis order at write time, and would grow quadratically with the
registry even for a scene using three classes. Long form is a column layout like everything else, it
concatenates across frames without special handling, and it densifies on demand.

## Where each metric writes

| System                          | Entity                                                             | Shape             |
| :------------------------------ | :----------------------------------------------------------------- | :---------------- |
| `AveragePrecisionSystem`        | `/metrics/ap`                                                      | `MetricValues`    |
| `AveragePrecisionHeadingSystem` | `/metrics/aph`                                                     | `MetricValues`    |
| `MeanAveragePrecisionSystem`    | as given                                                           | `MetricValues`    |
| `ClearSystem`                   | `/metrics/clear/mota`, `/motp`, `/id_switch`                       | `MetricValues`    |
| `PathDisplacementSystem`        | `/metrics/displacement/ade`, `/fde`, `/miss_rate`                  | `MetricValues`    |
| `ClassificationSystem`          | `/metrics/classification/accuracy`, `/precision`, `/recall`, `/f1` | `MetricValues`    |
| `ConfusionMatrixSystem`         | `/metrics/confusion_matrix`                                        | `ConfusionMatrix` |

## Where to go next

- [Metrics](../user-guide/metrics.md) -- producing and reading them.
- [Metric divergences](../development/metric-divergences.md) -- where the numbers differ from the official definitions.
