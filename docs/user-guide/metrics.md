# Metrics

A metric reads a matching result and the two entities it matched, and writes
[`MetricValues`](../archetypes/metrics.md) -- one row per class, per threshold.

```python
from t4perceval.system import AveragePrecisionSystem

ap = AveragePrecisionSystem.on(
    matcher.target,
    "/estimation/objects",
    "/ground_truth/objects",
)
ap.target  # /metrics/ap
```

## The metric family

| System                          | Writes                                                             | Needs from the objects                     |
| :------------------------------ | :----------------------------------------------------------------- | :----------------------------------------- |
| `AveragePrecisionSystem`        | `/metrics/ap`                                                      | `class_id`, `confidence`                   |
| `AveragePrecisionHeadingSystem` | `/metrics/aph`                                                     | `class_id`, `confidence`, `quaternion`     |
| `MeanAveragePrecisionSystem`    | one entity, averaging several                                      | -- (reads metric entities)                 |
| `ClearSystem`                   | `/metrics/clear/mota`, `/motp`, `/id_switch`                       | `class_id`, `instance_id`                  |
| `PathDisplacementSystem`        | `/metrics/displacement/ade`, `/fde`, `/miss_rate`                  | `class_id`, `waypoints`, `mode_confidence` |
| `ClassificationSystem`          | `/metrics/classification/accuracy`, `/precision`, `/recall`, `/f1` | `class_id`                                 |
| `ConfusionMatrixSystem`         | `/metrics/confusion_matrix`                                        | `class_id`                                 |

A metric that produces several results from one shared computation -- MOTA, MOTP and ID switches all
come out of the same identity tracking -- writes **one entity per result** and lists them all in
`targets`:

```python
from t4perceval.system import ClearSystem

clear = ClearSystem.on(matcher.target, EST, GT)
[str(t) for t in clear.targets]
# ['/metrics/clear/mota', '/metrics/clear/motp', '/metrics/clear/id_switch']
```

Move the family root with `target=`:

```python
ClearSystem.on(matcher.target, EST, GT, target="/metrics/clear_bev")
# → /metrics/clear_bev/mota, /motp, /id_switch
```

## Reading MetricValues

Every scalar metric shares one shape, and the metric's **name is the entity path** it is logged to.
A reader can therefore treat `/metrics/ap` and `/metrics/mota` identically, and comparing two runs is
one query rather than a bespoke traversal per metric family.

| Column      | Meaning                                                              |
| :---------- | :------------------------------------------------------------------- |
| `class_id`  | the class, or `ALL_CLASSES` (`-1`) for a row aggregated over classes |
| `threshold` | the threshold the value was produced at, `NaN` when it has none      |
| `value`     | the number                                                           |
| `support`   | how many ground-truth objects the value rests on                     |

```python
from t4perceval import MetricValues

values = store.range("/metrics/map", timeline=FRAME, time_range=TimeRange.everything()).materialize(
    MetricValues
)

values.aggregate  # the ALL_CLASSES / NaN-threshold row
values.of_class(labels.class_id("car"))  # one class, when it has exactly one row
values.value.values  # every row, as an array
values.support.values
```

`of_class` raises if a class has several rows -- a metric with several thresholds per class must be
read through the `threshold` column instead:

```python
import numpy as np

car = labels.class_id("car")
rows = values.class_id.values == car
dict(zip(values.threshold.values[rows], values.value.values[rows]))
```

A class with no ground truth in range gets a row with `NaN` and `support == 0`, rather than being
dropped -- so the shape of the result does not depend on the scene.

## Detection: AP, APH and mAP

```python
from t4perceval.system import Pipeline, average_precision_sweep

Pipeline(
    average_precision_sweep(
        "/estimation/objects",
        "/ground_truth/objects",
        thresholds=[0.5, 1.0, 2.0, 4.0],
        heading=True,
    )
).run(ctx, TimeRange.everything())
```

That writes `/metrics/ap/<i>` and `/metrics/aph/<i>` per threshold, then `/metrics/map` and
`/metrics/maph` averaging over thresholds and classes.

`AveragePrecisionSystem` interpolates precision over a recall grid; the defaults follow the
incumbent's:

| Parameter           | Default | Meaning                         |
| :------------------ | ------: | :------------------------------ |
| `min_recall`        |     0.1 | recall below this is discarded  |
| `min_precision`     |     0.1 | precision below this is clipped |
| `num_recall_points` |     101 | the recall grid                 |

`AveragePrecisionHeadingSystem` weights each true positive by heading agreement.

## Tracking: CLEAR

```python
from t4perceval.system import ClearSystem

ClearSystem.on(matcher.target, EST, GT)
```

MOTA, MOTP and ID switches, from one pass over the tracked identities. Both entities need an
`instance_id` column, so the inputs must be `Trackings3D` / `Trackings2D` / `Predictions3D`, not
plain detections.

## Prediction: ADE, FDE, miss rate

```python
from t4perceval.system import PathDisplacementSystem

PathDisplacementSystem.on(matcher.target, EST, GT, top_k=3, miss_tolerance=2.0)
```

| Parameter        | Default | Meaning                                                     |
| :--------------- | ------: | :---------------------------------------------------------- |
| `top_k`          |       3 | how many highest-confidence modes are considered            |
| `miss_tolerance` |     2.0 | final-displacement distance above which a prediction misses |
| `kernel`         |  `None` | which displacement kernel to use                            |

## Classification and the confusion matrix

```python
from t4perceval.system import ClassificationSystem, ConfusionMatrixSystem

ClassificationSystem.on(matcher.target, EST, GT)  # accuracy, precision, recall, f1
ConfusionMatrixSystem.on(matcher.target, EST, GT)  # long-form count matrix
```

Both are most useful behind a **class-agnostic** matcher, so a car detected as a truck lands in the
off-diagonal cell rather than being counted as one FP plus one FN.

### Confusion matrix

`ConfusionMatrix` is long-form -- one row per `(ground_truth_class_id, estimation_class_id, count)`
cell -- and densifies on demand:

```python
from t4perceval import ConfusionMatrix

matrix = store.range(
    "/metrics/confusion_matrix", timeline=FRAME, time_range=TimeRange.everything()
).materialize(ConfusionMatrix)

matrix.as_matrix()  # ordered by class id, then background
matrix.as_matrix(include_background=False)
matrix.at(labels.class_id("car"), labels.class_id("truck"))
```

`BACKGROUND_CLASS_ID` (`-2`) on the estimation axis is a false negative; the same value on the
ground-truth axis is a false positive. Duplicate cells are summed, so `as_matrix` is safe on
concatenated chunks.

## Averaging several metric entities

```python
from t4perceval.system import MeanAveragePrecisionSystem

MeanAveragePrecisionSystem.of(
    ["/metrics/ap/0", "/metrics/ap/1", "/metrics/ap/2"],
    target="/metrics/map",
)
```

It reads `MetricValues` and writes `MetricValues`, so it composes with any scalar metric, not only
AP.

## Per-frame and per-scene from one run

The range is the only difference:

```python
scene = store.range("/metrics/map", timeline=FRAME, time_range=TimeRange.everything())
```

Metrics are computed over the range the pipeline ran on, so to get per-frame values run the pipeline
per frame, writing to a per-frame target -- or read the underlying `MatchResults` per frame, which
costs nothing:

```python
per_frame = [
    store.range(matcher.target, timeline=FRAME, time_range=TimeRange.single(f)).materialize(
        MatchResults
    )
    for f in store.times("/ground_truth/objects", FRAME)
]
```

## Known divergences

Several metric implementations differ from the official benchmark definitions -- APH's denominator,
how ID switches are counted, whether MOTA may go negative, and prediction metrics ignoring validity
masks. They are catalogued, with the reasoning, in
[Metric divergences](../development/metric-divergences.md), and the
[benchmark](../development/benchmarks.md) checks every difference against that list.

## Where to go next

- [Evaluation tasks](../evaluation/detection-3d.md) -- which metric goes with which inputs.
- [Write a custom metric](../recipes/custom-metric.md).
