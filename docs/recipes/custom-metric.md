# Implement a custom metric

**Goal:** turn matching verdicts into a number the built-in metrics do not compute.

## The contract

Subclass `MetricSystem` and supply:

| Thing                   | What it is                                      |
| :---------------------- | :---------------------------------------------- |
| `REQUIRES`              | what the **matching** entity must carry         |
| `REQUIRES_ESTIMATION`   | what the estimation entity must carry           |
| `REQUIRES_GROUND_TRUTH` | what the ground-truth entity must carry         |
| `METRIC_NAME`           | the last path segment, `/metrics/<METRIC_NAME>` |
| `compute()`             | rows keyed by the entity each metric belongs to |

The base class builds the [`MatchJoin`](../user-guide/offline-analysis.md#which-object-was-it),
validates each of the three sources against **its own** declaration, picks the reporting time, and
wraps your rows as `MetricValues` chunks.

A row is `(class_id, threshold, value, support)`.

## Example: recall at a fixed threshold

```python
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define

from t4perceval.component import MatchStatus
from t4perceval.descriptors import CLASS_ID, EST_INDEX, GT_INDEX, MATCH_STATUS
from t4perceval.system.metric.base import MetricRow, MetricSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.entity import EntityPath
    from t4perceval.system import SystemContext
    from t4perceval.system.join import MatchJoin


@define(slots=True)
class RecallSystem(MetricSystem):
    """TP / (TP + FN) per class."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (EST_INDEX, GT_INDEX, MATCH_STATUS)
    REQUIRES_ESTIMATION: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    REQUIRES_GROUND_TRUTH: ClassVar[tuple[ComponentDescriptor, ...]] = (CLASS_ID,)
    METRIC_NAME: ClassVar[str] = "recall"

    def compute(self, join: MatchJoin, ctx: SystemContext) -> dict[EntityPath, list[MetricRow]]:
        rows: list[MetricRow] = []
        classes = self.classes(ctx, join)

        if not len(join.matches):
            return {self.target: [(int(c), float("nan"), float("nan"), 0) for c in classes]}

        status = join.match_component(MATCH_STATUS)
        gt_class = join.gt_component(CLASS_ID)
        is_tp = (status == int(MatchStatus.TP)) & join.is_label_correct()
        is_fn = status == int(MatchStatus.FN)

        for class_id in classes:
            of_class = gt_class == class_id
            true_positive = int(np.count_nonzero(is_tp & of_class))
            false_negative = int(np.count_nonzero(is_fn & of_class))
            support = true_positive + false_negative
            value = true_positive / support if support else float("nan")
            rows.append((int(class_id), float("nan"), value, support))

        return {self.target: rows}
```

Use it like any other metric:

```python
from t4perceval.system import CenterDistanceMatchingSystem, Pipeline

matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
recall = RecallSystem.on(matcher.target, EST, GT)
recall.target  # /metrics/recall

Pipeline([matcher, recall]).run(ctx, TimeRange.everything())
```

## Rules to follow

- **Use `self.classes(ctx, join)`.** It returns the registry's classes when there is one, so a class
  with no objects still gets a row saying so instead of silently vanishing from the report.
- **`NaN` is an answer.** An undefined value with `support == 0` keeps the result's shape
  independent of the scene. Do not skip the row.
- **Set `support` honestly.** It is what a later mean weights by, and what tells a reader whether a
  number means anything.
- **Reach inputs through the join,** not through `ctx.store`. `est_component` / `gt_component`
  gather a column onto the match rows, with `fill=` for rows that have no counterpart.
- **Check the label**, do not assume it. A class-agnostic matcher can pair across classes;
  `join.is_label_correct()` is the check.
- **Handle the empty join.** `len(join.matches) == 0` is an ordinary scene, not an error.
- **Declare each source separately.** The three carry different components, and `REQUIRES` is
  specifically what the _matching_ entity must have -- that is what `Pipeline` uses to link your
  metric to the matcher that feeds it.

## Several results from one computation

Override `targets` and return one entry per target:

```python
@property
def targets(self) -> tuple[EntityPath, ...]:
    root = as_entity_path(self.target)
    return (root / "precision", root / "recall")


def compute(self, join, ctx) -> dict[EntityPath, list[MetricRow]]:
    precision_target, recall_target = self.targets
    ...
    return {precision_target: precision_rows, recall_target: recall_rows}
```

This is what `ClearSystem` does for MOTA / MOTP / ID switches -- one pass over the identities, three
entities out -- and what lets `Pipeline` still tell who produces what.

## Averaging several metric entities

You do not need a custom system for this. `MeanAveragePrecisionSystem` reads `MetricValues` and
writes `MetricValues`, so it averages **any** scalar metric:

```python
from t4perceval.system import MeanAveragePrecisionSystem

MeanAveragePrecisionSystem.of(["/metrics/recall/0", "/metrics/recall/1"], target="/metrics/mrecall")
```

Use `nan_mean` from `t4perceval.system.metric.base` when you average inside your own metric: it
skips the undefined values instead of poisoning the result.

## A structured result instead of scalars

If your metric is not one number per class -- a confusion matrix, a curve -- override the
serialization the way `ConfusionMatrixSystem` does, while reusing the source wiring and class
discovery.

## A metric with no matching stage

`MetricSystem` assumes three sources and a `MatchJoin`. A metric over element-wise-aligned data --
segmentation, say -- has nothing to join. Subclass `SegmentationMetricSystem` instead: it takes
`(estimation, ground_truth)`, validates the row alignment per frame, applies `ignore=`, and hands
you the pooled `(ground-truth class, estimated class)` count matrix in `emit()`:

```python
from t4perceval.system import SegmentationMetricSystem


@define(slots=True)
class FrequencyWeightedIoUSystem(SegmentationMetricSystem):
    PROVIDES: ClassVar[tuple[ComponentDescriptor, ...]] = MetricValues.required_descriptors()
    METRIC_NAME: ClassVar[str] = "segmentation/fw_iou"

    def emit(
        self, counts, axes, ctx, at
    ): ...  # rows are ground truth, columns estimation; axes[-1] is BACKGROUND_CLASS_ID
```

For data that is neither matched nor row-aligned, implement the `System` protocol directly. See
[Extending systems](../development/extending-systems.md).

## Where to go next

- [Metrics](../user-guide/metrics.md) -- the built-in family.
- [MetricValues](../archetypes/metrics.md) -- the output shape.
