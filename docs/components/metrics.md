# Metric components

The columns a system writes: a filter's verdict, a matcher's verdicts, and a metric's numbers.

## BatchMask

`(N,)` `bool` -- a filter's verdict, one row per source row.

```python
from t4perceval.component import BatchMask

mask = BatchMask([True, True, False])
mask.num_selected  # 2
mask.indices()  # array([0, 1])
```

Descriptor: `MASK`. Filters emit a mask instead of dropping rows, so the reason a row was excluded
stays inspectable. See [Filtering](../user-guide/filtering.md).

## Matching columns

Five columns make up [`MatchResults`](../archetypes/matching.md), one row per verdict.

| Component            | Descriptor       | Shape  | dtype | Meaning                                              |
| :------------------- | :--------------- | :----- | :---- | :--------------------------------------------------- |
| `BatchRowIndex`      | `EST_INDEX`      | `(N,)` | `i64` | row index into the estimation chunk, `-1` for none   |
| `BatchRowIndex`      | `GT_INDEX`       | `(N,)` | `i64` | row index into the ground-truth chunk, `-1` for none |
| `BatchMatchingScore` | `MATCHING_SCORE` | `(N,)` | `f64` | the score the pair got                               |
| `BatchMatchStatus`   | `MATCH_STATUS`   | `(N,)` | `i8`  | `MatchStatus.TP` / `FP` / `FN`                       |
| `BatchThreshold`     | `THRESHOLD`      | `(N,)` | `f64` | the threshold the verdict was reached at             |

```python
from t4perceval.component import MatchStatus

int(MatchStatus.TP)  # 0 -- a matched pair within the threshold
int(MatchStatus.FP)  # 1 -- an estimation with no acceptable ground truth
int(MatchStatus.FN)  # 2 -- a ground-truth object no estimation claimed
```

Row indices rather than object references are what make a match result **storable and
re-analysable**: `DynamicObjectWithPerceptionResult` in the original package held references and
therefore could be neither saved nor queried.

`THRESHOLD` is recorded because it is the one thing a later stage could not recover by following
the indices back to the objects -- only the matcher knew it, and with per-class thresholds it
differs from row to row.

## Metric value columns

Four columns make up [`MetricValues`](../archetypes/metrics.md).

| Component          | Descriptor     | Shape  | dtype | Meaning                                    |
| :----------------- | :------------- | :----- | :---- | :----------------------------------------- |
| `BatchClassId`     | `CLASS_ID`     | `(N,)` | `i32` | the class, or `ALL_CLASSES` (`-1`)         |
| `BatchThreshold`   | `THRESHOLD`    | `(N,)` | `f64` | the threshold, `NaN` when the row has none |
| `BatchMetricValue` | `METRIC_VALUE` | `(N,)` | `f64` | the number, `NaN` when undefined           |
| `BatchSupport`     | `SUPPORT`      | `(N,)` | `i64` | how many ground-truth objects it rests on  |

`NaN` is a first-class answer here: a class with no ground truth in range gets a `NaN` value with
`support == 0` rather than being dropped, so the shape of a result does not depend on the scene.

## BatchCount

`(N,)` `i64`, non-negative. The cell counts of a
[`ConfusionMatrix`](../archetypes/metrics.md#confusionmatrix).

Descriptor: `COUNT`, alongside `GROUND_TRUTH_CLASS_ID` and `ESTIMATION_CLASS_ID`.

## Sentinel class ids

| Value | Constant                           | Where it appears                                                                 |
| ----: | :--------------------------------- | :------------------------------------------------------------------------------- |
|  `-1` | `ALL_CLASSES` / `UNKNOWN_CLASS_ID` | a `MetricValues` row aggregating over classes                                    |
|  `-2` | `BACKGROUND_CLASS_ID`              | a `ConfusionMatrix` axis: FN on the estimation side, FP on the ground-truth side |

```python
from t4perceval.component import ALL_CLASSES, BACKGROUND_CLASS_ID
```

## Where to go next

- [MatchResults](../archetypes/matching.md) · [MetricValues](../archetypes/metrics.md)
- [Metrics](../user-guide/metrics.md) -- reading them.
