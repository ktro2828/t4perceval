# MatchResults

The outcome of matching an estimation stream against a ground-truth stream. **One row per verdict.**

## Schema

| Component        | Requirement | Shape  | dtype | Description                                          |
| :--------------- | :---------- | :----- | :---- | :--------------------------------------------------- |
| `est_index`      | Required    | `(N,)` | `i64` | row index into the estimation chunk, `-1` for none   |
| `gt_index`       | Required    | `(N,)` | `i64` | row index into the ground-truth chunk, `-1` for none |
| `matching_score` | Required    | `(N,)` | `f64` | the score this pair got                              |
| `match_status`   | Required    | `(N,)` | `i8`  | `MatchStatus.TP` / `FP` / `FN`                       |
| `threshold`      | Required    | `(N,)` | `f64` | the threshold the verdict was reached at             |

No optional components.

## Reading it

```python
from t4perceval import FRAME, MatchResults, TimeRange
from t4perceval.component import MatchStatus

result = store.range(
    "/matching/center_distance", timeline=FRAME, time_range=TimeRange.everything()
).materialize(MatchResults)

result.num_tp, result.num_fp, result.num_fn
result.count(MatchStatus.FP)
MatchResults.empty()
```

A false positive is a row with `gt_index == -1`, a false negative one with `est_index == -1`.
An unknown `match_status` value raises at construction.

## Why row indices

Storing indices rather than object references is what makes a match result **storable and
re-analysable**. `DynamicObjectWithPerceptionResult` in the original package held references, so it
could be neither persisted nor queried after the fact.

The indices are per-partition -- they point into the frame the verdict was reached in. Over a
multi-frame range, [`MatchJoin`](../user-guide/offline-analysis.md#which-object-was-it) resolves
them into indices over the whole range for you:

```python
from t4perceval.descriptors import CLASS_ID, CONFIDENCE
from t4perceval.system import MatchJoin

join = MatchJoin.of(
    store,
    "/matching/center_distance",
    "/estimation/objects",
    "/ground_truth/objects",
    timeline=FRAME,
    time_range=TimeRange.everything(),
)
join.est_component(CONFIDENCE)  # gathered onto the match rows, NaN where there is none
join.gt_component(CLASS_ID)
join.is_label_correct()
```

## Why threshold is stored

Everything else a metric needs can be recovered by following the indices back to the objects. The
threshold cannot -- only the matcher knew it, and with
[per-class thresholds](../user-guide/matching.md#thresholds) it differs from row to row.

## Where it is written

`/matching/<mode>` by default, where `<mode>` is the matcher's `MATCHING_NAME`:

```text
/matching/center_distance
/matching/center_distance_bev
/matching/plane_distance
/matching/iou_bev
/matching/iou_3d
/matching/iou_roi
```

`average_precision_sweep` indexes them by declaration order --
`/matching/center_distance/0`, `/1`, ... -- because a per-class threshold has no single value to put
in a path.

## Where to go next

- [Matching](../user-guide/matching.md) -- producing it.
- [MetricValues](metrics.md) -- what a metric turns it into.
