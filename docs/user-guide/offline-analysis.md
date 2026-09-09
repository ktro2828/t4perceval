# Offline analysis

An evaluation's intermediate products are not discarded. Once a pipeline has run, the store holds
the inputs, every filter mask, every matching verdict and every metric value -- all as ordinary
entities, all queryable with the same two calls.

```text
/estimation/objects                      the inputs
/estimation/objects/filter/distance      what the distance filter decided
/estimation/objects/filter/confidence    what the confidence filter decided
/estimation/objects/kept                 what survived
/matching/center_distance/0              verdicts at threshold 0
/metrics/ap/0                            AP at threshold 0
/metrics/map                             the mean
```

That is the whole point of writing results back into the store: "why is this number low?" is a
query, not a re-run.

## Where did the objects go?

```python
from t4perceval.descriptors import MASK

for name in ("distance", "confidence", "label"):
    mask = store.range(
        f"/estimation/objects/filter/{name}",
        timeline=FRAME,
        time_range=TimeRange.everything(),
    )
    column = mask.component(MASK)
    print(name, column.num_selected, "of", len(column), "kept")
```

`BatchMask.indices()` gives the rows that passed, so you can go straight back to the objects:

```python
kept = store.range("/estimation/objects", timeline=FRAME, time_range=TimeRange.everything())
kept.select(mask.component(MASK).values).materialize(Detections3D)
```

## Which frames are the false positives in?

```python
import numpy as np
from t4perceval import MatchResults
from t4perceval.component import MatchStatus
from t4perceval.descriptors import MATCH_STATUS

view = store.range("/matching/center_distance/0", timeline=FRAME, time_range=TimeRange.everything())
status = view.component(MATCH_STATUS).values
frames = view.times(FRAME)

false_positives = frames[status == MatchStatus.FP]
np.unique(false_positives, return_counts=True)
```

Because the view carries the time of every row, a per-frame breakdown of a scene-wide result costs
one query.

## Which object was it?

A match row points back at its inputs by row index, and `MatchJoin` resolves those indices for you
across a whole range:

```python
from t4perceval.descriptors import CLASS_ID, CONFIDENCE, MATCHING_SCORE
from t4perceval.system import MatchJoin

join = MatchJoin.of(
    store,
    "/matching/center_distance/0",
    "/estimation/objects",
    "/ground_truth/objects",
    timeline=FRAME,
    time_range=TimeRange.everything(),
)

join.has_estimation  # which match rows point at an estimate (a property)
join.has_ground_truth
join.match_component(MATCHING_SCORE)  # a column of the match rows themselves
join.est_component(CONFIDENCE)  # an estimation column, gathered onto the match rows
join.gt_component(CLASS_ID)
join.is_label_correct()  # same class on both sides
```

`fill=` supplies the value for rows with no counterpart, so the gathered column is the same length
as the match rows.

## Comparing thresholds, and comparing runs

Every scalar metric shares the `MetricValues` shape and is named by its entity path, so a comparison
is a loop over paths rather than a bespoke traversal per metric family:

```python
from t4perceval import MetricValues

for index in range(4):
    values = store.range(
        f"/metrics/ap/{index}", timeline=FRAME, time_range=TimeRange.everything()
    ).materialize(MetricValues)
    print(index, values.value.values, values.support.values)
```

Two runs compare the same way, given two stores.

## Narrowing in time

The range is the only knob:

```python
scene = TimeRange.everything()
first_ten = TimeRange(0, 9)
just_one = TimeRange.single(3)
```

Nothing is recomputed for any of them.

## Persisting a run

```python
from t4perceval.io import read_recording, write_recording

write_recording(setup.into_recording(pipeline=systems), "result.t4eval")
recording = read_recording("result.t4eval")
```

Every query on this page works unchanged against the reopened recording: `Recording` exposes the
same read surface as `Store`, and the `.t4eval` format preserves everything those queries depend on
-- log order included. See [Persistence](persistence.md#saving-a-whole-recording).

## Where to go next

- [Querying data](querying-data.md) -- the query surface in full.
- [Metrics](metrics.md) -- what each metric writes.
