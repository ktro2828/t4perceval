# Analyse a finished evaluation

**Goal:** a pipeline has run and a number looks wrong. Find out why, without re-running anything.

Everything an evaluation produced is still in the store, as ordinary entities. This page is a set of
queries; the surface they use is described in [Offline analysis](../user-guide/offline-analysis.md)
and [Querying data](../user-guide/querying-data.md).

## Start by listing what is there

```python
sorted(str(path) for path in store.entity_paths())
```

```text
/estimation/objects
/estimation/objects/filter/confidence
/estimation/objects/filter/distance
/estimation/objects/filter/keep
/estimation/objects/kept
/ground_truth/objects
/ground_truth/objects/filter/distance
/ground_truth/objects/kept
/matching/center_distance/0
/matching/center_distance/1
/metrics/ap/0
/metrics/ap/1
/metrics/map
```

That list is the evaluation's own record of what it did.

## "Recall is low"

Work backwards: how many ground-truth objects survived filtering, and how many were matched?

```python
from t4perceval import FRAME, MatchResults, TimeRange
from t4perceval.descriptors import MASK

scene = TimeRange.everything()

kept = store.range("/ground_truth/objects/filter/distance", timeline=FRAME, time_range=scene)
column = kept.component(MASK)
print("ground truth kept:", column.num_selected, "of", len(column))

matches = store.range("/matching/center_distance/0", timeline=FRAME, time_range=scene).materialize(
    MatchResults
)
print("TP", matches.num_tp, "FP", matches.num_fp, "FN", matches.num_fn)
```

If the filter dropped most of the ground truth, the metric is measuring a smaller problem than you
think. If it did not, the misses are the matcher's.

## "Which frames are bad?"

```python
import numpy as np
from t4perceval.component import MatchStatus
from t4perceval.descriptors import MATCH_STATUS

view = store.range("/matching/center_distance/0", timeline=FRAME, time_range=scene)
status = view.component(MATCH_STATUS).values
frames = view.times(FRAME)

for name, code in (("FP", MatchStatus.FP), ("FN", MatchStatus.FN)):
    bad, counts = np.unique(frames[status == code], return_counts=True)
    worst = np.argsort(counts)[::-1][:5]
    print(name, list(zip(bad[worst].tolist(), counts[worst].tolist())))
```

Then look at one of them:

```python
store.range("/estimation/objects", timeline=FRAME, time_range=TimeRange.single(17)).materialize(
    Detections3D
)
```

## "Which objects are being missed?"

`MatchJoin` resolves a match row's indices back to the objects it points at:

```python
from t4perceval.descriptors import CLASS_ID, CONFIDENCE, MATCHING_SCORE
from t4perceval.system import MatchJoin

join = MatchJoin.of(
    store,
    "/matching/center_distance/0",
    "/estimation/objects/kept",
    "/ground_truth/objects/kept",
    timeline=FRAME,
    time_range=scene,
)

missed = ~join.has_estimation  # false negatives
labels.decode(join.gt_component(CLASS_ID)[missed].astype(int))
```

And the near-misses -- pairs the matcher scored but rejected:

```python
score = join.match_component(MATCHING_SCORE)
np.nanpercentile(score[join.has_estimation & join.has_ground_truth], [50, 90, 99])
```

If the 90th percentile is just above your threshold, the threshold is the problem, not the model.

## "Is it a labelling problem?"

```python
from t4perceval import ConfusionMatrix

matrix = store.range("/metrics/confusion_matrix", timeline=FRAME, time_range=scene).materialize(
    ConfusionMatrix
)
matrix.as_matrix()
```

Run `ConfusionMatrixSystem` behind a **class-agnostic** matcher for this, so a car detected as a
truck lands in the off-diagonal cell instead of being counted as one FP plus one FN.

```python
CenterDistanceMatchingSystem.between(
    EST, GT, threshold=1.0, class_agnostic=True, target="/matching/agnostic"
)
```

## "How does the threshold change it?"

```python
from t4perceval import MetricValues

for index, threshold in enumerate([0.5, 1.0, 2.0, 4.0]):
    values = store.range(f"/metrics/ap/{index}", timeline=FRAME, time_range=scene).materialize(
        MetricValues
    )
    print(threshold, values.value.values.round(4), values.support.values)
```

A metric that jumps between two neighbouring thresholds is dominated by localisation error; one that
does not move is dominated by classification or by misses.

## "What did the filters cost me?"

```python
for name in ("distance", "confidence", "keep"):
    column = store.range(
        f"/estimation/objects/filter/{name}", timeline=FRAME, time_range=scene
    ).component(MASK)
    print(name, column.num_selected, "of", len(column))
```

Because the masks are all still there, you can see each filter's contribution _and_ their
combination, without re-running any of them.

## Narrowing to one frame

Every query on this page takes `TimeRange.single(frame)` instead of `TimeRange.everything()`, at no
extra cost -- nothing is recomputed.

## Saving the answers

Save the whole recording rather than the answers: every query on this page then works unchanged
against the reopened directory, and nothing has to be recomputed to ask a new one later.

```python
from t4perceval.io import read_recording, write_recording

write_recording(setup.into_recording(pipeline=systems), "run.t4eval")
recording = read_recording("run.t4eval")
recording.range("/metrics/map", timeline=FRAME, time_range=scene).materialize(MetricValues)
```

To hand one entity to an Arrow tool instead, `write_parquet` still writes a single chunk:

```python
from t4perceval.io import write_parquet

chunk = store.range("/metrics/map", timeline=FRAME, time_range=scene).to_chunk()
write_parquet(chunk, "run/metrics_map.parquet", labels=labels)
```

See [Persistence](../user-guide/persistence.md#saving-a-whole-recording).

## Where to go next

- [Offline analysis](../user-guide/offline-analysis.md) -- the query surface.
- [Metric divergences](../development/metric-divergences.md) -- when the number is right but unexpected.
