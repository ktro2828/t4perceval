# First evaluation

The smallest complete workflow: log ground truth and estimations, narrow them to the region you
care about, match them, and read out mAP. Everything here runs on the base install -- no dataset,
no extras.

## The whole thing

```python
from t4perceval import (
    FRAME,
    Detections3D,
    LabelRegistry,
    MatchResults,
    MetricValues,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.descriptors import MASK
from t4perceval.system import (
    ApplyMaskSystem,
    FilterByDistanceSystem,
    Pipeline,
    SystemContext,
    average_precision_sweep,
)

labels = LabelRegistry.from_names(["car", "pedestrian"])
store = Store()

# 1. Log the inputs -----------------------------------------------------------
for frame in range(3):
    store.log(
        "/ground_truth/objects",
        Detections3D(
            position=[[float(frame), 0.0, 0.0], [10.0, 5.0, 0.0], [80.0, 0.0, 0.0]],
            quaternion=[[0.0, 0.0, 0.0, 1.0]] * 3,
            size=[[1.9, 4.5, 1.6]] * 3,
            class_id=labels.encode(["car", "pedestrian", "car"]),
            confidence=[1.0, 1.0, 1.0],
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )
    store.log(
        "/estimation/objects",
        Detections3D(
            position=[[float(frame) + 0.3, 0.0, 0.0], [10.4, 5.0, 0.0], [30.0, 30.0, 0.0]],
            quaternion=[[0.0, 0.0, 0.0, 1.0]] * 3,
            size=[[1.9, 4.5, 1.6]] * 3,
            class_id=labels.encode(["car", "pedestrian", "car"]),
            confidence=[0.9, 0.7, 0.4],
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )

ctx = SystemContext(store, FRAME, labels=labels)
scene = TimeRange.everything()

# 2. Narrow the inputs --------------------------------------------------------
narrow = []
for path in ("/ground_truth/objects", "/estimation/objects"):
    near = FilterByDistanceSystem.on(path, max_distance=50.0)
    narrow += [near, ApplyMaskSystem.of(path, near.target)]
Pipeline(narrow).run(ctx, scene)

# 3. Evaluate what survived ---------------------------------------------------
Pipeline(
    average_precision_sweep(
        "/estimation/objects/kept",
        "/ground_truth/objects/kept",
        thresholds=[0.5, 1.0, 2.0],
    )
).run(ctx, scene)

# 4. Read the answers ---------------------------------------------------------
matches = store.range("/matching/center_distance/1", timeline=FRAME, time_range=scene).materialize(
    MatchResults
)
print("TP/FP/FN:", matches.num_tp, matches.num_fp, matches.num_fn)

m_ap = store.range("/metrics/map", timeline=FRAME, time_range=scene).materialize(MetricValues)
print("mAP:", round(m_ap.aggregate, 4))
print("car AP:", round(m_ap.of_class(labels.class_id("car")), 4))
```

```text
TP/FP/FN: 6 3 0
mAP: 0.9969
car AP: 0.9938
```

## What each step did

### 1. Logging

Each `store.log` writes one **chunk**: the archetype's columns, the time it was observed at, and
the coordinate frame it is expressed in. Six writes, three frames, two entities.

Nothing about this call says "this is a detection evaluation". The store holds columns; what you
do with them is step 3.

### 2. Narrowing

`FilterByDistanceSystem` does not delete rows. It writes a boolean **mask** to
`/ground_truth/objects/filter/distance`, so the verdict stays queryable:

```python
mask = store.range("/ground_truth/objects/filter/distance", timeline=FRAME, time_range=scene)
mask.component(MASK).values
# array([ True,  True, False,  True,  True, False,  True,  True, False])
```

The object at 80 m fell outside the 50 m radius in all three frames.

`ApplyMaskSystem` then materializes the surviving rows into `/ground_truth/objects/kept`. Metrics
need this, not just the mask: recall divides by the number of ground-truth objects, so the
denominator has to _be_ the filtered set.

!!! note "Why two `Pipeline` runs?"

    `ApplyMaskSystem` copies whatever columns its source happens to hold, so it cannot declare what
    it provides. `Pipeline` validates wiring by component, and would reject a matcher reading an
    entity nothing declares it fills. Running the narrowing and the evaluation as two pipelines is
    the supported shape. See [Filtering](../user-guide/filtering.md#materializing-a-filtered-set).

### 3. Evaluating

`average_precision_sweep` is not magic and not a config value -- it is a plain function returning
a list of systems you can print, edit or extend:

```python
[type(s).__name__ for s in average_precision_sweep("/est", "/gt", thresholds=[0.5, 1.0, 2.0])]
# ['CenterDistanceMatchingSystem', 'AveragePrecisionSystem',
#  'CenterDistanceMatchingSystem', 'AveragePrecisionSystem',
#  'CenterDistanceMatchingSystem', 'AveragePrecisionSystem',
#  'MeanAveragePrecisionSystem']
```

Each threshold gets its own matching run, because a threshold changes which pairs the assignment is
allowed to make.

### 4. Reading

Every intermediate is a queryable entity, not a discarded temporary:

```python
sorted(str(path) for path in store.entity_paths())
```

```text
/estimation/objects
/estimation/objects/filter/distance
/estimation/objects/kept
/ground_truth/objects
/ground_truth/objects/filter/distance
/ground_truth/objects/kept
/matching/center_distance/0
/matching/center_distance/1
/matching/center_distance/2
/metrics/ap/0
/metrics/ap/1
/metrics/ap/2
/metrics/map
```

Which means one frame costs no recomputation -- it is the same query with a narrower range:

```python
store.range(
    "/matching/center_distance/1", timeline=FRAME, time_range=TimeRange.single(0)
).materialize(MatchResults)
# TP/FP/FN: 2 1 0
```

## Next steps

- [Concepts](../concepts/overview.md) -- the mental model behind what you just ran.
- [Detection 3D](../evaluation/detection-3d.md) -- the full task page: inputs, filters, matchers, metrics.
- [Evaluate a T4 dataset](../recipes/evaluate-t4-dataset.md) -- the same thing with real data.
