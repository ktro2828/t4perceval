# FP validation

Checking that a perception stack no longer fires where it once hallucinated an object.

## Overview

This is the `fp_validation` task of `perception_eval`. Its ground truth is a set of **annotated false
positives**: places where an earlier stack reported an object that was not there, labelled with the
class `false_positive`. The question is the inverse of detection:

- an annotated box that **no estimation matches** is a **pass**;
- an annotated box that **an estimation matches** is a **fail**.

Nothing new is needed to ask it: a matcher that exempts the annotation class from the class rule, and
one metric.

```text
/ground_truth/objects ─┐
                       ├──▶ filters ──▶ matcher (agnostic_classes) ──▶ FpValidationSystem ──▶ pass rate
/estimation/objects  ──┘                       └──▶ fp_validation_by_instance ──▶ per annotation
```

## Inputs

### Ground truth

The annotations are ordinary boxes whose class is `false_positive`. The registry has to know that
class, and **the same registry** encodes both sides. A T4 dataset that has it as a category registers
it like any other class:

```python
from t4perceval.importer.t4 import ImportOptions as T4ImportOptions
from t4perceval.importer.t4 import T4Importer

t4 = T4Importer.open("/data/fp_validation", options=T4ImportOptions(entity_root="/ground_truth"))
labels = t4.label_registry()
labels.names
# ('car', 'bicycle', 'pedestrian', 'motorbike', 'unknown', 'false_positive')
```

### Estimation

The stack's output, of any class. An estimation imported from a bag can emit Autoware names that the
dataset lacks, such as `trailer` or `hazard`. Pass `unknown_labels="unknown"` or `"drop"` on its
`ImportOptions` instead of building a second registry. See
[Dataset importers](../user-guide/dataset-importers.md).

## Required components

```text
Detections3D / Trackings3D
├── position      required   what the matcher compares
├── quaternion    required
├── size          required
├── class_id      required   on both sides; the ground truth's names the annotations
└── instance_id   optional   on the ground truth, for the per-annotation breakdown
```

`FpValidationSystem` itself needs only `class_id` on each side; the matcher needs the geometry.
[`fp_validation_by_instance`](#which-annotation-keeps-failing) additionally needs `instance_id` on the
ground truth.

## Filtering

Everything in [Filtering](../user-guide/filtering.md) applies. The `CriticalObjectFilterConfig` of
`perception_eval`, which bounds the region both sides are judged in, is a region filter applied to
each entity before the matcher reads it:

```python
from t4perceval.system import ApplyMaskSystem, FilterByRegionSystem

EST = "/estimation/objects"
GT = "/ground_truth/objects"

region = [
    FilterByRegionSystem.symmetric(EST, max_xy=(100.0, 100.0), name="critical"),
    FilterByRegionSystem.symmetric(GT, max_xy=(100.0, 100.0), name="critical"),
    ApplyMaskSystem.of(EST, f"{EST}/filter/critical"),
    ApplyMaskSystem.of(GT, f"{GT}/filter/critical"),
]
EST, GT = f"{EST}/kept", f"{GT}/kept"  # the matcher and the metric read the kept rows
```

Build the matcher on the kept entities: its match rows index into the rows it read, and the metric must
read the same entities. The rows outside the region stay in the store.

## Matching

Any estimation can claim an annotation, whatever class it reports, so the matcher must not require the
classes to agree on annotations. **`agnostic_classes` exempts exactly that class.** Every other ground
truth keeps the class rule, as in `perception_eval`, so a real pedestrian cannot take a car estimation
away from the annotation next to it.

`perception_eval` gates candidates by centre distance and judges by plane distance;
`max_matchable_distance` and the plane-distance mode reproduce that:

```python
from t4perceval.system import PlaneDistanceMatchingSystem

matcher = PlaneDistanceMatchingSystem.between(
    EST,
    GT,
    threshold=2.0,  # perception_eval's matching_threshold_list
    agnostic_classes=["false_positive"],
    max_matchable_distance=5.0,  # its max_matchable_radii
)
```

Any other matcher takes `agnostic_classes` the same way.

!!! warning "Not `class_agnostic=True`"

    `class_agnostic=True` lifts the class rule for **every** ground truth, so ordinary objects of the
    wrong class compete for the estimations, and the detection metrics that share the matching go
    wrong.

The matching records a pair across classes as `TP`, which is what makes an annotation's verdict
readable: on this ground truth a **`TP` row is the failure and an `FN` row is the pass**. An
estimation that matched nothing is recorded as `FP` as usual, but this task does not count it.

## Metrics

```python
from t4perceval.system import FpValidationSystem

metric = FpValidationSystem.on(matcher.target, EST, GT)  # fp_label="false_positive"
[str(t) for t in metric.targets]
# ['/metrics/fp_validation']
```

| Parameter  | Default            | Meaning                              |
| :--------- | :----------------- | :----------------------------------- |
| `fp_label` | `"false_positive"` | the class name the annotations carry |

The value is the **passes divided by the verdicts**, where every annotated box at every frame is one
verdict:

| Row                        | Value                                                            |
| :------------------------- | :--------------------------------------------------------------- |
| the `false_positive` class | the pass rate; `NaN` when nothing was annotated                  |
| every other class          | `NaN`, with its own ground-truth count as `support`              |
| all classes (`-1`)         | the same pass rate -- only the annotation class carries verdicts |

Other classes are left to the detection metrics. A registry without `false_positive` yields `NaN`
rather than a reassuring 1.

### Detection metrics on a mixed dataset

A dataset that mixes real objects and annotations can be scored for detection in the same pipeline,
but not on the same ground truth. On a ground truth that still contains the annotations,
`AveragePrecisionSystem` reports an AP of zero for `false_positive` -- no estimation reports that
class -- the mAP averages the zero in, and its all-class row counts the annotations as objects to
find. Give the detection metrics a ground truth without the annotations:

```python
from t4perceval.system import FilterByLabelSystem, average_precision_sweep

REAL = f"{GT}/real"
detection = [
    FilterByLabelSystem.on(GT, name="real", exclude=["false_positive"]),
    ApplyMaskSystem.of(GT, f"{GT}/filter/real", name="real"),
    *average_precision_sweep(EST, REAL, thresholds=[0.5, 1.0, 2.0, 4.0]),
]
```

On that ground truth the `false_positive` row has no support and reports `NaN`, which the mAP skips.
An estimation that claimed an annotation matches nothing real, so the detection metrics count it as a
false positive, which is what it is.

### Which annotation keeps failing?

`perception_eval` reports the verdicts per annotated uuid. The same breakdown is a query over what the
pipeline recorded -- the matching rows and the ground truth's instance ids are both in the store:

```python
from t4perceval.system import fp_validation_by_instance

verdicts = fp_validation_by_instance(store, matcher.target, EST, GT, timeline=FRAME, labels=labels)
for instance_id, verdict in sorted(verdicts.items(), key=lambda item: item[1].pass_rate):
    print(instances.uuid(instance_id), verdict.frames, verdict.passes, verdict.fails)
```

`frames` counts the frames in which the instance was annotated, so it need not equal the length of
the scene. Pass `time_range=` to narrow the query to a stretch of the scene.

## Complete example

Two annotated false positives over three frames. The stack still fires on the first one in frames 0
and 1, and never on the second:

```python
from t4perceval import (
    FRAME,
    InstanceRegistry,
    LabelRegistry,
    MetricValues,
    Store,
    TimePoint,
    TimeRange,
    Trackings3D,
)
from t4perceval.system import (
    FpValidationSystem,
    Pipeline,
    PlaneDistanceMatchingSystem,
    SystemContext,
    fp_validation_by_instance,
)

EST = "/estimation/objects"
GT = "/ground_truth/objects"
labels = LabelRegistry.from_names(["car", "pedestrian", "false_positive"])
instances = InstanceRegistry()
store = Store()


def boxes(xs, names, ids):
    return Trackings3D(
        position=[[x, 0.0, 0.0] for x in xs],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * len(xs),
        size=[[2.0, 4.0, 1.5]] * len(xs),
        class_id=labels.encode(names),
        confidence=[0.9] * len(xs),
        instance_id=ids,
    )


for frame in range(3):
    at = TimePoint.at(frame=frame)
    annotations = instances.encode(["ghost-at-10m", "ghost-at-30m"])
    store.log(
        GT, boxes([10.0, 30.0], ["false_positive"] * 2, annotations), at=at, frame_id="base_link"
    )
    fired = frame < 2  # the stack fires on the first ghost in frames 0 and 1
    store.log(
        EST,
        boxes([10.3] if fired else [], ["car"] if fired else [], [1] if fired else []),
        at=at,
        frame_id="base_link",
    )

matcher = PlaneDistanceMatchingSystem.between(
    EST, GT, threshold=2.0, agnostic_classes=["false_positive"], max_matchable_distance=5.0
)
metric = FpValidationSystem.on(matcher.target, EST, GT)
Pipeline([matcher, metric]).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

values = store.range(metric.target, timeline=FRAME, time_range=TimeRange.everything()).materialize(
    MetricValues
)
print("pass rate", round(values.aggregate, 4))

verdicts = fp_validation_by_instance(store, matcher.target, EST, GT, timeline=FRAME, labels=labels)
for instance_id, verdict in sorted(verdicts.items()):
    print(instances.uuid(instance_id), verdict.frames, verdict.passes, verdict.fails)
```

```text
pass rate 0.6667
ghost-at-10m 3 1 2
ghost-at-30m 3 3 0
```

Six verdicts, four passes. The first ghost fails in the two frames the stack still fired on it; the
estimation is a `car` and the annotation a `false_positive`, and `agnostic_classes` is what let them
pair.

## Known divergences

Thresholds follow the **ground-truth** class, so the task runs at one threshold: the one of the
`false_positive` class. `perception_eval` keys it by the label of the estimation instead. See
[Metric divergences](metric-divergences.md#fp-validation-threshold).

## Where to go next

- [Detection 3D](detection-3d.md) -- the metrics the real objects of a mixed dataset take.
- [Matching](../user-guide/matching.md) -- `agnostic_classes`, `max_matchable_distance` and the rest.
- [Evaluate a T4 dataset](../recipes/evaluate-t4-dataset.md) ·
  [an MCAP ROS bag](../recipes/evaluate-rosbag.md) -- building the store this page evaluates.
- [Analyse a finished evaluation](../recipes/offline-analysis.md) -- more questions the stored rows
  answer.
