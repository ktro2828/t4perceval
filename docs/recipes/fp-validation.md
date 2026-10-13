# Validate false positives

**Goal:** make sure that a perception stack no longer fires where it once hallucinated an object.

This is the `fp_validation` task of `perception_eval`. Its ground truth is a dataset of annotated
false positives: places where an earlier stack reported an object that was not there. The
question is the inverse of detection. An annotated box that no estimation matches is a pass. An
annotated box that an estimation matches is a fail. Nothing new is needed to ask it: a
class-agnostic matcher and one metric.

## 1. One registry that knows the annotation class

The annotations carry the class `false_positive`. A T4 dataset that has it as a category registers
it like any other class. Build the registry from the dataset and give the same one to the
importer of the estimation:

```python
from t4perceval.importer.t4 import ImportOptions as T4ImportOptions
from t4perceval.importer.t4 import T4Importer

t4 = T4Importer.open("/data/fp_validation", options=T4ImportOptions(entity_root="/ground_truth"))
labels = t4.label_registry()
labels.names
# ('car', 'bicycle', 'pedestrian', 'motorbike', 'unknown', 'false_positive')
```

An estimation imported from a bag can emit Autoware names that the dataset lacks, such as
`trailer` or `hazard`. Pass `unknown_labels="unknown"` or `"drop"` on its `ImportOptions` instead
of a second registry. See [Dataset importers](../user-guide/dataset-importers.md).

## 2. Match class-agnostically

Any estimation can claim an annotation, whatever class it reports, so the matcher must not require
the classes to agree. `perception_eval` gates candidates by centre distance and judges by plane
distance. `max_matchable_distance` and the plane-distance mode reproduce that:

```python
from t4perceval.system import PlaneDistanceMatchingSystem

EST = "/estimation/objects"
GT = "/ground_truth/objects"

region = []  # filled in step 3, if you bound the region
matcher = PlaneDistanceMatchingSystem.between(
    EST,
    GT,
    threshold=2.0,  # perception_eval's matching_threshold_list
    class_agnostic=True,
    max_matchable_distance=5.0,  # its max_matchable_radii
)
```

The matching records a pair across classes as `TP`. That is what makes the verdict of an
annotation readable: on this ground truth a `TP` row is the failure and an `FN` row is the pass.
An estimation that matched nothing is recorded as `FP` as usual, but this task does not count it.

Thresholds follow the ground-truth class, so the task runs at one threshold, the one of the
`false_positive` class. `perception_eval` keys it by the label of the estimation instead. See
[Metric divergences](../evaluation/metric-divergences.md#fp-validation-threshold).

## 3. Restrict to the critical region, if you must

The `CriticalObjectFilterConfig` of `perception_eval` bounds the region both sides are judged in.
The same bound is a region filter applied to each entity before the matcher reads it:

```python
from t4perceval.system import ApplyMaskSystem, FilterByRegionSystem

region = [
    FilterByRegionSystem.symmetric(EST, max_xy=(100.0, 100.0), name="critical"),
    FilterByRegionSystem.symmetric(GT, max_xy=(100.0, 100.0), name="critical"),
    ApplyMaskSystem.of(EST, f"{EST}/filter/critical"),
    ApplyMaskSystem.of(GT, f"{GT}/filter/critical"),
]
EST, GT = f"{EST}/kept", f"{GT}/kept"  # the matcher and the metric read the kept rows
matcher = PlaneDistanceMatchingSystem.between(
    EST, GT, threshold=2.0, class_agnostic=True, max_matchable_distance=5.0
)
```

The matcher of step 2 keeps the paths it was built with, so build it again on the kept entities.
Its match rows index into the kept rows, and the metric must read the same entities. The rows
outside the region stay in the store. The matcher does not read them.

## 4. Score

```python
from t4perceval import FRAME, MetricValues, TimeRange
from t4perceval.system import FpValidationSystem, Pipeline

metric = FpValidationSystem.on(matcher.target, EST, GT)  # fp_label="false_positive"
Pipeline([*region, matcher, metric]).run(
    setup.context(), TimeRange.everything()
)  # region is [] without step 3

values = setup.store.range(
    metric.target, timeline=FRAME, time_range=TimeRange.everything()
).materialize(MetricValues)
values.of_class(labels.class_id("false_positive"))  # pass rate over every annotated box-frame
values.aggregate  # the same number: only that class carries a value
values.support.values  # ground-truth count per class, as in every metric
```

The value is the passes divided by the verdicts. Every annotated box at every frame is one
verdict. If there is no annotated box, the value is `NaN`. Other classes report `NaN` with their
own support, because the detection metrics score them and this one does not. A dataset that mixes
real objects and annotations therefore evaluates both in one pipeline. `setup` is the evaluation
store of [Evaluate a T4 dataset](evaluate-t4-dataset.md) or [an MCAP ROS bag](evaluate-rosbag.md).

## 5. Which annotation keeps failing?

`perception_eval` reports the verdicts per annotated uuid as frame rates. The same breakdown is a
query over what the pipeline recorded. The matching rows and the instance ids of the ground truth
are both in the store:

```python
from t4perceval.system import fp_validation_by_instance

verdicts = fp_validation_by_instance(
    setup.store, matcher.target, EST, GT, timeline=FRAME, labels=labels
)
for instance_id, verdict in sorted(verdicts.items(), key=lambda item: item[1].pass_rate):
    print(setup.instances.uuid(instance_id), verdict.frames, verdict.passes, verdict.fails)
```

```text
3b1f...  12  3   9   <- fails in 9 of its 12 frames
a90c...  12  12  0
```

`frames` counts the frames in which the instance was annotated, so it need not equal the length
of the scene. Pass `time_range=` to narrow the query to a stretch of the scene.

## Where to go next

- [Matching](../user-guide/matching.md) for `class_agnostic`, `max_matchable_distance` and the rest.
- [Analyse a finished evaluation](offline-analysis.md) for more questions the stored rows answer.
- [Metric divergences](../evaluation/metric-divergences.md) for where this differs from `perception_eval`.
