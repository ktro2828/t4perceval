# Evaluate with grouped labels

**Goal:** evaluate with some classes collapsed into one -- `car`, `truck` and `bus` as `vehicle` --
so that a `car` estimated as a `truck` counts as a hit.

There are two starting points:

- **[Grouped from the start](#grouped-from-the-start)** -- you know the grouping before importing.
  Build the grouped registry first and import with it. Nothing else changes.
- **[Re-evaluate an existing evaluation](#re-evaluate-an-existing-evaluation)** -- an evaluation
  has already run on the original classes, and you want a grouped one next to it without importing
  again.

## Why the registry alone is not enough

Class ids are part of the data. A matcher compares the stored `class_id` columns directly, and
[`LabelRegistry.merged()`](../reference/api/label.md) yields a registry with **new** ids rather than
a flag read at match time. Handing the grouped registry to `SystemContext` over a store encoded with
the original one therefore does not group anything -- and does not fail either:

```text
original ids: car=0  truck=1  pedestrian=2
grouped ids:  car=truck=vehicle=0          pedestrian=1
```

- A `car` ground truth and a `truck` estimation still carry ids `0` and `1`, so they never match.
- `Thresholds(by_class={"pedestrian": ...})` resolves `"pedestrian"` to `1` -- which, in the stored
  columns, is `truck`.
- Metric rows are labelled by the grouped registry, so the `truck` row is reported as `pedestrian`.

The numbers look plausible. Both cases below therefore make sure the ids are **encoded** with the
grouped registry before anything runs.

## Grouped from the start

### 1. Build the grouped registry once

Start from the registry the ground truth defines, and merge it:

```python
from t4perceval.importer.t4 import T4Importer

t4 = T4Importer.open("/data/t4dataset")
labels = t4.label_registry().merged({"vehicle": ["car", "truck", "bus"]})

t4.label_registry().names
# ('car', 'truck', 'bus', 'pedestrian', 'bicycle')
labels.names
# ('vehicle', 'pedestrian', 'bicycle')
labels.class_id("truck") == labels.class_id("vehicle")
# True
```

`merged()` keeps the grouped names as **aliases** of their group, so a source that still says
`truck` encodes straight to the `vehicle` id. Classes you did not group keep their own. Grouping a
name the registry does not have raises, so a typo cannot silently leave a class out.

### 2. Give it to every source

The same object, to every importer -- exactly as with an ungrouped registry -- together with one
shared `InstanceRegistry`:

```python
from t4perceval import InstanceRegistry
from t4perceval.importer.rosbag import BagSelection, RosbagImporter

instances = InstanceRegistry()

ground_truth = t4.import_scene(labels=labels, instances=instances)

estimation = RosbagImporter.open("/data/bags/run_0").import_topic(
    labels=labels,
    instances=instances,
    selection=BagSelection(topic="/perception/object_recognition/objects"),
)
```

The T4 importer produces trackings by default, and a `TrackedObjects` or `PredictedObjects` topic
carries instance ids too. Those ids are encoded at import time, so two sources that each got their
own `InstanceRegistry` cannot be combined: `build_evaluation_store` raises. Sharing one costs
nothing when the bag holds only detections.

For model output you log yourself, `labels.encode(names)` resolves the aliases the same way.

Every name a source emits must be a class or an alias of the registry. The Autoware enum has
classes a dataset may not -- `trailer`, `motorcycle`, `unknown` -- and the importer raises on them
by default. Either add them to a group, or choose what happens with
`ImportOptions(unknown_labels="drop")` or `"unknown"`; see
[Dataset importers](../user-guide/dataset-importers.md).

### 3. Evaluate as usual

```python
from t4perceval import FRAME, TimeRange
from t4perceval.align import AlignOptions
from t4perceval.evaluation import build_evaluation_store
from t4perceval.system import Pipeline, Thresholds, average_precision_sweep

EST = "/estimation/objects"
GT = "/ground_truth/objects"
scene = TimeRange.everything()

systems = average_precision_sweep(
    EST,
    GT,
    thresholds=[Thresholds(1.0, by_class={"vehicle": 1.5}), 2.0],
)
setup = build_evaluation_store(
    ground_truth,
    estimation,
    align=AlignOptions(tolerance_ns=75_000_000),  # the dataset and the bag number frames apart
)  # the registries agree: no reconcile
Pipeline(systems).run(setup.context(), scene)
result = setup.into_recording(pipeline=systems)
```

The ground truth and the bag were imported independently, so their `FRAME` indices do not
correspond; `align=` pairs them by timestamp first. Check `setup.metadata.tags` for the pair count
before trusting the numbers -- see [Align frames by timestamp](align-frames.md). Leave `align=` out
only when both sources already share frame indices, as model output logged against the ground
truth's frames does.

Key per-class settings -- `Thresholds`, `FilterByLabelSystem`, `max_matchable_distance` -- by the
**group** name. The resulting recording carries the grouped registry, aliases included, so the
grouping is visible to anyone who reopens it.

Do not import the two sources with different registries and rely on `reconcile=True` to fix it:
reconciliation maps onto the ground truth's registry, which cannot hold a group the other side
invented. Group the one registry both share.

## Re-evaluate an existing evaluation

This is the case where the grouping comes **after** a first evaluation on the original classes.
The inputs are already encoded with the original ids, so they are re-encoded into a new store and
the pipeline runs again; the first result is left untouched.

### 1. The first evaluation

Nothing special here; keep the result as a recording.

```python
from t4perceval import FRAME, TimeRange
from t4perceval.evaluation import build_evaluation_store
from t4perceval.system import Pipeline, average_precision_sweep

EST = "/estimation/objects"
GT = "/ground_truth/objects"
scene = TimeRange.everything()

systems = average_precision_sweep(EST, GT, thresholds=[1.0, 2.0])
setup = build_evaluation_store(ground_truth, estimation)  # plus align=, as above, if imported apart
Pipeline(systems).run(setup.context(), scene)
original = setup.into_recording(pipeline=systems)
```

### 2. Re-encode the inputs into the grouped registry

Copy only the **input** entities into a new store, remapping their class-id columns on the way:

```python
from attrs import evolve

from t4perceval import Store
from t4perceval.evaluation import EvaluationSetup
from t4perceval.reconcile import class_id_lut, remap_class_ids


def relabeled(recording, labels, inputs):
    """Return a setup holding ``inputs`` with their class ids expressed in ``labels``."""
    lut = class_id_lut(recording.labels, labels)
    store = Store()
    for path in inputs:
        # Static chunks first, then temporal ones in log order -- the same order
        # `build_evaluation_store` moves them in.
        for chunk in (*recording.static_chunks(path), *recording.chunks(path)):
            store.send_chunk(remap_class_ids(chunk, lut))
    return EvaluationSetup(
        store,
        labels,
        recording.instances,
        evolve(recording.metadata, labels_fingerprint=labels.fingerprint()),
    )


grouped_labels = original.labels.merged({"vehicle": ["car", "truck", "bus"]})
grouped_setup = relabeled(original, grouped_labels, inputs=(GT, EST))
```

Leave the outputs behind -- matches, filter masks and metrics all depend on class, so none of them
survives regrouping:

| Output          | Why it must be recomputed                                           |
| :-------------- | :------------------------------------------------------------------ |
| `/matching/...` | whether two rows may pair depends on their classes being equal      |
| filter masks    | a label filter selects by class                                     |
| `/metrics/...`  | rows are per class; remapping them would fold several rows into one |

`class_id_lut` raises if an original class has nowhere to go. `merged()` keeps every class it was
not asked to group, so that only happens when the two registries are unrelated.

If you still hold the imported recordings, relabel each of them instead -- it is the same remapping
with less to copy. `relabeled` returns an `EvaluationSetup`, and `build_evaluation_store` takes
recordings, so convert each with `into_recording()` first:

```python
grouped_setup = build_evaluation_store(
    relabeled(ground_truth, grouped_labels, inputs=(GT,)).into_recording(),
    relabeled(estimation, grouped_labels, inputs=(EST,)).into_recording(),
    align=AlignOptions(tolerance_ns=75_000_000),  # as in the first evaluation, if it aligned
)
```

### 3. Run the pipeline again

Build the systems afresh, and **key per-class settings by the group names**. The original names
still resolve, as aliases, but when `car` and `truck` had different values only one of them can
apply to `vehicle` -- say which.

```python
from t4perceval.system import Thresholds

grouped_systems = average_precision_sweep(
    EST,
    GT,
    thresholds=[Thresholds(1.0, by_class={"vehicle": 1.5}), 2.0],
)
Pipeline(grouped_systems).run(grouped_setup.context(), scene)
grouped = grouped_setup.into_recording(pipeline=grouped_systems)
```

The same goes for `FilterByLabelSystem` and `max_matchable_distance`.

### 4. Compare

Each recording carries its own registry, so read each one's class names from its own `labels`:

```python
from t4perceval import MetricValues

for name, result in (("original", original), ("grouped", grouped)):
    m_ap = result.range("/metrics/map", timeline=FRAME, time_range=scene).materialize(MetricValues)
    per_class = {n: round(m_ap.of_class(result.labels.class_id(n)), 4) for n in result.labels.names}
    print(f"{name:>8}: mAP={m_ap.aggregate:.4f}", per_class)
```

```text
original: mAP=0.5000 {'car': 0.0, 'truck': 0.0, 'bus': 1.0, 'pedestrian': 1.0}
 grouped: mAP=1.0000 {'vehicle': 1.0, 'pedestrian': 1.0}
```

Here a `car` estimated as `truck` is a miss under the original classes and a hit under `vehicle`.

Both are ordinary recordings; save them side by side with
[`write_recording`](../user-guide/persistence.md#saving-a-whole-recording).

## Grouping is not `class_agnostic`

`class_agnostic=True` lets **any** two classes pair. Grouping lets classes pair only within their
group -- a `car` may match a `truck`, but not a `pedestrian`.

## Where to go next

- [Evaluate a T4 dataset](evaluate-t4-dataset.md) and [an MCAP ROS bag](evaluate-rosbag.md) -- the
  full import walkthroughs the first case builds on.
- [Analyse a finished evaluation](offline-analysis.md) -- querying either result.
