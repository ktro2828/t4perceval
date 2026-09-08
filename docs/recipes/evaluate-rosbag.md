# Evaluate an MCAP ROS bag

**Goal:** ground truth from a T4 dataset, estimations from a recorded Autoware run, and metrics out.

Requires the `rosbag` extra: `pip install 't4perceval[rosbag]'`. An MCAP bag is decoded from the
message definitions it embeds, so **no ROS installation** and no Autoware message packages are
needed.

## The shape of the answer

```text
T4 dataset ─▶ T4Importer     ─▶ Recording ─┐
                                            ├─▶ align ─▶ build_evaluation_store ─▶ Pipeline
MCAP bag   ─▶ RosbagImporter ─▶ Recording ─┘
```

The alignment step is not optional here. A T4 scene numbers its frames by sample and a bag by
message, so the two `FRAME` axes are unrelated.

## 1. Open the bag

```python
from t4perceval.importer.rosbag import RosbagImporter

importer = RosbagImporter.open("/data/bags/run_0")  # a directory of *.mcap, or one file

importer.topics()
# the DetectedObjects / TrackedObjects / PredictedObjects topics the bag holds
```

## 2. Share one label registry

```python
from t4perceval.importer.t4 import T4Importer

t4 = T4Importer.open("/data/t4dataset")
labels = t4.label_registry()  # the ground truth's registry is the authority
```

The bag importer can build its own from the Autoware class enum:

```python
labels = importer.label_registry()  # the Autoware enum, in enum order
```

Pick **one** of the two and give it to both importers. If you genuinely have to import with two
registries, pass `reconcile=True` to `build_evaluation_store` and it will remap the query's ids onto
the reference's -- but sharing one registry up front is better, because reconciliation can only map
names that exist in both.

## 3. Import a topic

```python
from t4perceval.importer.rosbag import BagSelection

estimation = importer.import_topic(
    labels=labels,
    selection=BagSelection(topic="/perception/object_recognition/tracking/objects"),
)
```

One topic at a time. The schema decides the archetype:

| Message            | Archetype       |
| :----------------- | :-------------- |
| `DetectedObjects`  | `Detections3D`  |
| `TrackedObjects`   | `Trackings3D`   |
| `PredictedObjects` | `Predictions3D` |

Objects land at `/estimation/objects`. `/tf` and `/tf_static` come along as transform edges unless
you pass `ImportOptions(transforms=False)`.

### What the conversion does

- `ObjectClassification` → canonical name → registry id.
- `Shape.dimensions` `(x=length, y=width, z=height)` → `BatchSize3D` `(width, length, height)`.
- A body-frame twist is rotated into the message frame before becoming `velocity`.
- `confidence` is the **top classification probability**; use
  `ImportOptions(confidence="existence")` for `existence_probability`.

## 4. Import the ground truth

```python
ground_truth = t4.import_scene(labels=labels)
```

## 5. Align the frames, then build the store

```python
from t4perceval.align import AlignOptions
from t4perceval.evaluation import build_evaluation_store

setup = build_evaluation_store(
    ground_truth,
    estimation,
    align=AlignOptions(tolerance_ns=75_000_000),  # 75 ms, the incumbent's default
)
setup.metadata.tags
# ('align.pairs', '87'), ('align.unmatched_reference', '3'), ...
```

Check those tags before trusting the numbers. A low pair count means the tolerance is too tight or
the clocks are offset -- `AlignOptions(offset_ns=...)` corrects a known skew. See
[Align frames by timestamp](align-frames.md).

## 6. Coordinate frames

This is the step that most often goes wrong between a dataset and a bag.

```python
ground_truth.static_frame_id("/ground_truth/objects")  # 'base_link' by default
```

Autoware topics are usually in `base_link` or `map` depending on the node. If the two disagree, the
matcher refuses:

```text
ValueError: Cannot compare geometry across coordinate frames: /estimation/objects in 'map',
/ground_truth/objects in 'base_link'. Bring the inputs into one frame first.
```

Options, best first:

1. Import the T4 side in the bag's frame -- `ImportOptions(coords="map")` on the T4 importer.
2. Rewrite one side yourself using the bag's own transform graph:

   ```python
   from t4perceval import TIMESTAMP
   from t4perceval.transform import TransformResolver

   resolver = TransformResolver.of(estimation, timeline=TIMESTAMP)
   pose = resolver.lookup(target_frame="base_link", source_frame="map", at=stamp_ns)
   ```

   Bag transforms live on `TIMESTAMP` only, because a `/tf` sample between two object messages has
   no frame index.

3. `check_frames=False` on the matcher, when you know they coincide.

## 7. Run and read

```python
from t4perceval import FRAME, MetricValues, TimeRange
from t4perceval.system import CenterDistanceMatchingSystem, ClearSystem, Pipeline

ctx = setup.context()
matcher = CenterDistanceMatchingSystem.between(
    "/estimation/objects", "/ground_truth/objects", threshold=1.0
)
clear = ClearSystem.on(matcher.target, "/estimation/objects", "/ground_truth/objects")

Pipeline([matcher, clear]).run(ctx, TimeRange.everything())

for target in clear.targets:
    values = setup.store.range(
        target, timeline=FRAME, time_range=TimeRange.everything()
    ).materialize(MetricValues)
    print(target.name, values.value.values)
```

A `TrackedObjects` topic carries instance ids, so [tracking metrics](../evaluation/tracking.md)
apply. For a `DetectedObjects` topic use [detection metrics](../evaluation/detection-3d.md) instead.

## Where to go next

- [Align frames by timestamp](align-frames.md) -- the alignment in detail.
- [Dataset importers](../user-guide/dataset-importers.md) -- every option.
