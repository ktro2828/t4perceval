# Evaluate a T4 dataset

**Goal:** ground truth from a T4 dataset, estimations from your model, and mAP out.

Requires the `t4` extra: `pip install 't4perceval[t4]'`.

## The shape of the answer

```text
T4 dataset ──▶ T4Importer ──▶ Recording ─┐
                                          ├─▶ build_evaluation_store ─▶ Store ─▶ Pipeline ─▶ Recording
your model ───────────────▶ Recording ───┘
```

A `Recording` is **read-only** and `Pipeline.run` writes results back into the store it reads from,
so the entities an evaluation needs are materialized into a fresh, writable store first. Only what
you name moves, which is what keeps a saved result about the _evaluation_ rather than about
everything that happened to be loaded.

## 1. Open the dataset and take its registry

```python
from t4perceval.importer.t4 import T4Importer

importer = T4Importer.open("/data/t4dataset")

labels = importer.label_registry()
labels.names
# ('car', 'pedestrian', 'bicycle', 'traffic_cone')
```

!!! danger "This registry is an input to everything else"

    Class ids are assigned in first-seen order. Hand **this object** to whatever produces your
    estimations. Two sources that each derive their own registry are both valid and silently
    incompatible, and the disagreement shows up as plausible numbers, not an error.

## 2. Import the ground truth

```python
ground_truth = importer.import_scene(labels=labels)  # -> Recording

[str(p) for p in ground_truth.entity_paths()]
# ['/tf/base_link', '/ground_truth/objects', '/tf/CAM_BACK', '/tf/CAM_FRONT', '/tf/LIDAR_CONCAT']
```

Objects land at `/ground_truth/objects`, plus the scene's transform tree. To narrow:

```python
from t4perceval.importer.t4 import SceneSelection

ground_truth = importer.import_scene(
    labels=labels,
    selection=SceneSelection(scene=0, samples=slice(0, 100)),
)

for recording in importer.import_scenes(labels=labels):  # every scene
    ...
```

To choose the archetype -- detections, trackings or predictions:

```python
from t4perceval.importer.t4 import ImportOptions, T4Importer

importer = T4Importer.open(
    "/data/t4dataset",
    options=ImportOptions(kind_3d="predictions", future_seconds=3.0),
)
```

## 3. Get your estimations into a Recording

If your model output is already arrays, log it to a store and bind the registries:

```python
from t4perceval import Recording, RecordingMetadata, SourceInfo, Store, TimePoint

store = Store()
for frame, est in enumerate(estimation_frames):
    store.log(
        "/estimation/objects",
        to_detections(est, labels),
        at=TimePoint.at(frame=frame, timestamp_ns=est["stamp_ns"]),
        frame_id="base_link",
    )

estimation = Recording.of(
    store,
    labels=labels,
    instances=ground_truth.instances,
    metadata=RecordingMetadata(
        sources=(
            SourceInfo(kind="model", uri="my-detector-v3", entity_path="/estimation/objects"),
        ),
    ),
)
```

Use the **same** `frame` indices the ground truth used. If you cannot, see step 4.

If your estimations are in an MCAP bag instead, use
[the ROS bag recipe](evaluate-rosbag.md) to produce this `Recording`.

## 4. Build the evaluation store

```python
from t4perceval.evaluation import build_evaluation_store

setup = build_evaluation_store(ground_truth, estimation)
ctx = setup.context()
```

Defaults: the reference is read from `/ground_truth/objects` and the query from
`/estimation/objects`, and both keep their paths. Override with `reference_path`, `query_path`,
`reference_target` and `query_target`.

Three guards fire here, and each has an escape hatch:

| Situation                                   | Default       | Opt out                                                          |
| :------------------------------------------ | :------------ | :--------------------------------------------------------------- |
| The two registries disagree about class ids | raises        | `reconcile=True` remaps onto the reference's                     |
| The two sources state different `frame_id`s | raises        | `require_same_frame_id=False`                                    |
| The two `FRAME` axes do not correspond      | _not_ checked | `align=AlignOptions(...)` -- see [Align frames](align-frames.md) |

The third one is the dangerous one: unrelated frame indices score as all-FP and all-FN frames rather
than failing. If the two recordings were produced independently, pass `align=`.

## 5. Run the pipeline

```python
from t4perceval import TimeRange
from t4perceval.system import Pipeline, average_precision_sweep

systems = average_precision_sweep(
    "/estimation/objects",
    "/ground_truth/objects",
    thresholds=[0.5, 1.0, 2.0, 4.0],
    heading=True,
)
Pipeline(systems).run(ctx, TimeRange.everything())
```

## 6. Read it, and freeze it

```python
from t4perceval import FRAME, MetricValues, TimeRange

values = setup.store.range(
    "/metrics/map", timeline=FRAME, time_range=TimeRange.everything()
).materialize(MetricValues)
print("mAP", round(values.aggregate, 4))

result = setup.into_recording(pipeline=systems)
result.metadata.sources  # which dataset, scene and options produced this
result.metadata.pipeline  # the system class names, in order
result.metadata.tags
```

The result holds the inputs, the masks, the matching results and the metrics -- everything the
evaluation touched and nothing it did not.

Save it as a `.t4eval` directory and every query works unchanged after reopening -- log order,
static data, registries and provenance all come back:

```python
from t4perceval.io import read_recording, write_recording

write_recording(result, "result.t4eval")
result = read_recording("result.t4eval")
```

See [Persistence](../user-guide/persistence.md#saving-a-whole-recording).

## Coordinate frames

The T4 importer expresses boxes in `base_link` by default (`ImportOptions(coords=...)`), which puts
the ego at the origin -- the frame `FilterByDistanceSystem` measures from. Make your estimations
match, or use the [resolver](../user-guide/transforms.md) to bring them over before matching.

The scene's frame graph comes along for free:

```python
from t4perceval.transform import TransformResolver

resolver = TransformResolver.of(ground_truth, timeline=FRAME)
resolver.lookup(target_frame="map", source_frame="LIDAR_CONCAT", at=1)
```

## Where to go next

- [Align frames by timestamp](align-frames.md) -- when the two `FRAME` axes disagree.
- [Evaluate an MCAP ROS bag](evaluate-rosbag.md)
- [Dataset importers](../user-guide/dataset-importers.md) -- every option.
