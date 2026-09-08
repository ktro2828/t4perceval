# Dataset importers

An importer converts an external representation into a [`Recording`](persistence.md): an immutable
log plus the registries that give its integer columns meaning.

```text
T4 dataset ────▶ T4Importer      ──┐
                                   ├──▶ Recording ──▶ build_evaluation_store ──▶ Store ──▶ Pipeline
MCAP ROS bag ──▶ RosbagImporter  ──┘
```

Importers are **optional dependencies**. The evaluation core never imports them, and keeping them
out of the base install is what enforces that -- see [Installation](../getting-started/installation.md#extras).

!!! danger "One registry per run"

    Class ids are assigned in first-seen order, so two sources that each derive their own registry
    are both valid and silently incompatible. Build the registry once and hand **the same object**
    to every importer in the run.

## T4 dataset

Requires the `t4` extra.

```python
from t4perceval.importer.t4 import T4Importer

importer = T4Importer.open("/data/t4dataset")

labels = importer.label_registry()  # every category in the dataset
labels.names
# ('car', 'pedestrian', 'bicycle', 'traffic_cone')

recording = importer.import_scene(labels=labels)  # -> Recording
```

What lands in the recording:

```python
[str(p) for p in recording.entity_paths()]
# ['/tf/base_link', '/ground_truth/objects',
#  '/tf/CAM_BACK', '/tf/CAM_FRONT', '/tf/LIDAR_CONCAT']

[str(t) for t in recording.timelines()]
# ['frame', 'timestamp_ns']
```

Objects at `/ground_truth/objects`, plus the scene's transform tree: `map -> base_link` per keyframe
from `ego_pose`, and a static `base_link -> <channel>` per sensor from `calibrated_sensor`.

### Selecting what to import

```python
from t4perceval.importer.t4 import SceneSelection

importer.scene_tokens()  # every scene in the dataset

recording = importer.import_scene(
    labels=labels,
    selection=SceneSelection(
        scene=0,  # a token, an index, or None for the first
        samples=slice(0, 50),  # a slice or an explicit list
        channel_3d="LIDAR_CONCAT",  # which channel's keyframes define the frames
        channels_2d=("CAM_FRONT",),  # cameras to import 2D boxes from
    ),
)

for recording in importer.import_scenes(labels=labels):
    ...
```

`channel_3d` defaults to `"LIDAR_CONCAT"`, the concatenated point cloud a T4 release annotates
against. An individual sensor is one _input_ to that cloud, not the cloud itself, so a dataset that
publishes its lidar under another name has to say so:

```python
importer.import_scene(labels=labels, selection=SceneSelection(channel_3d="LIDAR_FRONT"))
```

An unknown channel raises rather than falling back to whatever else the dataset holds:

```text
KeyError: "Unknown channel 'LIDAR_FRONT'; this dataset has ['LIDAR_CONCAT', 'CAM_FRONT', 'CAM_BACK']"
```

### Import options

```python
from t4perceval.importer.t4 import ImportOptions, T4Importer

importer = T4Importer.open(
    "/data/t4dataset",
    options=ImportOptions(kind_3d="detections", coords="base_link"),
)
```

| Option                                 | Default         | Meaning                                                    |
| :------------------------------------- | :-------------- | :--------------------------------------------------------- |
| `kind_3d`                              | `"trackings"`   | `"detections"`, `"trackings"` or `"predictions"`           |
| `kind_2d`                              | `"trackings"`   | the same, for camera boxes                                 |
| `coords`                               | `"base_link"`   | the frame boxes are expressed in                           |
| `future_seconds`                       | `0.0`           | how much future trajectory to attach (for `"predictions"`) |
| `num_modes`, `num_timesteps`           | `None`          | pin the trajectory shape instead of inferring it           |
| `velocity`, `num_points`, `visibility` | `"auto"`        | emit the optional column when the dataset has it           |
| `unknown_labels`                       | `"error"`       | what to do with a category the registry does not know      |
| `entity_root`                          | `/ground_truth` | where objects are filed                                    |
| `instance_namespace`                   | `"gt"`          | prefix for interned instance UUIDs                         |
| `transforms`                           | `True`          | also import the frame graph                                |

The archetype you get follows `kind_3d`: `Detections3D`, `Trackings3D` or `Predictions3D`. They are
a nested superset chain over the same annotation rows, so asking for a richer kind costs one
extraction, not a second pass.

## MCAP ROS bag

Requires the `rosbag` extra. An MCAP bag is decoded from the message definitions it embeds, so **no
ROS installation** and no Autoware message packages are needed.

```python
from t4perceval.importer.rosbag import BagSelection, RosbagImporter

importer = RosbagImporter.open("/data/bags/run_0")  # a directory of *.mcap, or one file

importer.topics()  # the DetectedObjects / TrackedObjects / PredictedObjects topics
labels = importer.label_registry()  # the Autoware class enum, in enum order

estimation = importer.import_topic(
    labels=labels,
    selection=BagSelection(topic="/perception/object_recognition/tracking/objects"),
)
```

One topic at a time, into one `Recording`.

### What the conversion does

| Message            | Archetype       |
| :----------------- | :-------------- |
| `DetectedObjects`  | `Detections3D`  |
| `TrackedObjects`   | `Trackings3D`   |
| `PredictedObjects` | `Predictions3D` |

- `ObjectClassification` maps enum → canonical name → registry id.
- `Shape.dimensions` `(x=length, y=width, z=height)` becomes `BatchSize3D` `(width, length, height)`.
- A body-frame twist is rotated into the message frame before becoming `velocity`.
- `confidence` is the top classification probability by default;
  `ImportOptions(confidence="existence")` uses `existence_probability` instead.

### Import options

| Option                          | Default                       | Meaning                                     |
| :------------------------------ | :---------------------------- | :------------------------------------------ |
| `confidence`                    | `"classification"`            | or `"existence"`                            |
| `velocity`                      | `"auto"`                      | emit the column when the message carries it |
| `num_modes`, `num_timesteps`    | `None`                        | pin the trajectory shape                    |
| `unknown_labels`                | `"error"`                     | what to do with an unmapped class           |
| `entity_root`                   | `/estimation`                 | where objects are filed                     |
| `instance_namespace`            | `"est"`                       | prefix for interned instance ids            |
| `transforms`                    | `True`                        | also import `/tf` and `/tf_static`          |
| `tf_topics`, `tf_static_topics` | `("/tf",)`, `("/tf_static",)` | which topics carry transforms               |
| `tf_scope`                      | `"selection"`                 | which time span of transforms to keep       |

### Bag transforms are on TIMESTAMP

`/tf_static` becomes static edges and `/tf` temporal ones -- on the `TIMESTAMP` timeline only, since
a sample between two object messages has no frame index.

```python
from t4perceval import TIMESTAMP
from t4perceval.transform import TransformResolver

resolver = TransformResolver.of(estimation, timeline=TIMESTAMP)
```

## Provenance

Every recording carries where it came from:

```python
recording.metadata.sources
# (SourceInfo(kind='t4', uri='/data/t4dataset/1', version='1',
#             scene='cacaf846...', topic=None, entity_path='/ground_truth/objects',
#             extra=(('channel_3d', 'LIDAR_CONCAT'), ('coords', 'base_link'),
#                    ('kind_3d', 'trackings'), ('frames', '3'))),)
```

That survives into the evaluation's own metadata, so a saved result says which dataset, scene and
options produced it.

## Bringing two recordings together

```python
from t4perceval.evaluation import build_evaluation_store

setup = build_evaluation_store(ground_truth, estimation)
```

A `Recording` is read-only, and `Pipeline.run` writes results back into the store it reads from, so
the entities an evaluation needs are materialized into a fresh, writable store first. Only what you
name moves -- an importer that also recorded ego poses does not drag them in, which is what keeps a
saved result about the _evaluation_.

Two recordings imported separately usually do not number their frames alike. See
[Align frames by timestamp](../recipes/align-frames.md).

## Writing your own importer

See [Write a custom importer](../recipes/custom-importer.md). The rule the built-in ones follow is
that all knowledge of the external library lives in a single `source.py` module, so the rest of the
importer -- and all of the evaluation core -- stays free of it.

## Where to go next

- [Evaluate a T4 dataset](../recipes/evaluate-t4-dataset.md) · [Evaluate an MCAP ROS bag](../recipes/evaluate-rosbag.md)
- [Persistence](persistence.md) -- what a `Recording` is and how it is saved.
