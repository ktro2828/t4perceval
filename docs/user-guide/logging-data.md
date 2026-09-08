# Logging data

Getting your own data into a `Store`.

## Log one frame

```python
from t4perceval import Detections3D, LabelRegistry, Store, TimePoint

labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian"])
store = Store()

store.log(
    "/estimation/objects",
    Detections3D(
        position=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
        size=[[1.9, 4.5, 1.6]] * 2,
        class_id=labels.encode(["car", "car"]),
        confidence=[0.9, 0.4],
    ),
    at=TimePoint.at(frame=0, timestamp_ns=1_624_164_470_849_887_000),
    frame_id="base_link",
)
```

| Argument    | Notes                                                                         |
| :---------- | :---------------------------------------------------------------------------- |
| entity path | a string or an `EntityPath`                                                   |
| archetype   | any [archetype](../archetypes/index.md); its columns must all be equal length |
| `at`        | a `TimePoint`; give both axes if you may need to align later                  |
| `frame_id`  | the [coordinate frame](../concepts/coordinate-system.md) the rows are in      |

Log both `frame=` and `timestamp_ns=` whenever you have them. A chunk indexed on both axes answers
a frame query _and_ a timestamp query, and only the timestamp axis is comparable across two
independently produced recordings.

## Column conventions

Every column is `N` rows with a fixed per-row shape and dtype. Lists are converted for you; the
shape and dtype are checked, not guessed.

| Property      | Shape    | dtype | Notes                                 |
| :------------ | :------- | :---- | :------------------------------------ |
| `position`    | `(N, 3)` | `f64` | metres, in `frame_id`                 |
| `quaternion`  | `(N, 4)` | `f64` | **`xyzw` order** (SciPy's convention) |
| `size`        | `(N, 3)` | `f64` | **`(width, length, height)`**         |
| `roi`         | `(N, 4)` | `i32` | **`(x_min, y_min, height, width)`**   |
| `class_id`    | `(N,)`   | `i32` | meaning comes from a `LabelRegistry`  |
| `confidence`  | `(N,)`   | `f64` | constrained to `[0, 1]`               |
| `instance_id` | `(N,)`   | `i64` | interned by an `InstanceRegistry`     |

The two easy mistakes are the quaternion order and the size order. See
[Components](../components/index.md) for the complete table.

## Empty frames

A frame with zero objects is an ordinary frame, and you should log it:

```python
import numpy as np

store.log(
    "/estimation/objects",
    Detections3D(
        position=np.empty((0, 3)),
        quaternion=np.empty((0, 4)),
        size=np.empty((0, 3)),
        class_id=np.empty(0, dtype=np.int32),
        confidence=np.empty(0),
    ),
    at=TimePoint.at(frame=7),
    frame_id="base_link",
)
```

Skipping it is not the same thing: a missing frame drops out of every range query, while an empty
one contributes its false negatives -- and keeps the entity's `frame_id` stated, which the
[cross-frame guard](../concepts/coordinate-system.md#the-cross-frame-guard) relies on.

## Class ids

```python
labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian"])
labels.encode(["car", "pedestrian"])  # array([0, 2], dtype=int32)
```

Ids are assigned in first-seen order. **Build the registry once and hand the same one to every
source in a run** -- two sources that each derive their own are both valid and silently
incompatible, and the disagreement shows up as plausible numbers rather than an error.

Other useful entry points:

```python
labels = LabelRegistry.from_names(["car", "truck", "bus"])
coarse = labels.merged({"vehicle": ["car", "truck", "bus"]})  # collapse classes
labels.class_id_or("van", default=-1)  # tolerate unknown names
labels.encode(["van"], strict=False)  # unknown → UNKNOWN_CLASS_ID
labels.fingerprint()  # stable digest of the id assignment
```

## Instance ids

Tracking and prediction need a **persistent** identifier per object. An `InstanceRegistry` interns
dataset UUID strings into the integers of an `instance_id` column:

```python
from t4perceval import InstanceRegistry, Trackings3D

instances = InstanceRegistry()

store.log(
    "/ground_truth/objects",
    Trackings3D(
        position=[[0.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]],
        size=[[1.9, 4.5, 1.6]],
        class_id=labels.encode(["car"]),
        confidence=[1.0],
        instance_id=instances.encode(["7f3c...-a1"]),
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)
```

`intern` assigns the next free id to a new UUID; `instance_id` looks one up without interning, which
is what a filter uses so a typo raises instead of silently matching nothing.

## Static data

Anything that belongs to every point in time -- a sensor extrinsic, a shared time axis:

```python
from t4perceval import Transform3D

store.log_static(
    "/tf/lidar",
    Transform3D(
        translation=[1.2, 0.0, 1.8],
        rotation=[0.0, 0.0, 0.0, 1.0],
        child_frame_id="lidar",
    ),
    frame_id="base_link",
)
```

Loose columns, without an archetype:

```python
from t4perceval.descriptors import TIME_OFFSET
from t4perceval.component import BatchTimeOffset

store.log_static_components(
    "/estimation/objects",
    {TIME_OFFSET: BatchTimeOffset([[0, 500_000_000, 1_000_000_000]])},
)
```

Static columns take precedence over temporal ones carrying the same descriptor, so use them only for
things that genuinely do not vary.

## Logging from an importer instead

If your data is a T4 dataset or an MCAP bag, do not hand-roll the conversion:

```python
from t4perceval.importer.t4 import T4Importer

importer = T4Importer.open("/data/t4dataset")
labels = importer.label_registry()
ground_truth = importer.import_scene(labels=labels)  # -> Recording
```

See [Dataset importers](dataset-importers.md).

## Where to go next

- [Querying data](querying-data.md) -- reading it back.
- [Archetypes](../archetypes/index.md) -- what each bundle requires.
