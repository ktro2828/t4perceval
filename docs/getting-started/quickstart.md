# Quick start

This page is the five-minute tour. It shows the vocabulary you need to read every other page, and
nothing more. If you would rather see a whole evaluation first, jump to
[First evaluation](first-evaluation.md).

## Log a frame of objects

One frame of detections is one **columnar batch**, not a list of objects. You hand it to a
[`Store`](../concepts/store.md) at an **entity path** and a **time**.

```python
from t4perceval import Detections3D, LabelRegistry, Store, TimePoint

labels = LabelRegistry.from_names(["car", "pedestrian"])
store = Store()

store.log(
    "/ground_truth/objects",
    Detections3D(
        position=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
        size=[[1.9, 4.5, 1.6]] * 2,
        class_id=labels.encode(["car", "car"]),
        confidence=[1.0, 1.0],
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)
```

Four things just happened, and each has a name:

| Name                                                                  | What it is here                                          |
| :-------------------------------------------------------------------- | :------------------------------------------------------- |
| [Entity path](../concepts/entity-component-archetype.md#entity-paths) | `"/ground_truth/objects"` -- _where_ the data is filed   |
| [Archetype](../concepts/entity-component-archetype.md#archetypes)     | `Detections3D` -- a validated bundle of components       |
| [Component](../concepts/entity-component-archetype.md#components)     | `position`, `quaternion`, `size`, ... -- one column each |
| [Timeline](../concepts/timeline.md)                                   | `TimePoint.at(frame=0)` -- _when_ it was observed        |

`frame_id` names the [coordinate frame](../concepts/coordinate-system.md) the rows are expressed in.

## Read it back

A query returns an `EntityView`, a lazy window onto the chunks that matched. `materialize` turns it
back into the archetype.

```python
from t4perceval import FRAME, TimeRange

view = store.range(
    "/ground_truth/objects",
    timeline=FRAME,
    time_range=TimeRange.everything(),
)
objects = view.materialize(Detections3D)
objects.position.values  # (N, 3) read-only float64
```

Every column is a NumPy array of `N` rows with a fixed per-row shape, and it is **read-only**.

## Run a system

A [system](../concepts/evaluation-pipeline.md) declares the components it needs and the components
it produces, and writes its result back into the store as new entities.

```python
from t4perceval.system import CenterDistanceMatchingSystem, Pipeline, SystemContext

matcher = CenterDistanceMatchingSystem.between(
    "/estimation/objects",
    "/ground_truth/objects",
    threshold=1.0,
)
Pipeline([matcher]).run(
    SystemContext(store, FRAME, labels=labels),
    TimeRange.everything(),
)

matcher.target  # /matching/center_distance -- a new entity in the same store
```

## The whole model, in one picture

```text
     Entity path              /ground_truth/objects
          │
          ▼
     Component columns        position  quaternion  size  class_id  confidence
          │
          ▼
     Archetype                Detections3D  (which columns must be present)
          │
          ▼
     Chunk                    one write: columns + time indexes + frame_id
          │
          ▼
     Store                    every chunk, indexed along timelines
          │
          ▼
     System                   reads entities, writes entities
```

## What makes this different

There is no `EvaluationTask` enum and no single config dict. **A task is the set of components
present plus the pipeline you compose.** A system that declares `REQUIRES = (POSITION,)` runs
against detections, trackings and predictions alike, because all three carry a `position` column
under the same [descriptor](../concepts/data-model.md#descriptors).

## Next steps

- [First evaluation](first-evaluation.md) -- the smallest complete workflow.
- [Concepts](../concepts/overview.md) -- why the pieces are shaped this way.
- [Evaluation tasks](../evaluation/detection-3d.md) -- what each task needs.
