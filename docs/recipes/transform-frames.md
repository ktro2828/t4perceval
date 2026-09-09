# Transform between coordinate frames

**Goal:** evaluate an estimation recorded in one coordinate frame against ground truth recorded in
another, without importing either side again.

Not to be confused with [Align frames by timestamp](align-frames.md), which pairs frame _indices_
on the time axis. This recipe is about `frame_id` -- `base_link` versus `map`.

## The problem

A T4 scene imports its boxes in `base_link` by default; an Autoware topic may publish in `map`. Both
are valid, and a distance between a `base_link` position and a `map` position is a number, not an
error -- so the matcher refuses rather than guess:

```text
ValueError: Cannot compare geometry across coordinate frames: /estimation/objects in 'base_link',
/ground_truth/objects in 'map'. Bring the inputs into one frame first.
```

Both recordings know their frame tree -- the T4 importer logs the ego pose per frame at
`/tf/base_link` and each sensor calibration at `/tf/<channel>`; a bag importer logs `/tf` and
`/tf_static` -- so the transform is recorded data, waiting to be applied.

## The one-liner

Put a `TransformEntitySystem` in front of the matcher and point the matcher at its target:

```python
from t4perceval import TimeRange
from t4perceval.evaluation import SourceSpec, build_evaluation_store_from
from t4perceval.system import (
    AveragePrecisionSystem,
    CenterDistanceMatchingSystem,
    Pipeline,
    TransformEntitySystem,
)
from t4perceval.transform import TransformResolver

setup = build_evaluation_store_from(
    [
        SourceSpec.of(ground_truth, "/ground_truth/objects"),  # in map
        SourceSpec.of(estimation, "/estimation/objects"),  # in base_link
    ],
    require_same_frame_id=False,  # the disagreement is the point; the next stage resolves it
)

moved = TransformEntitySystem.of(
    "/estimation/objects",
    target_frame="map",
    resolver=TransformResolver.of(estimation),  # the frame tree lives in the recording
)
matcher = CenterDistanceMatchingSystem.between(moved.target, "/ground_truth/objects", threshold=1.0)
metric = AveragePrecisionSystem.on(matcher.target, moved.target, "/ground_truth/objects")

Pipeline([moved, matcher, metric]).run(setup.context(), TimeRange.everything())
```

`moved.target` is `/estimation/objects/in/map`: a new entity, every chunk stamped `frame_id="map"`,
row for row the source in the other frame. The source is untouched.

## Why the resolver comes from the recording

An evaluation store holds only the entities you named in a `SourceSpec`, so the `/tf/*` edges are
not in it unless you name them too. Two ways to give the system a frame graph:

1. **Pass a resolver** built from the recording that has the edges -- as above. This is also where a
   lookup policy is chosen (`TransformResolver.of(..., policy=LookupPolicy.INTERPOLATE)`).
2. **Bring the edges along** and let the system build its own resolver from `ctx.store`:

   ```python
   (SourceSpec.of(estimation, "/tf/base_link"),)
   ```

   The edge's chunks state `frame_id="map"`, so this too needs `require_same_frame_id=False`.

## Which timeline

Each row's pose is looked up at that row's own time, on the **resolver's** timeline. A T4 scene logs
its ego pose on both `FRAME` and `TIMESTAMP`, so the default `FRAME` resolver works. A bag's `/tf`
lives on `TIMESTAMP` only -- there is no frame index between two object messages -- so build the
resolver there; the pipeline itself can still run on `FRAME`, because the system reads the
timestamp of each row from the chunk:

```python
TransformEntitySystem.of(
    "/estimation/objects",
    target_frame="base_link",
    resolver=TransformResolver.of(estimation, timeline=TIMESTAMP),
)
```

An entity that carries no index on the resolver's timeline is reported, naming the timeline.

## What moves and how

The rule is keyed on the component, not on the archetype, so a `Detections3D`, a `Predictions3D`
and a `SemanticSegmentation3D` all transform without any of them saying anything:

| Column                                    | Moves as                       |
| :---------------------------------------- | :----------------------------- |
| `position`, `point`, `translation`        | rotate + translate             |
| `waypoints`, every waypoint of every mode | rotate + translate             |
| `velocity`                                | rotate only                    |
| `quaternion`, `rotation`                  | composed with the frame's pose |
| `size`, ids, scores, everything else      | unchanged                      |
| `mask`                                    | **dropped**                    |

- **Velocity is rotated but never translated.** That is the velocity a frame-fixed observer in the
  target frame would measure _if the two frames did not move relative to each other_. The relative
  motion between `map` and a moving `base_link` is ignored -- a `map`-frame velocity that includes
  ego motion is a different quantity, and this system does not compute it.
- **Every waypoint of a chunk uses that chunk's own transform.** A predicted trajectory is expressed
  in the frame of the moment it was predicted, and moves as one rigid thing.
- **`mask` is dropped.** A distance or region mask is a claim about the source frame -- "within 50 m
  of the ego" -- and would be silently wrong in `map`. Run the filter on the transformed entity
  instead; the result is a passthrough, so a filter, an `ApplyMaskSystem`, the matcher and the metric
  all fit in the same pipeline:

  ```python
  near = FilterByDistanceSystem.on(moved.target, max_distance=50.0)
  kept = ApplyMaskSystem.of(moved.target, near.target)
  Pipeline([moved, near, kept, matcher_on(kept.target), metric_on(kept.target)]).run(...)
  ```

- **Static columns are not carried.** A static column of the source -- a trajectory time axis, a
  fixed extrinsic -- broadcasts over rows and is not frame-dependent in practice, and a static frame
  claim would be wrong in the new frame. Re-log it on the target if a consumer needs it.
- **Zero-row frames keep their place.** A frame with no objects still becomes an (empty) partition
  stamped with the target frame, so `latest_at` does not carry an earlier frame forward.

## Checking it worked

The transformed entity answers every query the source did:

```python
before = setup.store.range("/estimation/objects", timeline=FRAME, time_range=scene)
after = setup.store.range(moved.target, timeline=FRAME, time_range=scene)

after.frame_id  # 'map'
len(after) == len(before)  # row for row
after.times(FRAME).tolist() == before.times(FRAME).tolist()
```

The strongest check is the one the library's own tests make: transform an estimation authored in
`base_link` into `map`, and the match verdicts and AP come out **identical** to the same estimation
authored directly in `map`.

## When it refuses

| Message                                                                      | Meaning                                                                                            |
| :--------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| `... states no coordinate frame`                                             | The source was logged without `frame_id`; there is nothing to transform _from_.                    |
| `... cannot bring ... into 'map' at frame=3: Unknown coordinate frame 'map'` | The resolver's frame graph has no such frame -- the `/tf` edges are not where the resolver looked. |
| `... No recorded transform connects ...`                                     | Both frames exist but no chain of edges joins them.                                                |
| `... has no sample at frame=3`                                               | `LookupPolicy.EXACT`, and no ego pose at that time; use `LATEST`, `NEAREST` or `INTERPOLATE`.      |
| `... has no 'timestamp_ns' index`                                            | The resolver walks a timeline the entity is not logged on.                                         |

A source already in the target frame needs no frame graph at all: the rows are re-filed under the
new path unchanged.

## Where to go next

- [Transforms](../user-guide/transforms.md) -- the frame graph, the resolver and the lookup policies.
- [Coordinate system](../concepts/coordinate-system.md) -- what `frame_id` means and how edges are
  recorded.
- [Align frames by timestamp](align-frames.md) -- the _other_ kind of frame mismatch.
