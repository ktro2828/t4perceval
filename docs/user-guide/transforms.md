# Working with transforms

Coordinate frames are recorded data. This page is the how-to; the reasoning is in
[Coordinate system](../concepts/coordinate-system.md).

## Record an edge

Split the two frames the way ROS splits a `TransformStamped`: the **parent** is the chunk's
`frame_id`, the **child** is a column.

```python
from t4perceval import Store, TimePoint, Transform3D

store = Store()

# A fixed extrinsic: base_link -> lidar
store.log_static(
    "/tf/lidar",
    Transform3D(
        translation=[1.2, 0.0, 1.8],
        rotation=[0.0, 0.0, 0.0, 1.0],
        child_frame_id="lidar",
    ),
    frame_id="base_link",
)

# An ego pose: map -> base_link, once per frame
store.log(
    "/tf/base_link",
    Transform3D(
        translation=[10.0, 0.0, 0.0],
        rotation=[0.0, 0.0, 0.0, 1.0],
        child_frame_id="base_link",
    ),
    at=TimePoint.at(frame=1),
    frame_id="map",
)
```

Same archetype either way. `log_static` says only that the value does not depend on a timeline.

`tf_path("lidar")` gives the conventional entity for a child frame's edges, but the filing is up to
you -- edges are found by reading chunks, not by parsing paths.

## Discover the graph

```python
from t4perceval.transform import FrameGraph, transform_edges

sorted(edge.frames for edge in transform_edges(recording))
# [('base_link', 'CAM_BACK'), ('base_link', 'CAM_FRONT'),
#  ('base_link', 'LIDAR_CONCAT'), ('map', 'base_link')]

graph = FrameGraph.of(recording)
graph.frames()  # every frame named by an edge
graph.edge("base_link", "LIDAR_CONCAT")  # the edge, or None
graph.path(target_frame="map", source_frame="LIDAR_CONCAT")  # the hops
```

`transform_edges` and `FrameGraph.of` accept a `Store` or a `Recording`.

## Look up a pose

```python
from t4perceval import FRAME
from t4perceval.transform import TransformResolver

resolver = TransformResolver.of(recording, timeline=FRAME)
pose = resolver.lookup(target_frame="map", source_frame="LIDAR_CONCAT", at=1)

pose.translation.value  # array([10.,  0.,  2.])
pose.rotation.value  # xyzw
```

Static and temporal edges compose in one graph:

```text
T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar
```

An unreachable frame **raises** rather than resolving to identity.

### Choosing a sample

A temporal edge has to pick which of its samples answers a query:

| `LookupPolicy` | Behaviour                                              |
| :------------- | :----------------------------------------------------- |
| `LATEST`       | the most recent sample at or before `at` (the default) |
| `EXACT`        | the sample at `at`, or raise                           |
| `NEAREST`      | the closest sample either side                         |
| `INTERPOLATE`  | interpolate between the samples either side            |

```python
from t4perceval.transform import LookupPolicy, TransformResolver

resolver = TransformResolver.of(recording, timeline=FRAME, policy=LookupPolicy.INTERPOLATE)
```

### Which timeline?

Bag transforms are recorded on `TIMESTAMP` only: a `/tf` sample between two object messages has no
frame index.

```python
from t4perceval import TIMESTAMP

resolver = TransformResolver.of(estimation, timeline=TIMESTAMP)
resolver.lookup(target_frame="map", source_frame="base_link", at=1_624_164_470_849_887_000)
```

A T4 scene records its ego pose per keyframe, so `FRAME` works there.

## Compose poses yourself

```python
from t4perceval.transform import chain, compose, identity, interpolate, invert

identity()
invert(pose)
compose(outer, inner)  # inner first, then outer
chain([map_to_base, base_to_lidar])
interpolate(before, after, fraction=0.25)
```

Each takes and returns a _pose_: a `(translation, rotation)` pair of a `(3,)` and an `xyzw` `(4,)`
array. `pose_of(transform)` turns a resolver's `Transform3D` into one.

## Expressing an entity in another frame

`TransformEntitySystem` reads an entity, looks each frame's pose up, and writes the result as a
**new** entity in the target frame:

```python
from t4perceval.system import Pipeline, TransformEntitySystem

moved = TransformEntitySystem.of("/estimation/objects", target_frame="map")
Pipeline([moved]).run(ctx, scene)

moved.target  # /estimation/objects/in/map, every chunk with frame_id="map"
```

Each row is transformed with the pose at its own time, so an object seen from a moving ego lands at a
different `map` position in every frame. What moves is decided per component, not per archetype:

| Column                                     | Moves as              |
| :----------------------------------------- | :-------------------- |
| `position`, `point`, `translation`         | rotate + translate    |
| `waypoints` (every waypoint of every mode) | rotate + translate    |
| `velocity`                                 | rotate only           |
| `quaternion`, `rotation`                   | compose with the pose |
| `size`, ids, scores, everything else       | unchanged             |
| `mask`                                     | **dropped**           |

Velocity is rotated but never translated: that is the velocity a frame-fixed observer in the target
frame would measure _if the two frames did not move relative to each other_ -- the relative motion
between them is ignored. A `mask` is a claim about the source frame, so it does not survive; filter
the result instead. Static columns of the source are not carried either -- they are not
frame-dependent in practice, and a static frame claim would be wrong in the new frame.

The system builds its `TransformResolver` from `ctx.store` unless you pass one. An evaluation store
holds only the entities you named, so when the `/tf` edges are not in it, pass a resolver built from
the recording that has them -- and on the timeline they live on (a bag's `/tf` is `TIMESTAMP` only):

```python
TransformEntitySystem.of(
    "/estimation/objects",
    target_frame="base_link",
    resolver=TransformResolver.of(estimation, timeline=TIMESTAMP),
)
```

The result is a passthrough of the source minus `mask`, so a filter, matcher or metric can read it in
the same pipeline. The full walk-through is
[Transform between coordinate frames](../recipes/transform-frames.md).

## The cross-frame guard

A matcher or a geometric metric refuses to compare two _different, stated_ frames rather than
silently producing plausible numbers:

```text
ValueError: Cannot compare geometry across coordinate frames: /estimation/objects in 'base_link',
/ground_truth/objects in 'map'. Bring the inputs into one frame first.
```

Three ways out, in order of preference:

1. **Import both sides in the same frame.** The T4 importer takes `coords=` in its
   [`ImportOptions`](dataset-importers.md#t4-dataset); `"base_link"` is the default.
2. **Transform one side** with `TransformEntitySystem`, as above, and point the matcher at its
   target.
3. **Opt out**, when you know the two frames coincide: `check_frames=False` on the matcher, and on
   `MatchJoin.of` for a metric.

Only two _different, stated_ frames raise; an unstated frame is not a disagreement.

## Where to go next

- [Coordinate system](../concepts/coordinate-system.md) -- why it is shaped this way.
- [Transform3D](../archetypes/transform-3d.md) -- the archetype's schema.
