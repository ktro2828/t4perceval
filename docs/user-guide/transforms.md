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
#  ('base_link', 'LIDAR_TOP'), ('map', 'base_link')]

graph = FrameGraph.of(recording)
graph.frames()  # every frame named by an edge
graph.edge("base_link", "LIDAR_TOP")  # the edge, or None
graph.path(target_frame="map", source_frame="LIDAR_TOP")  # the hops
```

`transform_edges` and `FrameGraph.of` accept a `Store` or a `Recording`.

## Look up a pose

```python
from t4perceval import FRAME
from t4perceval.transform import TransformResolver

resolver = TransformResolver.of(recording, timeline=FRAME)
pose = resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=1)

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

Each takes and returns a `Transform3D`.

## The cross-frame guard

Nothing rewrites an entity's rows into another frame yet, so a matcher or a geometric metric refuses
to compare two _different, stated_ frames:

```text
ValueError: Cannot compare geometry across coordinate frames: /estimation/objects in 'base_link',
/ground_truth/objects in 'map'. Bring the inputs into one frame first.
```

Three ways out, in order of preference:

1. **Import both sides in the same frame.** The T4 importer takes `coords=` in its
   [`ImportOptions`](dataset-importers.md#t4-dataset); `"base_link"` is the default.
2. **Rewrite one side yourself** -- read the column, apply the resolved pose, log it to a new entity
   with the other frame's `frame_id`. A system that does this is on the
   [roadmap](../development/roadmap.md), not in the library yet.
3. **Opt out**, when you know the two frames coincide: `check_frames=False` on the matcher, and on
   `MatchJoin.of` for a metric.

Only two _different, stated_ frames raise; an unstated frame is not a disagreement.

## Where to go next

- [Coordinate system](../concepts/coordinate-system.md) -- why it is shaped this way.
- [Transform3D](../archetypes/transform-3d.md) -- the archetype's schema.
