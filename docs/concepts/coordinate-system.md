# Coordinate system

Coordinate frames are **recorded data**, not state held outside the store. An ego pose and a sensor
extrinsic are both rows; the frame a set of objects is expressed in is a field on the chunk.

## frame_id: the frame rows are in

Every chunk states the frame its rows are expressed in.

```python
store.log("/ground_truth/objects", detections, at=..., frame_id="base_link")
```

`frame_id` is a plain string. It may contain a `/`, because it is not an entity path -- a path says
where data is filed, a frame names a node of the transform graph, and conflating them would force
frame names to be path-safe and stop a graph being re-filed without being renamed.

## Transform3D: one edge of the graph

A transform is one `parent -> child` relationship, split the way ROS splits a `TransformStamped`:

| ROS `TransformStamped`  | `t4perceval`                  |
| :---------------------- | :---------------------------- |
| `header.frame_id`       | `Chunk.frame_id` (the parent) |
| `child_frame_id`        | `Transform3D.child_frame_id`  |
| `transform.translation` | `Transform3D.translation`     |
| `transform.rotation`    | `Transform3D.rotation`        |

The parent belongs on the chunk because `frame_id` already means "the frame these rows are expressed
in", and that is exactly true of a transform's own row. A point in the child frame maps into the
parent as `p_parent = R p_child + t`.

`Transform3D` is the one archetype whose components are **mono** -- it describes a single
relationship, not `N` objects -- so its fields are values rather than columns:

```python
from t4perceval import Transform3D

pose = Transform3D(
    translation=[1.2, 0.0, 1.8],
    rotation=[0.0, 0.0, 0.0, 1.0],
    child_frame_id="lidar",
)
pose.translation.value  # array([1.2, 0. , 1.8])
pose.child_frame_id.name  # 'lidar'
```

Storage stays columnar: a mono value is widened into its `Batch*` counterpart on the way into a
chunk, so a range query over three ego samples still returns a three-row column.

## Static and temporal edges are the same archetype

A fixed extrinsic is logged with `log_static`, an ego pose with `log`. `static` says only that the
value does not depend on a timeline -- it is a statement about _time_, not about the kind of data --
and a static write keeps its `frame_id`, so nothing about the edge is lost.

```text
   map ──────(temporal: ego pose per frame)────▶ base_link
                                                     │
                                    ┌────────────────┼────────────────┐
                        (static)    │     (static)   │    (static)    │
                                    ▼                ▼                ▼
                                LIDAR_TOP        CAM_FRONT        CAM_BACK
```

## Discovery and composition

Edges are found by **reading the chunks**, not by parsing entity paths.

```python
from t4perceval import FRAME
from t4perceval.transform import TransformResolver, transform_edges

sorted(edge.frames for edge in transform_edges(recording))
# [('base_link', 'CAM_BACK'), ('base_link', 'CAM_FRONT'),
#  ('base_link', 'LIDAR_TOP'), ('map', 'base_link')]

resolver = TransformResolver.of(recording, timeline=FRAME)
pose = resolver.lookup(target_frame="map", source_frame="LIDAR_TOP", at=1)
pose.translation.value  # array([10.,  0.,  2.])
```

Static and temporal edges compose in one graph:

```text
T_map_lidar(t) = T_map_base_link(t) @ T_base_link_lidar
```

A temporal edge picks its sample with a `LookupPolicy` -- `LATEST`, `EXACT`, `NEAREST` or
`INTERPOLATE`. An unreachable frame **raises** rather than quietly resolving to identity.

!!! warning "Bag transforms live on TIMESTAMP only"

    `/tf_static` becomes static edges and `/tf` temporal ones -- on the `TIMESTAMP` timeline only,
    since a `/tf` sample between two object messages has no frame index. Look them up on that axis:
    `TransformResolver.of(estimation, timeline=TIMESTAMP)`.

## The cross-frame guard

Nothing rewrites an entity's rows into another frame yet. Rather than silently producing plausible
numbers, the system layer refuses to compare geometry across frames:

```text
ValueError: Cannot compare geometry across coordinate frames: /estimation/objects in 'base_link',
/ground_truth/objects in 'map'. Bring the inputs into one frame first.
```

Every matcher is covered through `MatchingSystem`, and every geometric metric through `MatchJoin`,
so one check covers all of them. Two rules keep it from firing spuriously:

- **Only two _different, stated_ frames raise.** An unstated frame (`None`) is not a disagreement:
  the entity may have had nothing in range, or may hold a metric rather than geometry.
- **An empty frame is still consulted.** An importer logs an empty frame with its `frame_id` intact;
  ignoring that would make a system's output frame flicker across a scene.

`check_frames=False` opts out per system.

## Where to go next

- [Working with transforms](../user-guide/transforms.md) -- the how-to.
- [Transform3D](../archetypes/transform-3d.md) -- the archetype's schema.
- [ADR 0003: coordinate frames as recorded data](../development/design-decisions/0003-coordinate-system.md).
