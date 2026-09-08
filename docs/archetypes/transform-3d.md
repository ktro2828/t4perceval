# Transform3D

The pose of a child frame expressed in its parent frame. A point in the child maps into the parent
as `p_parent = R p_child + t`.

## Schema

| Component        | Requirement | Type         | Description                  |
| :--------------- | :---------- | :----------- | :--------------------------- |
| `translation`    | Required    | `Position3D` | the offset, one `(3,)` value |
| `rotation`       | Required    | `Quaternion` | one `(4,)` value, `xyzw`     |
| `child_frame_id` | Required    | `FrameId`    | the child frame's name       |

Descriptors: `TRANSLATION`, `ROTATION`, `CHILD_FRAME_ID` -- deliberately **not** `POSITION` and
`QUATERNION`.

The **parent** frame is not a component: it is the chunk's `frame_id`.

## The two frames

Split the way ROS splits a `TransformStamped`:

| ROS `TransformStamped`  | `t4perceval`                  |
| :---------------------- | :---------------------------- |
| `header.frame_id`       | `Chunk.frame_id` (the parent) |
| `child_frame_id`        | `child_frame_id`              |
| `transform.translation` | `translation`                 |
| `transform.rotation`    | `rotation`                    |

The parent belongs on the chunk because `frame_id` already means "the frame these rows are expressed
in", and that is exactly true of a transform's own row.

`child_frame_id` is required rather than optional: a transform whose parent is known but whose child
is not describes nothing, and the resolver would have to defend against it on every hop.

## Mono components

`Transform3D` is the **only** archetype whose components are mono. Every other archetype describes
`N` objects; this describes one relationship, of which an entity holds exactly one per point in
time.

```python
from t4perceval import Transform3D

pose = Transform3D(
    translation=[1.2, 0.0, 1.8],
    rotation=[0.0, 0.0, 0.0, 1.0],
    child_frame_id="lidar",
)
pose.translation.value  # array([1.2, 0. , 1.8])  -- not (1, 3)
pose.rotation.value  # array([0., 0., 0., 1.])
pose.child_frame_id.name  # 'lidar'
```

So "what if it has three rows?" is not a question the type can be asked. Underneath it is still a
one-row column -- the chunk, the Arrow schema and the store are unaffected -- and a range query over
three ego samples returns a three-row column.

## Static or temporal

The same archetype either way:

```python
# A fixed extrinsic
store.log_static("/tf/lidar", pose, frame_id="base_link")

# An ego pose, once per frame
store.log("/tf/base_link", ego, at=TimePoint.at(frame=1), frame_id="map")
```

`static` is a statement about _time_, not about the kind of data, and a static write keeps its
`frame_id`, so nothing about the edge is lost.

Static rows do not surface through `latest_at` or `range`, which is why the frame graph is built
from `static_chunks()` and `chunks()` rather than from a time query.

## Neither frame is in the entity path

A path says where data is filed; a frame names a node of the transform graph. Conflating them would
force a frame name to be path-safe and stop a graph being re-filed without being renamed. Edges are
discovered by reading chunks, so `tf_path("lidar")` is a convention, not a requirement, and a frame
name may contain a `/`.

## Where to go next

- [Coordinate system](../concepts/coordinate-system.md) -- the graph and the guard.
- [Working with transforms](../user-guide/transforms.md) -- the how-to.
- [Transform components](../components/transform.md) -- the mono column types.
