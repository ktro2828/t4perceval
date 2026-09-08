# Transform components

The only **mono** components in the package. A mono component holds exactly one value, written and
read without a row index, because it describes one relationship rather than `N` objects.

```python
from t4perceval.component import FrameId, Position3D, Quaternion

Position3D([1.2, 0.0, 1.8]).value  # array([1.2, 0. , 1.8])  -- not (1, 3)
Quaternion([0.0, 0.0, 0.0, 1.0]).value
FrameId("lidar").name  # 'lidar'
```

Storage stays columnar: a mono value is widened into its `Batch*` counterpart on the way into a
chunk, so a range query over three ego samples still returns a three-row column. The chunk, the
Arrow schema and the store are unaffected.

```python
Position3D([1.2, 0.0, 1.8]).as_batch()  # BatchPosition3D with one row
```

## Position3D

One 3D translation, `f64`. Subclasses `BatchPosition3D`, so everything a column can do it can do --
`value` is the extra.

Descriptor: **`TRANSLATION`**, not `POSITION`.

## Quaternion

One unit quaternion in `xyzw` order, `f64`. Subclasses `BatchQuaternion`.

```python
Quaternion([0.0, 0.0, 0.0, 1.0]).as_rotation()  # one SciPy rotation, not a length-one stack
```

Descriptor: **`ROTATION`**, not `QUATERNION`.

## FrameId / BatchFrameId

A coordinate-frame **name**, encoded as an Arrow `string` array.

```python
from t4perceval.component import BatchFrameId, FrameId

FrameId("lidar").name  # 'lidar'
BatchFrameId(["lidar", "radar"]).names()  # ('lidar', 'radar')
BatchFrameId(["lidar", "radar"]).matching("radar")  # array([1]) -- rows naming 'radar'
```

A frame name is opaque -- the package never parses one -- so a ROS-namespaced
`/robot1/base_link` means exactly what it says. Names must be non-empty strings.

Descriptor: `CHILD_FRAME_ID`.

## Why separate descriptors

`TRANSLATION` and `ROTATION` are deliberately **not** `POSITION` and `QUATERNION`. A transform is a
relationship _between_ frames, not a thing _in_ a frame. Separate names stop a system that asks for
a 3D position -- a distance filter, say -- from being pointed at a transform entity and appearing to
work.

Being mono is the second half of the same guard: `Transform3D` describes one relationship, of which
an entity holds exactly one per point in time, so "what if it has three rows?" is not a question the
type can be asked.

## Where to go next

- [Transform3D](../archetypes/transform-3d.md) -- the archetype these make up.
- [Coordinate system](../concepts/coordinate-system.md) -- how the graph works.
