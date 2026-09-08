# Entity, component, archetype

Three words carry most of the model. This page defines each one and shows how they fit together.

```text
Entity path        /ground_truth/objects
   │                       │
   │                       ├── position    (N, 3) f64     ┐
   │                       ├── quaternion  (N, 4) f64     │  components
   │                       ├── size        (N, 3) f64     │
   │                       ├── class_id    (N,)   i32     │
   │                       └── confidence  (N,)   f64     ┘
   │                       └──────────── which of those must be present,
   │                                     and what they are called: Detections3D
   ▼
one write of those columns at one time = a Chunk
```

## Entity paths

An **entity** is a `/`-separated path. It says _where_ data is filed, and nothing else.

```python
from t4perceval import EntityPath

path = EntityPath.parse("/ground_truth/objects")
path.name  # 'objects'
str(path.parent)  # '/ground_truth'
path / "kept"  # /ground_truth/objects/kept
path.is_descendant_of(EntityPath.parse("/ground_truth"))  # True
```

Anywhere the API takes an entity path, a plain string works too.

Paths are a **filing decision**, not a type. The convention the built-in systems follow is:

| Path                                | Written by                             |
| :---------------------------------- | :------------------------------------- |
| `/ground_truth/objects`             | you, or an importer                    |
| `/estimation/objects`               | you, or an importer                    |
| `/estimation/objects/filter/<name>` | a [filter](../user-guide/filtering.md) |
| `/estimation/objects/kept`          | `ApplyMaskSystem`                      |
| `/matching/<mode>`                  | a [matcher](../user-guide/matching.md) |
| `/metrics/<name>`                   | a [metric](../user-guide/metrics.md)   |
| `/tf/<child frame>`                 | an importer's transform edges          |

Keeping a filter's mask _under_ the entity it judges means a prefix query finds an entity together
with every verdict recorded about it.

!!! note "A frame name is not an entity path"

    Transform edges are found by reading chunks, not by parsing paths, so a coordinate frame may
    contain a `/` and a graph can be re-filed without being renamed. See
    [Coordinate system](coordinate-system.md).

## Components

A **component** is one column: a NumPy array with a fixed per-row shape and dtype, plus a
[descriptor](data-model.md#descriptors) naming what it means.

```python
from t4perceval.component import BatchClassId

BatchClassId([0, 1, 0]).values  # array([0, 1, 0], dtype=int32)
```

The descriptor is not a property of the column type -- the same `BatchClassId` column means
`class_id` in a detection and `ground_truth_class_id` in a confusion matrix. It is the archetype
field that fixes it:

```python
from t4perceval import ConfusionMatrix, Detections3D

Detections3D.descriptor_of("class_id").component  # 'class_id'
ConfusionMatrix.descriptor_of("ground_truth_class_id").component  # 'ground_truth_class_id'
```

Two flavours exist:

- **Columnar** (`ColumnarComponent`, everything prefixed `Batch*`): `N` rows of something.
- **Mono** (`MonoComponent`): exactly one value, written and read without a row index. Only
  [`Transform3D`](../archetypes/transform-3d.md) uses these -- it describes one relationship between
  two frames, not `N` objects.

```python
from t4perceval.component import Position3D

Position3D([1.2, 0.0, 1.8]).value  # array([1.2, 0. , 1.8]) -- not a (1, 3) column
```

A mono value is widened into its `Batch*` counterpart on the way into a chunk, so storage stays
columnar and a range query over three ego samples still returns a three-row column.

The full catalogue is in [Components](../components/index.md).

## Archetypes

An **archetype** is a validated bundle of components: which are required, which are optional, and
what each field is called.

```python
from t4perceval import Detections3D

[d.component for d in Detections3D.required_descriptors()]
# ['position', 'quaternion', 'size', 'class_id', 'confidence']
[d.component for d in Detections3D.optional_descriptors()]
# ['velocity', 'num_points', 'visibility']
```

Constructing one checks that every column has the same length, that the required ones are present,
and that shapes and dtypes are what the schema says.

```python
detections = Detections3D(
    position=[[0.0, 0.0, 0.0]],
    quaternion=[[0.0, 0.0, 0.0, 1.0]],
    size=[[1.9, 4.5, 1.6]],
    class_id=[0],
    confidence=[0.9],
)
len(detections)  # 1
detections.has(*Detections3D.required_descriptors())  # True
```

### Composition, not inheritance

`Trackings3D` does not inherit from `Detections3D`. It re-declares the same box components, which
resolve to the _same descriptors_:

```python
from t4perceval import Detections3D, Trackings3D

tracking = Trackings3D(
    position=[[0.0, 0.0, 0.0]],
    quaternion=[[0.0, 0.0, 0.0, 1.0]],
    size=[[1.9, 4.5, 1.6]],
    class_id=[0],
    confidence=[0.9],
    instance_id=[7],
)
tracking.has(*Detections3D.required_descriptors())  # True
```

So any system that requires a detection's components runs unchanged against a tracking archetype --
and against `Predictions3D`, which composes the box, instance _and_ trajectory columns. Inheritance
would have made `Predictions3D` a `Trackings3D` and forced every consumer to know the hierarchy;
composition lets a system state its requirement and stop caring.

The full catalogue is in [Archetypes](../archetypes/index.md).

## Selecting rows

Components, archetypes and chunks all support `select`, which produces **independent** data:

```python
import numpy as np

near = detections.select(np.array([True]))  # a boolean mask
first = detections.select([0])  # or indices
```

Lazy narrowing is [`EntityView`](store.md#entityview)'s job, not `select`'s.

## Where to go next

- [Store](store.md) -- how chunks of these columns are logged and queried.
- [Components](../components/index.md) and [Archetypes](../archetypes/index.md) -- the catalogues.
- [Extending archetypes](../development/extending-archetypes.md) -- adding your own.
