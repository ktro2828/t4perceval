# Data model

## Columns, not objects

Every component is **one column of `N` rows** with a fixed per-row shape and dtype. A frame with 200
detections is a handful of NumPy arrays, not 200 Python objects.

```python
from t4perceval.component import BatchPosition3D

positions = BatchPosition3D([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]])
positions.values.shape  # (2, 3)
positions.values.dtype  # float64
len(positions)  # 2
```

Three rules hold everywhere:

1. **Columns are read-only.** `values.flags.writeable` is `False`, and a component never shares
   memory with a writable array you passed in. A view handed to a metric cannot be mutated under it.
2. **The batch dimension is explicit.** A batch of predicted trajectories is `(N, M, T, 3)` --
   objects, modes, timesteps, spatial dimensions -- not a list of lists.
3. **Zero rows is legal.** An empty frame is an ordinary frame, in every archetype.

### Why it matters

|                   | `List[DynamicObject]`    | Component columns   |
| :---------------- | :----------------------- | :------------------ |
| Work per frame    | a Python loop            | one vectorized call |
| Unused fields     | present and `None`       | absent              |
| Cross-frame query | traverse a nest of lists | one range query     |
| Intermediates     | discarded                | logged as entities  |

The [benchmark](../development/benchmarks.md) puts numbers on the first row: 67x to 224x faster on
the metric workloads, and 16x smaller in retained memory.

## Descriptors

A **descriptor** names a column by _what it means_, not by which archetype declared it.

```python
from t4perceval.descriptors import POSITION
from t4perceval import Detections3D, Trackings3D

Detections3D.descriptor_of("position") == POSITION  # True
Trackings3D.descriptor_of("position") == POSITION  # True
```

Both archetypes expose their 3D centre as `POSITION`. That is the whole reason a system can declare

```python
REQUIRES = (POSITION,)
```

and then run against detections, trackings _and_ predictions, instead of branching on an
`EvaluationTask` enum. A descriptor carries an optional `archetype` hint and a `component_type`
name, but **neither affects identity** -- only the component name does.

The canonical descriptors live in [`t4perceval.descriptors`](../reference/api/core.md)
and are catalogued in [Components](../components/index.md).

### Deliberately distinct names

A [`Transform3D`](../archetypes/transform-3d.md) uses `TRANSLATION` and `ROTATION`, not `POSITION`
and `QUATERNION`. A transform is a relationship _between_ frames, not a thing _in_ a frame; separate
names stop a distance filter from being pointed at a transform entity and appearing to work.

## Read-only, and why

```python
import numpy as np
from t4perceval.component import BatchPosition3D

source = np.zeros((2, 3))
column = BatchPosition3D(source)
source[0, 0] = 99.0

column.values[0, 0]  # 0.0 -- the column did not alias `source`
column.values.flags.writeable  # False
```

Chunks are frozen and their arrays read-only, which is what makes moving an entity between stores
close to free: the new store references the same objects rather than copying them. See
[Evaluate a T4 dataset](../recipes/evaluate-t4-dataset.md) for where that shows up.

## Meaning for the integers

`class_id` and `instance_id` are integer columns. What those integers _mean_ is not in the store --
it is in a registry, and the two travel together in a [`Recording`](../user-guide/persistence.md).

```python
from t4perceval import LabelRegistry

labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian"])
labels.encode(["car", "pedestrian"])  # array([0, 2], dtype=int32)
labels.decode([0, 2])  # ('car', 'pedestrian')
```

Ids are assigned in first-seen order, so **two sources that each derive their own registry are both
valid and silently incompatible**. Hand the same registry to every importer in a run; the
disagreement would otherwise surface as plausible numbers, not an error. `Recording.agrees_with`
and `LabelRegistry.fingerprint` exist to catch it, and
[`t4perceval.reconcile`](../reference/api/label.md) to repair it.

## Where to go next

- [Entity, component, archetype](entity-component-archetype.md) -- how columns get named and bundled.
- [Store](store.md) -- where chunks live.
- [Data model design](../development/design/en/data_model.md) -- the long-form rationale, in English and Japanese.
