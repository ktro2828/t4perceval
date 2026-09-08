# Components

A component is one column: a NumPy array of `N` rows with a fixed per-row shape and dtype, named by
a [descriptor](../concepts/data-model.md#descriptors). Columns are always **read-only** and never
share memory with a writable array you passed in.

This section is a **schema catalogue**: what each column means, what shape it has, and what
depends on it.

| Page                      | Covers                                                       |
| :------------------------ | :----------------------------------------------------------- |
| [Geometry](geometry.md)   | positions, rotations, sizes, velocities, ROIs, pixels        |
| [Object](object.md)       | class, confidence, instance, quality, trajectories           |
| [Transform](transform.md) | translation, rotation and frame names -- the mono components |
| [Metrics](metrics.md)     | masks, match verdicts, metric values, counts                 |

## Every column at a glance

| Component                                           | Shape          | dtype          | Notes                                      |
| :-------------------------------------------------- | :------------- | :------------- | :----------------------------------------- |
| `BatchPosition3D` / `BatchVelocity` / `BatchSize3D` | `(N, 3)`       | `f64`          | `BatchSize3D` is `(width, length, height)` |
| `BatchPosition2D` / `BatchSize2D`                   | `(N, 2)`       | `f64`          |                                            |
| `BatchQuaternion`                                   | `(N, 4)`       | `f64`          | `xyzw` order (SciPy's convention)          |
| `BatchRoi`                                          | `(N, 4)`       | `i32`          | `(x_min, y_min, height, width)`            |
| `BatchClassId`                                      | `(N,)`         | `i32`          | meaning comes from `LabelRegistry`         |
| `BatchConfidence`                                   | `(N,)`         | `f64`          | constrained to `[0, 1]`                    |
| `BatchInstanceId`                                   | `(N,)`         | `i64`          | interned by `InstanceRegistry`             |
| `BatchNumPoints` / `BatchPixel`                     | `(N,)`         | `i32`          |                                            |
| `BatchVisibility` / `BatchMatchStatus`              | `(N,)`         | `i8`           | ordered enums                              |
| `BatchMask`                                         | `(N,)`         | `bool`         | a filter's verdict                         |
| `BatchRowIndex`                                     | `(N,)`         | `i64`          | `-1` means "no counterpart"                |
| `BatchMatchingScore` / `BatchThreshold`             | `(N,)`         | `f64`          |                                            |
| `BatchMetricValue`                                  | `(N,)`         | `f64`          | `NaN` when undefined                       |
| `BatchSupport` / `BatchCount`                       | `(N,)`         | `i64`          | non-negative                               |
| `BatchWaypoints3D`                                  | `(N, M, T, 3)` | `f64`          | `M` modes, `T` timesteps                   |
| `BatchModeConfidence` / `BatchModeValid`            | `(N, M)`       | `f64` / `bool` |                                            |
| `BatchTimestepValid`                                | `(N, M, T)`    | `bool`         |                                            |
| `BatchTimeOffset`                                   | `(N, T)`       | `i64`          | nanoseconds, strictly increasing           |
| `Position3D` / `Quaternion` / `FrameId`             | one value      | `f64` / `str`  | **mono** -- see [Transform](transform.md)  |

## Naming rules

- A component that holds several rows is prefixed **`Batch`**.
- Its **mono** counterpart -- one value, no row index -- drops the prefix: `Position3D`,
  `Quaternion`, `FrameId`.
- A descriptor is named for **meaning**, not for the archetype that declared it: `POSITION` is
  `"position"` in `Detections3D` and in `Trackings3D` alike.

## Shared surface

Every column type has the same handful of operations:

```python
from t4perceval.component import BatchPosition3D

column = BatchPosition3D([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]])

len(column)  # 2
column.values  # the read-only (2, 3) array
column.row_shape  # (3,)
column.as_array()  # the same array
column.select([0])  # independent data, not a view
column.to_arrow()  # a pyarrow array
BatchPosition3D.empty()  # zero rows, correct shape and dtype
BatchPosition3D.from_array(values)
```

Shapes with wildcard dimensions need them supplied when empty: `BatchWaypoints3D.empty(2, 5)` for
2 modes and 5 timesteps.

## Where to go next

- [Archetypes](../archetypes/index.md) -- how these columns are bundled.
- [Extending components](../development/extending-components.md) -- adding your own.
