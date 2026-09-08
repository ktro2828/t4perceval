# Object components

Class, confidence, identity, ground-truth quality, and trajectories.

## BatchClassId

`(N,)` `i32` -- a class index. What the integer _means_ comes from a
[`LabelRegistry`](../reference/api/label.md), not from the column.

```python
from t4perceval import LabelRegistry

labels = LabelRegistry.from_names(["car", "bicycle", "pedestrian"])
labels.encode(["car", "pedestrian"])  # array([0, 2], dtype=int32)
labels.decode([0, 2])  # ('car', 'pedestrian')
```

Descriptor: `CLASS_ID`. Two sentinel values are reserved:

| Value | Name                               | Meaning                                                     |
| ----: | :--------------------------------- | :---------------------------------------------------------- |
|  `-1` | `UNKNOWN_CLASS_ID` / `ALL_CLASSES` | an unmapped class, or a metric row aggregating over classes |
|  `-2` | `BACKGROUND_CLASS_ID`              | the "nothing here" axis of a confusion matrix               |

The confusion matrix re-uses the same column type under two other descriptors,
`GROUND_TRUTH_CLASS_ID` and `ESTIMATION_CLASS_ID` -- one more illustration that the descriptor, not
the column type, carries the meaning.

## BatchConfidence

`(N,)` `f64`, constrained to `[0, 1]`. Out-of-range values raise at construction.

Descriptor: `CONFIDENCE`. `AveragePrecisionSystem` ranks by it, and
`FilterByConfidenceSystem` thresholds it. Ground truth conventionally uses `1.0`.

## BatchInstanceId

`(N,)` `i64` -- a persistent per-object identifier. Interned from dataset UUID strings by an
`InstanceRegistry`:

```python
from t4perceval import InstanceRegistry

instances = InstanceRegistry()
instances.encode(["7f3c...-a1", "8b2d...-c4"])  # array([0, 1])
instances.uuid(0)  # '7f3c...-a1'
instances.instance_id("7f3c...-a1")  # 0, without interning
```

Descriptor: `INSTANCE_ID`. Required by `Trackings2D`, `Trackings3D` and `Predictions3D`, and by
`ClearSystem` and `PathDisplacementSystem`.

## BatchNumPoints

`(N,)` `i32` -- how many sensor points fall inside a 3D box. A ground-truth quality signal, used by
`FilterByNumPointsSystem` to drop annotations no sensor could have supported.

Descriptor: `NUM_POINTS`.

## BatchVisibility

`(N,)` `i8` -- an ordered occlusion level.

```python
from t4perceval.component import VisibilityLevel

VisibilityLevel.UNAVAILABLE  # -1, sorts below every real level
VisibilityLevel.NONE  #  0, 90-100% occluded
VisibilityLevel.PARTIAL  #  1, occluded by more than 50%
VisibilityLevel.MOST  #  2, occluded by less than 50%
VisibilityLevel.FULL  #  3, not occluded
```

Numeric and ordered so a `>=` threshold works directly on the column, unlike the string enum the
dataset schema uses. `UNAVAILABLE` sorts below every real level so a threshold never accidentally
accepts it.

Descriptor: `VISIBILITY`. Used by `FilterByVisibilitySystem`.

## Trajectory components

Four columns describe multi-modal futures for `N` objects over `M` modes and `T` timesteps. `M` and
`T` are fixed within one instance; shorter trajectories are padded and masked.

| Component             | Shape          | dtype  | Meaning                                       |
| :-------------------- | :------------- | :----- | :-------------------------------------------- |
| `BatchWaypoints3D`    | `(N, M, T, 3)` | `f64`  | the predicted positions                       |
| `BatchModeConfidence` | `(N, M)`       | `f64`  | per-mode confidence, in `[0, 1]`              |
| `BatchModeValid`      | `(N, M)`       | `bool` | which modes carry real data                   |
| `BatchTimestepValid`  | `(N, M, T)`    | `bool` | which timesteps of which mode carry real data |
| `BatchTimeOffset`     | `(N, T)`       | `i64`  | nanoseconds from now, strictly increasing     |

```python
from t4perceval.component import BatchWaypoints3D

waypoints = BatchWaypoints3D(np.zeros((4, 6, 12, 3)))
waypoints.num_modes  # 6
waypoints.num_timesteps  # 12
```

Descriptors: `WAYPOINTS`, `MODE_CONFIDENCE`, `MODE_VALID`, `TIMESTEP_VALID`, `TIME_OFFSET`.

!!! warning "The validity masks are not applied by the metric yet"

    `PathDisplacementSystem` does not read `mode_valid`, `timestep_valid` or `time_offset`. See
    [Prediction](../evaluation/prediction.md#optional-components).

## Where to go next

- [Geometry components](geometry.md) · [Metric components](metrics.md)
- [Archetypes](../archetypes/index.md) -- how these are bundled.
