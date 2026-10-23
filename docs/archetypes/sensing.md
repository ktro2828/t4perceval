# SensingResults

What the sensor saw of each ground-truth box.

## Schema

| Component        | Requirement | Shape    | dtype | Description                                           |
| :--------------- | :---------- | :------- | :---- | :---------------------------------------------------- |
| `num_points`     | Required    | `(N,)`   | `i32` | points of the evaluated cloud inside the scaled box   |
| `sensing_status` | Required    | `(N,)`   | `i8`  | `DETECTED` / `FAILED` / `WARNING`, a `SensingStatus`  |
| `nearest_point`  | Required    | `(N, 3)` | `f64` | the inside point closest to the origin; `NaN` if none |

```python
from t4perceval import SensingResults
from t4perceval.component import SensingStatus

results = SensingResults(
    num_points=[12, 0],
    sensing_status=[SensingStatus.DETECTED, SensingStatus.FAILED],
    nearest_point=[[8.2, 0.3, 0.5], [float("nan")] * 3],
)
results.num_detected, results.num_failed, results.num_warned  # (1, 1, 0)
results.is_detected  # array([ True, False])
results.count(SensingStatus.FAILED)  # 1
```

Construction checks the status values, that counts are non-negative, and that `nearest_point` is
`NaN` exactly on the rows whose count is zero.

## Aligned with the boxes

`SensingDetectionSystem` writes this archetype with the **objects entity's partitioning**: row `i`
of `<objects>/sensing` is the verdict on row `i` of `<objects>` at the same frame, the way a mask is
aligned with what it masks. Nothing here names the box -- no instance id, no class -- because the box
is one index away, and a per-class breakdown is a join by row, which is what
`SensingDetectionRateSystem` does.

This replaces `DynamicObjectWithSensingResult`, which held the object and the inside points and so
could not be stored. The points themselves are not kept: the cloud is in the store, and the scaled
box is a function of the object and two parameters.

## `num_points` is the annotation's descriptor

The inside count uses the same `NUM_POINTS` descriptor as the `num_points` an importer puts on a
ground-truth box. The meaning is the same, "sensor points inside this box", so
`FilterByNumPointsSystem.on(sensing.target, min_num_points=5)` re-judges a result without
re-counting. The numbers differ: the annotation's count came from the dataset's own cloud and an
unscaled box; this one from the cloud you evaluated and the scaled box. Join them by row when
comparing.

## Where it is written

| System                   | Entity              |
| :----------------------- | :------------------ |
| `SensingDetectionSystem` | `<objects>/sensing` |

## Where to go next

- [Sensing evaluation](../evaluation/sensing.md) -- the task this records.
- [Sensing systems](../systems/sensing.md) -- the systems that write and read it.
