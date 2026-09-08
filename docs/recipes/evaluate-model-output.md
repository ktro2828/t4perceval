# Evaluate model output

**Goal:** you have your model's predictions and the matching ground truth as arrays in Python, and
you want mAP.

No dataset, no extras, no importer -- this is the base install.

## The shape of the answer

```text
your arrays ──▶ Store.log per frame ──▶ narrow ──▶ average_precision_sweep ──▶ read /metrics/map
```

## 1. Fix a label registry

Do this **once**, before anything else, and use the same object for both sides. Class ids are
assigned in first-seen order, so two registries built independently are silently incompatible.

```python
from t4perceval import LabelRegistry

labels = LabelRegistry.from_names(["car", "truck", "bicycle", "pedestrian"])
```

If your model emits ids rather than names, build the registry in _your_ id order so the integers
line up:

```python
labels = LabelRegistry.from_names([name for _, name in sorted(MY_MODEL_CLASSES.items())])
```

## 2. Log both sides, frame by frame

```python
from t4perceval import Detections3D, Store, TimePoint

store = Store()

for frame, (gt, est) in enumerate(zip(ground_truth_frames, estimation_frames)):
    store.log(
        "/ground_truth/objects",
        Detections3D(
            position=gt["xyz"],  # (N, 3) f64
            quaternion=gt["quat_xyzw"],  # (N, 4) f64, xyzw
            size=gt["wlh"],  # (N, 3) f64, (width, length, height)
            class_id=labels.encode(gt["names"]),
            confidence=[1.0] * len(gt["xyz"]),
        ),
        at=TimePoint.at(frame=frame, timestamp_ns=gt["stamp_ns"]),
        frame_id="base_link",
    )
    store.log(
        "/estimation/objects",
        Detections3D(
            position=est["xyz"],
            quaternion=est["quat_xyzw"],
            size=est["wlh"],
            class_id=labels.encode(est["names"]),
            confidence=est["score"],
        ),
        at=TimePoint.at(frame=frame, timestamp_ns=est["stamp_ns"]),
        frame_id="base_link",
    )
```

Three things to get right:

- **`quaternion` is `xyzw`.** Convert if your model emits `wxyz`.
- **`size` is `(width, length, height)`.** Most detectors emit `(length, width, height)`; swap the
  first two.
- **Log empty frames too.** A frame with no detections is `np.empty((0, 3))` and friends, not a
  skipped iteration -- skipping it silently removes its false negatives from the denominator.

Both sides must use the same `frame_id`, or the
[cross-frame guard](../concepts/coordinate-system.md#the-cross-frame-guard) will refuse to match.

## 3. Narrow to the evaluated region

Filter **both** sides identically, and materialize -- recall divides by the ground-truth count.

```python
from t4perceval import FRAME, TimeRange
from t4perceval.system import ApplyMaskSystem, FilterByDistanceSystem, Pipeline, SystemContext

ctx = SystemContext(store, FRAME, labels=labels)
scene = TimeRange.everything()

narrow = []
for path in ("/ground_truth/objects", "/estimation/objects"):
    near = FilterByDistanceSystem.on(path, max_distance=50.0)
    narrow += [near, ApplyMaskSystem.of(path, near.target)]
Pipeline(narrow).run(ctx, scene)
```

Skip this step entirely if you evaluate everything.

## 4. Match and score

```python
from t4perceval.system import average_precision_sweep

Pipeline(
    average_precision_sweep(
        "/estimation/objects/kept",
        "/ground_truth/objects/kept",
        thresholds=[0.5, 1.0, 2.0, 4.0],
        heading=True,
    )
).run(ctx, scene)
```

## 5. Read the result

```python
from t4perceval import MetricValues

for path in ("/metrics/map", "/metrics/maph"):
    values = store.range(path, timeline=FRAME, time_range=scene).materialize(MetricValues)
    print(path, round(values.aggregate, 4))

per_class = store.range("/metrics/map", timeline=FRAME, time_range=scene).materialize(MetricValues)
for name in labels.names:
    print(name, per_class.of_class(labels.class_id(name)))
```

## Variations

=== "Tracking instead"

    Log `Trackings3D` with an `instance_id`, and swap the metric:

    ```python
    from t4perceval.system import CenterDistanceMatchingSystem, ClearSystem

    matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
    Pipeline([matcher, ClearSystem.on(matcher.target, EST, GT)]).run(ctx, scene)
    ```

    See [Tracking](../evaluation/tracking.md).

=== "2D instead"

    Log `Detections2D` with a `roi`, use `IoURoiMatchingSystem`, and set `frame_id` to the camera
    channel. See [Detection 2D](../evaluation/detection-2d.md).

=== "IoU instead of centre distance"

    ```python
    from t4perceval.system import IoUBEVMatchingSystem

    average_precision_sweep(EST, GT, matcher=IoUBEVMatchingSystem, thresholds=[0.3, 0.5, 0.7])
    ```

=== "Per-class thresholds"

    ```python
    from t4perceval.system import Thresholds

    average_precision_sweep(
        EST,
        GT,
        thresholds=[
            Thresholds(2.0, by_class=(("pedestrian", 0.5),)),
            Thresholds(4.0, by_class=(("pedestrian", 1.0),)),
        ],
    )
    ```

## Then what?

Everything the run produced is still in the store -- masks, verdicts, per-threshold AP. See
[Offline analysis](../user-guide/offline-analysis.md) for asking _why_ a number came out the way it
did.

## Where to go next

- [First evaluation](../getting-started/first-evaluation.md) -- the same thing, smaller.
- [Evaluate a T4 dataset](evaluate-t4-dataset.md) -- when the ground truth is a dataset.
