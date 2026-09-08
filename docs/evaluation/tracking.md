# Tracking

Matching identified objects across frames, and scoring the identities with CLEAR.

## Overview

A tracking evaluation is a detection evaluation whose objects carry a **persistent instance id**, so
a metric can ask "is this the same object the estimator called `7` last frame?".

```text
/ground_truth/objects ─┐
                       ├──▶ filters ──▶ matcher ──▶ ClearSystem ──▶ MOTA / MOTP / ID switches
/estimation/objects  ──┘
```

## Inputs

### Ground truth

```python
from t4perceval import InstanceRegistry, Trackings3D

instances = InstanceRegistry()

store.log(
    "/ground_truth/objects",
    Trackings3D(
        position=[[0.0, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]],
        size=[[1.9, 4.5, 1.6]],
        class_id=labels.encode(["car"]),
        confidence=[1.0],
        instance_id=instances.encode(["7f3c...-a1"]),
    ),
    at=TimePoint.at(frame=0),
    frame_id="base_link",
)
```

### Estimation

Same archetype, with the tracker's own ids. The two sides' id spaces are **independent** -- a metric
compares identity _continuity_, not id equality -- so the tracker's own integers are fine. If both
sides are interned into one registry, use distinct namespaces (`instance_namespace="gt"` and
`"est"` is what the importers do).

## Required components

```text
Trackings3D
├── position      required   3D centre
├── quaternion    required   orientation
├── size          required   (width, length, height)
├── class_id      required
├── confidence    required
└── instance_id   required   (N,) i64 -- what makes this a tracking
```

`ClearSystem` requires `class_id` and `instance_id` on **both** sides. Pointing it at plain
`Detections3D` raises:

```text
ValueError: /estimation/objects is missing required component(s): instance_id
```

## Optional components

`velocity`, `num_points` and `visibility`, exactly as in [Detection 3D](detection-3d.md#optional-components).

`Trackings2D` is the image-plane counterpart: `roi` plus `instance_id`.

## Filtering

Everything in [Filtering](../user-guide/filtering.md) applies, plus:

```python
from t4perceval.system import FilterByInstanceSystem

FilterByInstanceSystem.on("/ground_truth/objects", exclude=["7f3c...-a1"])
```

!!! warning "Filtering changes ID-switch counts"

    A filter that removes an object in some frames but not others creates gaps in its track, and a
    gap can read as a switch. Filter both sides identically, and prefer filters that are stable over
    a track's lifetime (class, region) over ones that flicker (confidence near a threshold).

## Matching

Any 3D matcher works. Matching is done **per frame**, and the identity bookkeeping is what spans
frames:

```python
from t4perceval.system import CenterDistanceMatchingSystem

CenterDistanceMatchingSystem.between("/estimation/objects", "/ground_truth/objects", threshold=1.0)
```

MOTP is the mean matching score over the true positives, so the matcher you choose changes MOTP's
units: centre distance gives metres, IoU gives an overlap fraction.

## Metrics

```python
from t4perceval.system import ClearSystem

clear = ClearSystem.on(matcher.target, "/estimation/objects", "/ground_truth/objects")
[str(t) for t in clear.targets]
# ['/metrics/clear/mota', '/metrics/clear/motp', '/metrics/clear/id_switch']
```

All three come out of one pass over the tracked identities, which is why they are one system with
three targets rather than three systems. Move the family root with `target=`:

```python
ClearSystem.on(matcher.target, EST, GT, target="/metrics/clear_bev")
```

The run must cover **several frames** -- `TimeRange.everything()`, not `TimeRange.single(0)` -- or
there is no continuity to score.

## Complete example

```python
from t4perceval import FRAME, LabelRegistry, MetricValues, Store, TimePoint, TimeRange, Trackings3D
from t4perceval.system import CenterDistanceMatchingSystem, ClearSystem, Pipeline, SystemContext

labels = LabelRegistry.from_names(["car", "pedestrian"])
store = Store()

for frame in range(3):
    store.log(
        "/ground_truth/objects",
        Trackings3D(
            position=[[float(frame), 0.0, 0.0]],
            quaternion=[[0.0, 0.0, 0.0, 1.0]],
            size=[[2.0, 4.0, 2.0]],
            class_id=labels.encode(["car"]),
            confidence=[1.0],
            instance_id=[1],
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )
    store.log(
        "/estimation/objects",
        Trackings3D(
            position=[[float(frame) + 0.2, 0.0, 0.0]],
            quaternion=[[0.0, 0.0, 0.0, 1.0]],
            size=[[2.0, 4.0, 2.0]],
            class_id=labels.encode(["car"]),
            confidence=[0.9],
            instance_id=[7 if frame < 2 else 8],  # the tracker loses the id at frame 2
        ),
        at=TimePoint.at(frame=frame),
        frame_id="base_link",
    )

matcher = CenterDistanceMatchingSystem.between(
    "/estimation/objects", "/ground_truth/objects", threshold=1.0
)
clear = ClearSystem.on(matcher.target, "/estimation/objects", "/ground_truth/objects")

Pipeline([matcher, clear]).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

for target in clear.targets:
    values = store.range(target, timeline=FRAME, time_range=TimeRange.everything()).materialize(
        MetricValues
    )
    print(target.name, values.value.values)
```

```text
mota      [0.66666667        nan]
motp      [0.2               nan]
id_switch [1.                nan]
```

Three frames, one ground-truth object per frame, three true positives, no false positives and one
switch:

```text
MOTA = max(0, (TP - FP - switches) / ground-truth count) = (3 - 0 - 1) / 3 = 0.667
MOTP = mean matching score over the true positives          = 0.2 m
```

The `nan` is the `pedestrian` class, which has no ground truth in range.

## Known divergences

ID switches are counted only over rows that were true positives, so a frame where the object was
missed disappears before switch counting and a lost-then-recovered identity can read as a switch;
MOTA is clamped at zero, where standard CLEAR permits negative values; and only the ground-truth
side of a switch is counted. All of these are catalogued in
[Metric divergences](../development/metric-divergences.md).

`HotaSystem` is [on the roadmap](../development/roadmap.md), not implemented.

## Where to go next

- [Prediction](prediction.md) -- tracking plus future trajectories.
- [Trackings3D](../archetypes/tracking.md) -- the archetype's schema.
