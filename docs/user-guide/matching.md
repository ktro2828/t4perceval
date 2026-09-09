# Matching

A matcher pairs an estimation entity against a ground-truth entity and writes one
[`MatchResults`](../archetypes/matching.md) row per verdict.

```python
from t4perceval.system import CenterDistanceMatchingSystem

matcher = CenterDistanceMatchingSystem.between(
    "/estimation/objects",
    "/ground_truth/objects",
    threshold=1.0,
)
matcher.target  # /matching/center_distance
```

## The matching family

| System                            | Requires                                     | Score                          | Better | Default threshold |
| :-------------------------------- | :------------------------------------------- | :----------------------------- | :----- | ----------------: |
| `CenterDistanceMatchingSystem`    | `position`, `class_id`                       | 3D distance between centres    | lower  |               1.0 |
| `CenterDistanceBEVMatchingSystem` | `position`, `class_id`                       | xy distance between centres    | lower  |               1.0 |
| `PlaneDistanceMatchingSystem`     | `position`, `quaternion`, `size`, `class_id` | distance between nearest faces | lower  |               2.0 |
| `IoUBEVMatchingSystem`            | `position`, `quaternion`, `size`, `class_id` | IoU of footprints              | higher |               0.5 |
| `IoU3DMatchingSystem`             | `position`, `quaternion`, `size`, `class_id` | IoU of volumes                 | higher |               0.5 |
| `IoURoiMatchingSystem`            | `roi`, `class_id`                            | IoU of image-plane regions     | higher |               0.5 |

All six share a base, so they take the same parameters and write the same columns. Every one writes
to `/matching/<mode>` unless you pass `target=`.

The geometry behind them is vectorized and pairwise; see
[`t4perceval.geometry`](../reference/api/geometry.md).

## What a match result holds

One row per verdict, with `-1` meaning "no counterpart":

| Column           | Meaning                                        |
| :--------------- | :--------------------------------------------- |
| `est_index`      | row index into the estimation chunk, or `-1`   |
| `gt_index`       | row index into the ground-truth chunk, or `-1` |
| `matching_score` | the score this pair got                        |
| `match_status`   | `MatchStatus.TP` / `FP` / `FN`                 |
| `threshold`      | the threshold the verdict was reached at       |

```python
from t4perceval import MatchResults

result = store.range(matcher.target, timeline=FRAME, time_range=TimeRange.everything()).materialize(
    MatchResults
)

result.num_tp, result.num_fp, result.num_fn
result.count(MatchStatus.FP)
```

A false positive is a row with `gt_index == -1`, a false negative one with `est_index == -1`. The
`threshold` column is recorded because it is the one thing a later stage could not recover by
following the indices back to the objects -- and with per-class thresholds it differs from row to
row.

## Assignment

Pairs are chosen by a globally optimal **linear-sum (Hungarian) assignment** over the score matrix,
subject to the threshold and -- unless `class_agnostic=True` -- to the two rows sharing a class.

That differs from `autoware_perception_evaluation`, which walks estimations in confidence order and
greedily takes the best still-available ground truth. On scenes where each estimate has exactly one
feasible ground truth the two agree exactly; on dense scenes they do not. See
[Metric divergences](../development/metric-divergences.md).

## Thresholds

A threshold is either a number or a `Thresholds`:

```python
from t4perceval.system import Thresholds

CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)

CenterDistanceMatchingSystem.between(
    EST, GT, threshold=Thresholds(2.0, by_class=(("pedestrian", 0.5),))
)
```

`Thresholds(default, by_class=...)` names the classes that differ; every other class gets
`default`.

!!! warning "A bare mapping is rejected"

    Passing `threshold={"car": 2.0}` raises, because a mapping alone does not say what the classes
    it omits should get -- and defaulting them to "always feasible" would silently inflate the
    result:

    ```text
    ValueError: A per-class threshold mapping needs a default for the classes it omits;
    pass Thresholds(default, by_class=...) explicitly
    ```

    `Thresholds.coerce(mapping, default=1.0)` is the explicit form if you already hold a dict.

Per-class thresholds are keyed by the **ground-truth** class. Keys may be class names or raw ids;
names are resolved through the registry on the `SystemContext`, so `labels=` must be set there.

## Class-agnostic matching

```python
CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0, class_agnostic=True)
```

Pairs may then cross classes, which is what a
[confusion matrix](metrics.md#confusion-matrix) needs: a car detected as a truck should be a
mislabelled true positive, not one FP plus one FN.

## Coordinate frames

A matcher refuses to compare geometry across two _different, stated_ frames:

```text
ValueError: Cannot compare geometry across coordinate frames: /estimation/objects in 'base_link',
/ground_truth/objects in 'map'. Bring the inputs into one frame first.
```

Bring one side over with `TransformEntitySystem` and match its target
([Transforms](transforms.md#expressing-an-entity-in-another-frame)); `check_frames=False` opts out
per system when the frames are known to coincide. See
[Coordinate system](../concepts/coordinate-system.md#the-cross-frame-guard).

## Sweeping thresholds

Each threshold needs its **own matching run**: a threshold changes which pairs the assignment is
allowed to make, and re-thresholding one loose run would not free a rejected pair's counterpart to
be matched elsewhere.

`average_precision_sweep` does that for you:

```python
from t4perceval.system import Pipeline, average_precision_sweep

Pipeline(
    average_precision_sweep(
        "/estimation/objects",
        "/ground_truth/objects",
        thresholds=[0.5, 1.0, 2.0, 4.0],
    )
).run(ctx, TimeRange.everything())
```

Targets are indexed by declaration order -- `/matching/center_distance/0`, `/metrics/ap/0`, ... --
rather than by threshold value, since a per-class threshold has no single value to put in a path.

Pass `matcher=IoUBEVMatchingSystem` to sweep a different mode, and per-class `Thresholds` objects in
the `thresholds` list to sweep per-class values.

## Where to go next

- [Metrics](metrics.md) -- turning verdicts into numbers.
- [Write a custom matcher](../recipes/custom-matcher.md).
- [MatchResults](../archetypes/matching.md) -- the archetype's schema.
