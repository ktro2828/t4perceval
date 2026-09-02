# Metrics Implementation Review

## Conclusion

The metric implementations pass the existing tests, but some behaviors do not match the standard or official benchmark definitions.

No implementation changes were made as part of this review.

## Important findings

### 1. Prediction metrics ignore validity masks and time offsets

[`t4perceval/system/metric/prediction.py`](../../t4perceval/system/metric/prediction.py) does not require or use `MODE_VALID`, `TIMESTEP_VALID`, or `TIME_OFFSET`.

Consequently:

- Padded invalid modes and timesteps can contribute to ADE, FDE, and miss rate.
- Trajectories with different temporal intervals are compared by array index alone.
- An invalid mode can be selected among the top-k modes.

This conflicts with the trajectory archetype documentation, which states that padded elements are masked.

### 2. APH differs from the Waymo definition

[`t4perceval/system/metric/detection.py`](../../t4perceval/system/metric/detection.py) treats heading similarity as a fractional true-positive count and uses that value for both recall and precision.

In the Waymo APH definition:

- Recall is calculated from the ordinary true-positive count.
- Heading-aware precision uses the sum of heading similarities divided by the number of true positives and false positives.

The current implementation therefore also changes the denominator and treats a true positive with zero heading similarity as a false positive.

This behavior is compatible with the corresponding implementation in `autoware_perception_evaluation`, but it is not compatible with the [official Waymo implementation](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/metrics/detection_metrics.cc).

### 3. AP matching is not performed in confidence order

The current pipeline first fixes all associations using a globally optimal Hungarian assignment in [`t4perceval/system/matching.py`](../../t4perceval/system/matching.py), then ranks the resulting match verdicts by confidence in [`t4perceval/system/metric/detection.py`](../../t4perceval/system/metric/detection.py).

nuScenes processes predictions in confidence order and matches each prediction to the closest ground truth that is still available. See the [official nuScenes implementation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py).

For example, consider one ground truth with two predictions inside the matching threshold:

- A high-confidence prediction that is slightly farther away.
- A low-confidence prediction that is very close.

The current Hungarian assignment can make the low-confidence prediction the true positive, whereas nuScenes makes the high-confidence prediction the true positive. The resulting AP values differ.

## CLEAR tracking findings

### 4. An identity change across a missing frame is counted as an ID switch

[`ClearSystem._count_switches()`](../../t4perceval/system/metric/tracking.py) only receives rows that were counted as true positives. Frames where the object was missed disappear before switch counting.

For example:

```text
time:        0  1  2
GT 100:      1  -  2
```

Calling `_count_switches()` with times `[0, 2]`, ground-truth IDs `[100, 100]`, and estimation IDs `[1, 2]` returns `1`.

This conflicts with the method documentation, which says that only the immediately preceding frame is compared.

### 5. MOTA is clamped to zero

[`t4perceval/system/metric/tracking.py`](../../t4perceval/system/metric/tracking.py) applies `max(0.0, ...)` to MOTA. Standard CLEAR MOTA permits negative values.

This is compatible with the original Autoware implementation, but results can differ from other CLEAR metric implementations.

Additionally, when class-agnostic matching is enabled, a matched pair with different ground-truth and estimation classes is not counted as a true positive, but the estimation is not counted as a false positive for its predicted class either.

## Definition differences

### Classification accuracy

The accuracy reported by [`t4perceval/system/metric/classification.py`](../../t4perceval/system/metric/classification.py) is:

```text
TP / (TP + FP + FN)
```

This is Jaccard/IoU rather than conventional classification accuracy. It can be reasonable when true negatives cannot be defined, but the name `accuracy` may be misleading.

### Prediction miss rate

The current prediction miss rate is the fraction of all mode/timestep distances that exceed the tolerance.

A common forecasting definition instead reports the fraction of objects whose selected or best trajectory has a final displacement above the threshold. For an example, see the [official Argoverse implementation](https://github.com/argoverse/argoverse-api/blob/master/argoverse/evaluation/eval_forecasting.py).

## Test results

The following test runs succeeded:

```text
Metric tests: 152 passed
Full suite:    924 passed
```

The full suite was run with pytest plugin auto-loading disabled to avoid unrelated ROS pytest plugins:

```bash
UV_CACHE_DIR=/tmp/t4perceval-uv-cache \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
uv run pytest -q
```

The current tests adequately verify the behavior implemented by the project, but they do not cover official benchmark compatibility, validity masks, time alignment, or an identity change across a missing frame.
