# Align frames by timestamp

**Goal:** two recordings produced independently, whose `FRAME` axes do not correspond.

## The problem

A T4 scene numbers its frames by sample; a bag numbers them by message. Two recordings imported
separately therefore share only their `TIMESTAMP` axis.

Matching evaluates the **union** of both `FRAME` sets, so unrelated indices do not fail -- they
score as all-FP and all-FN frames. You get plausible numbers, not an error. That is exactly the
failure mode this step exists to prevent.

```text
ground truth   frame:  0     1     2     3
               stamp: 100   200   300   400   (ms)

estimation     frame:  0   1   2   3   4   5
               stamp: 95  145 205 255 305 355

               ── align (tolerance 75 ms) ──▶

estimation     frame:  0    -    1    -    2    -
                       ↑         ↑         ↑
                    paired    paired    paired
```

## The one-liner

```python
from t4perceval.align import AlignOptions
from t4perceval.evaluation import build_evaluation_store

setup = build_evaluation_store(
    ground_truth,
    estimation,
    align=AlignOptions(tolerance_ns=75_000_000),  # 75 ms, the incumbent's default
)
```

Every ground-truth frame takes the nearest estimation frame within the tolerance, **one-to-one**,
and the estimation's `FRAME` values are rewritten to the ground truth's.

## Checking it worked

```python
setup.metadata.tags
# (('align.pairs', '87'), ('align.unmatched_reference', '3'),
#  ('align.unmatched_query', '12'), ('align.tolerance_ns', '75000000'), ...)
```

Look at `align.pairs` against the number of ground-truth frames. A low count means the tolerance is
too tight, or the clocks are offset.

## Options

```python
AlignOptions(
    tolerance_ns=75_000_000,  # how far apart two frames may be and still pair
    offset_ns=0,  # a known clock skew, added to the query's times
    unmatched_reference="keep",  # or "drop"
)
```

| Situation                                         | Default (`"keep"`)              | `"drop"`                       |
| :------------------------------------------------ | :------------------------------ | :----------------------------- |
| A ground-truth frame no estimation answered       | stays, and scores as all-FN     | removed from the evaluated set |
| An estimation frame no ground-truth frame claimed | always leaves the evaluated set | same                           |

`"drop"` is what `autoware_perception_evaluation` does. `"keep"` is the default here because a frame
the estimator missed entirely is a real failure, and silently discarding it flatters the result.

### Correcting a known skew

```python
AlignOptions(tolerance_ns=75_000_000, offset_ns=-12_000_000)  # the bag runs 12 ms late
```

`offset_ns` is added to the query's timestamps before pairing, so residual error after the
correction is what the tolerance has to absorb.

## The steps, separately

`align_recordings` is the whole thing; its three parts are public if you want to inspect or
intervene between them.

```python
from t4perceval.align import align_frames, align_recordings, drop_frames, reindex_frames

# 1. Which query frame answers which reference frame?
alignment = align_frames(
    ground_truth,
    estimation,
    reference_path="/ground_truth/objects",
    query_path="/estimation/objects",
    tolerance_ns=75_000_000,
)

alignment.num_pairs
alignment.deltas_ns()  # query_times - reference_times, per pair
alignment.mapping()  # query frame -> reference frame
alignment.unmatched_reference_frames
alignment.unmatched_query_frames
alignment.describe()  # the string pairs that become metadata tags

# 2. Rewrite the query's FRAME values onto the reference's.
estimation = reindex_frames(estimation, alignment)

# 3. Optionally remove the reference frames nothing answered.
ground_truth = drop_frames(ground_truth, alignment.unmatched_reference_frames)
```

`align_recordings` composes exactly these:

```python
aligned = align_recordings(ground_truth, estimation, options=AlignOptions(tolerance_ns=75_000_000))
aligned.reference, aligned.query, aligned.alignment
```

## Diagnosing a bad alignment

```python
import numpy as np

deltas = alignment.deltas_ns()
np.median(deltas) / 1e6  # ms -- a large constant is a clock offset; feed it to offset_ns
np.std(deltas) / 1e6  # ms -- large spread means jitter, so widen the tolerance
```

If `num_pairs` is zero, the two recordings' timestamps probably do not overlap at all: check
`frame_times(recording, entity_path)` on each side.

```python
from t4perceval.align import frame_times

frames, times = frame_times(ground_truth, "/ground_truth/objects")
```

## When you do not need this

- Both sides were logged into the **same** store with the same frame indices.
- Both came from the same importer call.

Alignment is only for recordings whose frame numbering was decided independently.

## Where to go next

- [Evaluate an MCAP ROS bag](evaluate-rosbag.md) -- where this most often comes up.
- [Timeline](../concepts/timeline.md) -- `FRAME` versus `TIMESTAMP`.
