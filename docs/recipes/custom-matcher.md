# Implement a custom matcher

**Goal:** pair estimations with ground truth on a score the six built-in modes do not compute.

## The contract

Subclass `MatchingSystem` and supply five things:

| Thing               | What it is                                         |
| :------------------ | :------------------------------------------------- |
| `REQUIRES`          | the descriptors **both** entities must carry       |
| `MATCHING_NAME`     | the last path segment, `/matching/<MATCHING_NAME>` |
| `HIGHER_IS_BETTER`  | `True` for an overlap, `False` for a distance      |
| `DEFAULT_THRESHOLD` | used when the caller gives none                    |
| `score_matrix()`    | an `(len(est_view), len(gt_view))` array of scores |

The base class handles the range queries, the frame guard, the per-class threshold resolution, the
one-to-one assignment, the TP/FP/FN bookkeeping and writing the chunk. **You write one function that
scores every pair.**

## Example: distance weighted by size agreement

```python
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field

from t4perceval.descriptors import CLASS_ID, POSITION, SIZE
from t4perceval.system import MatchingSystem

if TYPE_CHECKING:
    from t4perceval.core.descriptor import ComponentDescriptor
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayF64


@define(slots=True)
class SizeAwareDistanceMatchingSystem(MatchingSystem):
    """Centre distance, inflated when the two boxes disagree about size."""

    REQUIRES: ClassVar[tuple[ComponentDescriptor, ...]] = (POSITION, SIZE, CLASS_ID)
    MATCHING_NAME: ClassVar[str] = "size_aware_distance"
    HIGHER_IS_BETTER: ClassVar[bool] = False
    DEFAULT_THRESHOLD: ClassVar[float] = 2.0

    size_weight: float = field(default=1.0, kw_only=True)

    def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
        est_position = est_view.component(POSITION).values  # (N, 3)
        gt_position = gt_view.component(POSITION).values  # (M, 3)
        est_size = est_view.component(SIZE).values
        gt_size = gt_view.component(SIZE).values

        distance = np.linalg.norm(est_position[:, None, :] - gt_position[None, :, :], axis=-1)
        size_error = np.abs(est_size[:, None, :] - gt_size[None, :, :]).sum(axis=-1)
        return distance + self.size_weight * size_error
```

Use it like any other:

```python
from t4perceval.system import Thresholds

matcher = SizeAwareDistanceMatchingSystem.between(
    "/estimation/objects",
    "/ground_truth/objects",
    threshold=Thresholds(3.0, by_class=(("pedestrian", 1.0),)),
    size_weight=0.5,
)
matcher.target  # /matching/size_aware_distance
```

It also drops straight into the AP sweep, because that takes a matcher **class**:

```python
from t4perceval.system import average_precision_sweep

average_precision_sweep(EST, GT, matcher=SizeAwareDistanceMatchingSystem, thresholds=[2.0, 4.0])
```

## Rules to follow

- **Return `(len(est_view), len(gt_view))`.** Broadcast; do not loop. The two views are whole
  frames, and a Python loop here is what the columnar model exists to avoid.
- **Get `HIGHER_IS_BETTER` right.** It decides both the direction of the threshold and the sign the
  assignment minimises. Getting it backwards produces a matcher that reliably pairs the worst
  candidates.
- **Declare everything you read in `REQUIRES`.** The base calls `require()` on both views, so a
  missing column becomes a clear error instead of a `None`.
- **Do not implement the assignment.** The base solves a globally optimal one-to-one assignment per
  frame, so a good pair is not lost to a greedy earlier choice, and it handles infeasible pairs,
  class separation and the threshold.
- **Handle the empty case implicitly.** A zero-row view means a zero-size matrix; NumPy broadcasting
  gives you that for free.

## Reuse the vectorized geometry

Before writing your own, check [`t4perceval.geometry`](../reference/api/geometry.md) -- every
pairwise helper already returns the `(N, M)` shape you need:

```python
from t4perceval.geometry import (
    pairwise_bev_iou,
    pairwise_plane_distance,
    pairwise_roi_iou,
    pairwise_volume_iou,
)
```

## Class-agnostic matching

`class_agnostic` is a base-class parameter; you do not implement it. `score_matrix` scores every
pair regardless, and the base masks out cross-class pairs unless the caller opted in.

## Coordinate frames

Also handled by the base: two _different, stated_ frames raise before `score_matrix` is called.
`check_frames=False` opts out.

## Where to go next

- [Matching](../user-guide/matching.md) -- the built-in family.
- [Write a custom metric](custom-metric.md) -- scoring the verdicts you produce.
