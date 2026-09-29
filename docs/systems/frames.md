# Frames and points

Two systems rewrite a source's rows rather than judging them. Each is a **passthrough**: its
target carries the source's columns, so a filter, matcher or metric can read it in the same
pipeline.

| System                  | Requires | Parameters (default)                      | Writes                                                                          |
| :---------------------- | :------- | :---------------------------------------- | :------------------------------------------------------------------------------ |
| `TransformEntitySystem` | --       | `target_frame`, `resolver` (None)         | the source expressed in `target_frame`, at `<source>/in/<frame>`; masks dropped |
| `AlignPointsSystem`     | `point`  | `tolerance` (1e-6), `check_frames` (True) | the estimation cloud in the ground truth's row order, at `<source>/aligned`     |

`TransformEntitySystem.of(source, target_frame=)` looks each frame's pose up through a
`TransformResolver` -- built over the store by default, or passed in when the `/tf` edges live in a
recording or on another timeline. Positions, waypoints and segmentation points are moved, velocity
is rotated only, orientations are composed, and every other column is carried unchanged.

`AlignPointsSystem.between(estimation, ground_truth)` prepares two point clouds for the row-wise
segmentation metrics; [`FilterByCoverageSystem`](filters.md#point) first reports which rows can
be compared at all.

## Where to go next

- [Transforms](../user-guide/transforms.md) -- resolvers, lookup policies, the cross-frame guard.
- [Segmentation](../evaluation/segmentation-3d.md) -- the point clouds end to end.
