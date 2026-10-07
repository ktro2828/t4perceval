# Metrics

A metric turns match verdicts into numbers. Every scalar metric writes a
[`MetricValues`](../archetypes/metrics.md) chunk -- `class_id`, `threshold`, `value`, `support` --
to `/metrics/<name>`; the two confusion matrices write a `ConfusionMatrix` chunk instead.

## Over a matching entity

These build with `.on(matching, estimation, ground_truth, **params)` and share `MetricSystem`.
Each requires `est_index`, `gt_index` and `match_status` on the matching entity, plus the columns
below on the two object entities.

| System                          | Estimation needs                             | Ground truth needs                           | Parameters (default)                                                 | Writes                                                   |
| :------------------------------ | :------------------------------------------- | :------------------------------------------- | :------------------------------------------------------------------- | :------------------------------------------------------- |
| `AveragePrecisionSystem`        | `class_id`, `confidence`                     | `class_id`                                   | `min_recall` (0.1), `min_precision` (0.1), `num_recall_points` (101) | AP per class at `/metrics/ap`                            |
| `AveragePrecisionHeadingSystem` | `class_id`, `confidence`, `quaternion`       | `class_id`, `quaternion`                     | as above                                                             | APH per class at `/metrics/aph`                          |
| `ClearSystem`                   | `class_id`, `instance_id`                    | `class_id`, `instance_id`                    | --                                                                   | `/metrics/clear/{mota,motp,id_switch}`                   |
| `PathDisplacementSystem`        | `class_id`, `waypoints`, `mode_confidence`   | `class_id`, `waypoints`                      | `top_k` (3), `miss_tolerance` (2.0), `kernel` (None)                 | `/metrics/displacement/{ade,fde,miss_rate}`              |
| `CornerErrorSystem`             | `class_id`, `position`, `quaternion`, `size` | `class_id`, `position`, `quaternion`, `size` | `percentiles` ((95.0,))                                              | `/metrics/corner_error/{mean,max,p95}`                   |
| `HeadingFlipRateSystem`         | `class_id`, `quaternion`                     | `class_id`, `quaternion`                     | `flip_threshold` (pi/2)                                              | flip rate per class at `/metrics/heading_flip_rate`      |
| `ClassificationSystem`          | `class_id`                                   | `class_id`                                   | --                                                                   | `/metrics/classification/{accuracy,precision,recall,f1}` |
| `ConfusionMatrixSystem`         | `class_id`                                   | `class_id`                                   | --                                                                   | `ConfusionMatrix` at `/metrics/confusion_matrix`         |

`MeanAveragePrecisionSystem.of(sources, target="/metrics/map")` is the one metric that reads other
metrics: it averages the `MetricValues` of several AP entities across thresholds and classes.
`average_precision_sweep(...)` builds the matchers, the AP systems and this mean in one call.

A system with several results writes one entity per result, so `ClearSystem` yields three metric
entities. Cross-frame metrics (CLEAR, displacement) read the whole `TimeRange` they are given; the
others work per frame or per scene alike.

## Over aligned label columns

Segmentation has no matching stage: the two entities are compared element by element. These build
with `.between(estimation, ground_truth, **params)` and share `SegmentationMetricSystem`. Either
label column satisfies them, which a `REQUIRES` tuple cannot say, so `REQUIRES` is empty and a
missing label column is reported per frame at run time.

| System                              | Requires                       | Parameters (default)                                           | Writes                                                        |
| :---------------------------------- | :----------------------------- | :------------------------------------------------------------- | :------------------------------------------------------------ |
| `SegmentationIoUSystem`             | `class_id` or `class_id_image` | `ignore` (()), `check_frames` (True), `point_tolerance` (1e-6) | `/metrics/segmentation/{iou,accuracy,pixel_accuracy}`         |
| `SegmentationConfusionMatrixSystem` | `class_id` or `class_id_image` | as above                                                       | `ConfusionMatrix` at `/metrics/segmentation/confusion_matrix` |

When both entities carry `point`, the coordinates are compared row by row within
`point_tolerance`, so a reordered cloud is refused rather than scored against the wrong points;
`AlignPointsSystem` puts the rows in order first.

## Where to go next

- [Metrics](../user-guide/metrics.md) -- reading values back, per class and aggregate.
- [Metric divergences](../development/metric-divergences.md) -- where the definitions differ from
  `autoware_perception_evaluation`.
- [Write a custom metric](../recipes/custom-metric.md).
