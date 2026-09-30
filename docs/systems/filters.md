# Filters

Every filter tests each row of its source entity and writes a `mask` column to
`<source>/filter/<name>` -- the rows stay where they are. All of them share `MaskSystem`, and all
but one build with `.on(source, **params)`.

Bounds are inclusive on both ends, so a filter with its default parameters passes every row. The
sections below follow the modules of `t4perceval.system.filter`, each named for the column it
judges.

## Position

`t4perceval.system.filter.position` -- where an object is. All three require `position`.

| System                   | Parameters (default)                                    | Keeps rows whose                                    |
| :----------------------- | :------------------------------------------------------ | :-------------------------------------------------- |
| `FilterByDistanceSystem` | `min_distance` (0), `max_distance` (inf), `bev` (False) | distance from the origin is in range; 3D or xy only |
| `FilterByRegionSystem`   | `min_xy` (-inf), `max_xy` (inf)                         | xy lies inside an axis-aligned box                  |
| `FilterByMapSystem`      | `polygon` (None), `resolver` (None)                     | xy lies inside a polygon stated in `map`            |

Distance and region are measured **in the frame the source declares**; a distance from the ego
needs the rows in `base_link`. `FilterByRegionSystem.symmetric(source, max_xy=)` mirrors the box
about the origin. The map filter is the one filter that looks the ego pose up itself, so a
`base_link` source needs no transform first; `FilterByMapSystem.on_lanelet(source, lanelet_map,
subtypes=)` builds its polygon from a [Lanelet2 map](../reference/api/lanelet.md).

## Identity

`t4perceval.system.filter.identity` -- what an object is. Names are resolved through the
context's registries, and an unknown name raises.

| System                   | Requires      | Parameters (default)                 | Keeps rows whose                    |
| :----------------------- | :------------ | :----------------------------------- | :---------------------------------- |
| `FilterByLabelSystem`    | `class_id`    | `labels` (None), `exclude` (None)    | class is listed and not excluded    |
| `FilterByInstanceSystem` | `instance_id` | `instances` (None), `exclude` (None) | instance is listed and not excluded |

## Quality

`t4perceval.system.filter.quality` -- how well an object was observed.

| System                     | Requires     | Parameters (default)                          | Keeps rows whose                                 |
| :------------------------- | :----------- | :-------------------------------------------- | :----------------------------------------------- |
| `FilterByConfidenceSystem` | `confidence` | `min_confidence` (0), `max_confidence` (1)    | confidence is in range                           |
| `FilterBySpeedSystem`      | `velocity`   | `min_speed` (0), `max_speed` (inf)            | speed, the L2 norm of `velocity`, is in range    |
| `FilterByNumPointsSystem`  | `num_points` | `min_num_points` (0), `max_num_points` (None) | point count is in range                          |
| `FilterByVisibilitySystem` | `visibility` | `min_visibility` (`NONE`)                     | at least as visible; `UNAVAILABLE` always passes |

## Point

`t4perceval.system.filter.point` -- points of a cloud rather than objects.

| System                         | Requires                | Parameters (default)                                    | Keeps rows whose                                    |
| :----------------------------- | :---------------------- | :------------------------------------------------------ | :-------------------------------------------------- |
| `FilterByCoverageSystem`       | `point` on both sources | `tolerance` (1e-6), `check_frames` (True)               | point has a reference point within `tolerance`      |
| `FilterPointsByDistanceSystem` | `point`                 | `min_distance` (0), `max_distance` (inf), `bev` (False) | distance from the origin is in range; 3D or xy only |
| `FilterPointsByRegionSystem`   | `point`                 | `min_xy` (-inf), `max_xy` (inf)                         | xy lies inside an axis-aligned box                  |
| `FilterPointsByMapSystem`      | `point`                 | `polygon` (None), `resolver` (None)                     | xy lies inside a polygon stated in `map`            |

`FilterByCoverageSystem` is the one filter with two sources: `.between(source, reference)` masks
`source` by whether `reference` covers it, frame by frame, and writes to `<source>/filter/coverage`.
It is how a cropped or downsampled estimation leaves the uncovered ground truth out of a
segmentation score.

The three `FilterPointsBy*` systems are the [position](#position) predicates over the `point`
column, with the same parameters, the same inclusive bounds and the same targets
(`<source>/filter/distance`, `/region`, `/map`); `symmetric()` and `on_lanelet()` work the same.
They are separate classes rather than the object filters pointed at a cloud because
[`POINT` is not `POSITION`](../archetypes/segmentation.md#point-not-position): a filter has to say
which of the two it judges, and each refuses the other.

## Mask

`t4perceval.system.filter.mask` -- composing masks, and materializing the rows one kept.

| System               | Requires       | Parameters (default) | Writes                                          |
| :------------------- | :------------- | :------------------- | :---------------------------------------------- |
| `CombineMasksSystem` | `mask` on each | `mode` (`"all"`)     | one `mask`: AND (`"all"`) or OR (`"any"`)       |
| `ApplyMaskSystem`    | `mask`         | --                   | the rows the mask kept, as a copy of the source |

`CombineMasksSystem.of(masks, target)` reads several mask entities that describe the same rows.
`ApplyMaskSystem.of(source, mask)` writes to `<source>/kept` by default; it is a passthrough, so a
matcher can read its target in the same pipeline.

## Where to go next

- [Filtering](../user-guide/filtering.md) -- using them, and reading a mask back.
- [Write a custom filter](../recipes/custom-filter.md).
