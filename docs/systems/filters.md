# Filters

Every filter reads **one** entity, tests each row, and writes a `mask` column to
`<source>/filter/<name>` -- the rows stay where they are. All of them build with
`.on(source, **params)` and share `MaskSystem`.

Bounds are inclusive on both ends, so a filter with its default parameters passes every row.

| System                     | Requires      | Parameters (default)                                    | Keeps rows whose                                    |
| :------------------------- | :------------ | :------------------------------------------------------ | :-------------------------------------------------- |
| `FilterByDistanceSystem`   | `position`    | `min_distance` (0), `max_distance` (inf), `bev` (False) | distance from the origin is in range; 3D or xy only |
| `FilterByRegionSystem`     | `position`    | `min_xy` (-inf), `max_xy` (inf)                         | xy lies inside an axis-aligned box                  |
| `FilterByMapSystem`        | `position`    | `polygon` (None), `resolver` (None)                     | xy lies inside a polygon stated in `map`            |
| `FilterByLabelSystem`      | `class_id`    | `labels` (None), `exclude` (None)                       | class is listed and not excluded                    |
| `FilterByConfidenceSystem` | `confidence`  | `min_confidence` (0), `max_confidence` (1)              | confidence is in range                              |
| `FilterByInstanceSystem`   | `instance_id` | `instances` (None), `exclude` (None)                    | instance is listed and not excluded                 |
| `FilterBySpeedSystem`      | `velocity`    | `min_speed` (0), `max_speed` (inf)                      | speed, the L2 norm of `velocity`, is in range       |
| `FilterByNumPointsSystem`  | `num_points`  | `min_num_points` (0), `max_num_points` (None)           | point count is in range                             |
| `FilterByVisibilitySystem` | `visibility`  | `min_visibility` (`NONE`)                               | at least as visible; `UNAVAILABLE` always passes    |

Two of them have extra constructors. `FilterByRegionSystem.symmetric(source, max_xy=)` mirrors the
box about the origin. `FilterByMapSystem.on_lanelet(source, lanelet_map, subtypes=)` builds the
polygon from a [Lanelet2 map](../reference/api/lanelet.md); the map filter is the one filter that
looks the ego pose up itself, so a `base_link` source needs no transform first.

Distance, region and speed are measured **in the frame the source declares**. A distance from the
ego needs the rows in `base_link`.

## Composing and applying masks

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
