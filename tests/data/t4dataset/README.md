# Minimal T4 dataset fixture

Annotation tables for a synthetic single-scene T4 dataset, used to test
`t4perceval.importer.t4` against the real on-disk format rather than against
hand-built objects.

## Provenance

Copied from [`tier4/t4-devkit`](https://github.com/tier4/t4-devkit),
`tests/sample/t4dataset/annotation/`, at commit `a36f6967` (`v0.8.0`).
Both projects are licensed under Apache-2.0.

The upstream `status.json` records that the data is synthetic: *"All data files are
placeholders and do not contain real sensor data."* There is no real sensor data, no
personal data, and no map-licensing exposure here.

## What was and was not copied

Only `annotation/*.json` is vendored. The importer resolves sensor paths but never opens
them — `T4Devkit.__init__` reads only the annotation tables — so upstream's `data/`
(placeholder JPEG/PCD) and `map/` (a `.osm` that is never parsed) are omitted.

The `1/` directory is a version directory. `t4_devkit.load_metadata` matches
subdirectories against `r".*/\d+$"` and raises a `DeprecationWarning` when it finds none,
so this layout both mirrors a real T4 release and keeps the test run warning-free.

## Why this fixture

For 84 KB it covers, with no editing, the cases that would otherwise need hand-built
input:

- a three-sample `next` chain whose **last frame has zero 3D and zero 2D annotations**
- **mixed finite and NaN velocity within one frame** (the car is estimable, the
  pedestrian is not)
- a `future` trajectory on **exactly one** box in the scene, so mode/timestep padding and
  the validity masks are exercised for real
- a **non-identity quaternion** (30° yaw), so an `xyzw` / `wxyz` swap changes the result
- 2D annotations on `CAM_FRONT` only — simultaneously the empty-camera case and the
  `get_box2ds` channel-leak case
- four categories with `index` populated

Cases the fixture cannot express — non-monotonic future timestamps, categories absent
from a caller's registry, multiple scenes — are built on the fly by
`tests/t4_builder.py`.
