# Matchers

A matcher pairs the rows of an estimation entity with the rows of a ground-truth entity, frame by
frame, and writes a [`MatchResults`](../archetypes/matching.md) chunk to `/matching/<mode>`. All
six build with `.between(estimation, ground_truth, **params)` and share `MatchingSystem`.

| System                            | Requires                                     | Score                       | Better | Default threshold |
| :-------------------------------- | :------------------------------------------- | :-------------------------- | :----- | ----------------: |
| `CenterDistanceMatchingSystem`    | `position`, `class_id`                       | 3D distance between centres | lower  |               1.0 |
| `CenterDistanceBEVMatchingSystem` | `position`, `class_id`                       | xy distance between centres | lower  |               1.0 |
| `PlaneDistanceMatchingSystem`     | `position`, `quaternion`, `size`, `class_id` | RMS gap of the facing faces | lower  |               2.0 |
| `IoUBEVMatchingSystem`            | `position`, `quaternion`, `size`, `class_id` | IoU of footprints           | higher |               0.5 |
| `IoU3DMatchingSystem`             | `position`, `quaternion`, `size`, `class_id` | IoU of volumes              | higher |               0.5 |
| `IoURoiMatchingSystem`            | `roi`, `class_id`                            | IoU of image-plane regions  | higher |               0.5 |

Shared parameters:

| Parameter                | Default  | Meaning                                                                                    |
| :----------------------- | :------- | :----------------------------------------------------------------------------------------- |
| `threshold`              | per mode | a number, or `Thresholds(default, by_class=...)` keyed by ground-truth class               |
| `class_agnostic`         | `False`  | pair rows whose classes differ; the verdict then records the disagreement                  |
| `max_matchable_distance` | `None`   | largest 3D centre distance (m) a pair may have, on top of `threshold`; not for `IoURoi...` |
| `check_frames`           | `True`   | refuse two sources that declare different coordinate frames                                |

Pairs are chosen by a globally optimal assignment over the score matrix; a score past the
threshold, or a non-finite one, is never assigned. Each mode has its own default target, so several
can run over the same frame and be compared afterwards.

## Where to go next

- [Matching](../user-guide/matching.md) -- thresholds, sweeps, reading verdicts back.
- [Write a custom matcher](../recipes/custom-matcher.md).
