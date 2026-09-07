"""Synthetic scenes, as plain NumPy arrays both libraries are built from.

A scene is written to ``.npz`` by the coordinator and loaded by each worker, so "the same
input" is a file identity rather than trust that two NumPy majors draw identical random
streams.

Two regimes:

* ``unambiguous`` -- ground truths sit on a jittered grid whose spacing exceeds twice the
  largest matching threshold plus the largest estimation offset, and every estimate lies
  within that offset of its own ground truth. At every threshold the feasible graph has
  degree at most one, so greedy assignment in any order and optimal assignment pick the
  same pairs. This is the regime in which the two libraries are expected to agree exactly.
* ``dense`` -- random positions, per-frame noise, label confusion, missed frames. Used for
  the performance tables and for reporting the divergences the libraries are known to have.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np

CLASS_NAMES = ("car", "truck", "bus", "bicycle", "pedestrian", "motorbike")
"""Valid ``AutowareLabel`` values, and the registry names on the t4perceval side."""

CLASS_SIZES = {
    "car": (1.9, 4.5, 1.6),
    "truck": (2.5, 8.0, 3.0),
    "bus": (2.9, 11.0, 3.3),
    "bicycle": (0.6, 1.8, 1.4),
    "pedestrian": (0.6, 0.6, 1.7),
    "motorbike": (0.8, 2.0, 1.4),
}
"""``(width, length, height)`` per class, the order both libraries use."""

DETECTION_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
MATCHING_THRESHOLD = 1.0
TOP_KS = (1, 3)
MISS_TOLERANCE = 2.0

V_MAX = 2.0
GRID_JITTER = 0.5

Regime = Literal["unambiguous", "dense"]


@dataclass(frozen=True)
class SceneParams:
    regime: Regime
    num_frames: int
    num_objects: int
    num_classes: int = len(CLASS_NAMES)
    num_modes: int = 3
    num_timesteps: int = 6
    dt_s: float = 0.5
    recall: float = 0.9
    fp_rate: float = 0.15
    confusion_rate: float = 0.0
    position_noise_max: float = 0.8
    yaw_noise_max: float = 0.2
    num_id_switches: int | None = None
    seed: int = 0
    check: bool = True
    """Whether to assert the unambiguity spacing (O(N^2); skipped for huge scenes)."""


@dataclass(frozen=True)
class Frame:
    """One frame of one side, as arrays."""

    position: np.ndarray
    yaw: np.ndarray
    size: np.ndarray
    class_id: np.ndarray
    instance: np.ndarray
    velocity: np.ndarray
    confidence: np.ndarray
    waypoints: np.ndarray
    """``(n, M, T, 3)``; ``M == 1`` for ground truth."""

    mode_confidence: np.ndarray

    def __len__(self) -> int:
        return len(self.position)


@dataclass
class Scene:
    params: SceneParams
    class_names: tuple[str, ...]
    time_offset_ns: np.ndarray
    gt_offsets: np.ndarray
    gt_position: np.ndarray
    gt_yaw: np.ndarray
    gt_size: np.ndarray
    gt_class: np.ndarray
    gt_instance: np.ndarray
    gt_velocity: np.ndarray
    gt_waypoints: np.ndarray
    est_offsets: np.ndarray
    est_position: np.ndarray
    est_yaw: np.ndarray
    est_size: np.ndarray
    est_class: np.ndarray
    est_instance: np.ndarray
    est_velocity: np.ndarray
    est_confidence: np.ndarray
    est_waypoints: np.ndarray
    est_mode_confidence: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.gt_offsets) - 1

    def gt(self, frame: int) -> Frame:
        lo, hi = self.gt_offsets[frame], self.gt_offsets[frame + 1]
        count = hi - lo
        return Frame(
            self.gt_position[lo:hi],
            self.gt_yaw[lo:hi],
            self.gt_size[lo:hi],
            self.gt_class[lo:hi],
            self.gt_instance[lo:hi],
            self.gt_velocity[lo:hi],
            np.ones(count, dtype=np.float64),
            self.gt_waypoints[lo:hi],
            np.ones((count, 1), dtype=np.float64),
        )

    def est(self, frame: int) -> Frame:
        lo, hi = self.est_offsets[frame], self.est_offsets[frame + 1]
        return Frame(
            self.est_position[lo:hi],
            self.est_yaw[lo:hi],
            self.est_size[lo:hi],
            self.est_class[lo:hi],
            self.est_instance[lo:hi],
            self.est_velocity[lo:hi],
            self.est_confidence[lo:hi],
            self.est_waypoints[lo:hi],
            self.est_mode_confidence[lo:hi],
        )

    def save(self, path: Path) -> Path:
        arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        np.savez(
            path,
            params_json=np.array(json.dumps(asdict(self.params))),
            class_names=np.array(self.class_names),
            **arrays,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> Scene:
        with np.load(path, allow_pickle=False) as data:
            params = SceneParams(**json.loads(str(data["params_json"])))
            class_names = tuple(str(name) for name in data["class_names"])
            arrays = {name: data[name] for name in _ARRAY_FIELDS}
        return cls(params, class_names, **arrays)


_ARRAY_FIELDS = tuple(f.name for f in fields(Scene) if f.name not in ("params", "class_names"))


def yaw_to_quaternion_xyzw(yaw: np.ndarray) -> np.ndarray:
    """Rotation about z as ``(x, y, z, w)``."""
    half = np.asarray(yaw, dtype=np.float64) / 2.0
    out = np.zeros((len(half), 4), dtype=np.float64)
    out[:, 2] = np.sin(half)
    out[:, 3] = np.cos(half)
    return out


def unambiguous_spacing(params: SceneParams) -> float:
    """Grid spacing that keeps every pair of ground truths apart at every frame."""
    travel = 2.0 * (params.num_frames - 1) * params.dt_s * V_MAX
    return (
        2.0 * (max(DETECTION_THRESHOLDS) + params.position_noise_max)
        + travel
        + 2 * GRID_JITTER
        + 1.0
    )


def make_scene(params: SceneParams) -> Scene:
    rng = np.random.default_rng(params.seed)
    names = CLASS_NAMES[: params.num_classes]
    count, frames = params.num_objects, params.num_frames
    modes, steps, dt = params.num_modes, params.num_timesteps, params.dt_s
    sizes = np.asarray([CLASS_SIZES[name] for name in names], dtype=np.float64)

    # -- instances --------------------------------------------------------------------
    # Round-robin classes so every class has ground truth whenever N >= K.
    classes = rng.permutation(np.arange(count) % len(names)).astype(np.int32)
    heading = rng.uniform(-np.pi, np.pi, count)
    speed = rng.uniform(0.0, V_MAX, count)
    velocity = np.stack([speed * np.cos(heading), speed * np.sin(heading), np.zeros(count)], axis=1)

    if params.regime == "unambiguous":
        spacing = unambiguous_spacing(params)
        cols = max(1, math.ceil(math.sqrt(count)))
        index = np.arange(count)
        base = np.stack(
            [
                (index % cols) * spacing + rng.uniform(-GRID_JITTER, GRID_JITTER, count),
                (index // cols) * spacing + rng.uniform(-GRID_JITTER, GRID_JITTER, count),
                np.zeros(count),
            ],
            axis=1,
        )
        yaw = rng.uniform(0.3, np.pi - 0.3, count)
        # A per-instance constant offset: the same distance every frame, so MOTP does not
        # depend on whether a library reads the current or the previous frame's score.
        direction = rng.uniform(-np.pi, np.pi, count)
        norm = rng.uniform(0.0, params.position_noise_max, count)
        offset = np.stack([norm * np.cos(direction), norm * np.sin(direction), np.zeros(count)], 1)
        yaw_noise = rng.uniform(-params.yaw_noise_max, params.yaw_noise_max, count)
        box = cols * spacing
    else:
        box = 3.0 * math.sqrt(count)
        base = np.concatenate([rng.uniform(0.0, box, (count, 2)), np.zeros((count, 1))], axis=1)
        yaw = rng.uniform(-np.pi, np.pi, count)
        offset = None
        yaw_noise = None

    # -- identities and switches ------------------------------------------------------
    est_ids = np.tile(1000 + np.arange(count, dtype=np.int64), (frames, 1))
    switch_frames = np.full(count, frames + 1, dtype=np.int64)
    if frames >= 3:
        num_switches = params.num_id_switches
        if num_switches is None:
            num_switches = max(1, count // 10)
        for offset_id, instance in enumerate(
            rng.choice(count, min(num_switches, count), replace=False)
        ):
            frame = int(rng.integers(2, frames))
            switch_frames[instance] = frame
            est_ids[frame:, instance] = 10_000 + offset_id

    keep = rng.random((frames, count)) < params.recall
    if params.regime == "unambiguous":
        for instance in np.flatnonzero(switch_frames <= frames):
            keep[switch_frames[instance] - 1 : switch_frames[instance] + 1, instance] = True
    elif frames >= 2:
        # One frame in which nothing of the first class is estimated: an identity that
        # resumes afterwards is the "switch across a missed frame" case.
        keep[frames // 2, classes == 0] = False

    # -- per-frame arrays -------------------------------------------------------------
    horizon = (np.arange(steps, dtype=np.float64) + 1.0) * dt
    lateral = np.stack([-np.sin(heading), np.cos(heading), np.zeros(count)], axis=1)
    num_fp = int(round(params.fp_rate * count))

    gt_chunks: list[dict[str, np.ndarray]] = []
    est_chunks: list[dict[str, np.ndarray]] = []
    for frame in range(frames):
        position = base + velocity * (frame * dt)
        future = position[:, None, :] + velocity[:, None, :] * horizon[None, :, None]  # (N, T, 3)
        gt_chunks.append(
            {
                "position": position,
                "yaw": yaw,
                "size": sizes[classes],
                "class": classes,
                "instance": np.arange(count, dtype=np.int64),
                "velocity": velocity,
                "waypoints": future[:, None, :, :],
            },
        )

        kept = np.flatnonzero(keep[frame])
        n_kept = len(kept)
        if params.regime == "unambiguous":
            est_position = position[kept] + offset[kept]
            est_yaw = yaw[kept] + yaw_noise[kept]
            est_class = classes[kept]
            mode_future = np.repeat(future[kept][:, None, :, :], modes, axis=1)
            mode_future += offset[kept][:, None, None, :]
            drift = 0.9 * np.arange(modes)[None, :, None] * (horizon / horizon[-1])[None, None, :]
            mode_future += drift[..., None] * lateral[kept][:, None, None, :]
        else:
            noise = np.concatenate([rng.normal(0.0, 0.4, (n_kept, 2)), np.zeros((n_kept, 1))], 1)
            est_position = position[kept] + noise
            est_yaw = yaw[kept] + rng.normal(0.0, 0.3, n_kept)
            est_class = classes[kept].copy()
            confused = rng.random(n_kept) < params.confusion_rate
            est_class[confused] = (
                est_class[confused] + rng.integers(1, len(names), confused.sum())
            ) % len(names)
            mode_future = np.repeat(future[kept][:, None, :, :], modes, axis=1)
            mode_future += (
                rng.normal(0.0, 0.5, mode_future.shape)
                * (horizon / horizon[-1])[None, None, :, None]
            )
            mode_future[..., 2] = future[kept][:, None, :, 2]

        # Clutter: far from every ground truth in the unambiguous regime, anywhere in the
        # dense one.
        if params.regime == "unambiguous":
            fp_position = np.stack(
                [
                    np.arange(num_fp) * unambiguous_spacing(params)
                    + rng.uniform(-GRID_JITTER, GRID_JITTER, num_fp),
                    np.full(num_fp, -3.0 * unambiguous_spacing(params)),
                    np.zeros(num_fp),
                ],
                axis=1,
            )
        else:
            fp_position = np.concatenate(
                [rng.uniform(0.0, box, (num_fp, 2)), np.zeros((num_fp, 1))], 1
            )
        fp_class = rng.integers(0, len(names), num_fp).astype(np.int32)
        fp_yaw = rng.uniform(0.3, np.pi - 0.3, num_fp)
        fp_velocity = np.zeros((num_fp, 3))
        fp_future = np.repeat(fp_position[:, None, None, :], modes, axis=1)
        fp_future = np.repeat(fp_future, steps, axis=2)

        est_chunks.append(
            {
                "position": np.concatenate([est_position, fp_position]),
                "yaw": np.concatenate([est_yaw, fp_yaw]),
                "size": sizes[np.concatenate([est_class, fp_class])],
                "class": np.concatenate([est_class, fp_class]).astype(np.int32),
                "instance": np.concatenate(
                    [
                        est_ids[frame, kept],
                        20_000 + frame * num_fp + np.arange(num_fp, dtype=np.int64),
                    ],
                ),
                "velocity": np.concatenate([velocity[kept], fp_velocity]),
                "confidence": np.concatenate(
                    [rng.uniform(0.4, 1.0, n_kept), rng.uniform(0.0, 0.7, num_fp)],
                ),
                "waypoints": np.concatenate([mode_future, fp_future]),
                "mode_confidence": rng.dirichlet(np.ones(modes), n_kept + num_fp),
            },
        )

    def stack(chunks: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
        return np.concatenate([chunk[key] for chunk in chunks]) if chunks else np.empty(0)

    def offsets(chunks: list[dict[str, np.ndarray]]) -> np.ndarray:
        return np.concatenate(
            [[0], np.cumsum([len(chunk["position"]) for chunk in chunks])]
        ).astype(np.int64)

    scene = Scene(
        params,
        names,
        (horizon * 1e9).astype(np.int64),
        offsets(gt_chunks),
        stack(gt_chunks, "position"),
        stack(gt_chunks, "yaw"),
        stack(gt_chunks, "size"),
        stack(gt_chunks, "class").astype(np.int32),
        stack(gt_chunks, "instance").astype(np.int64),
        stack(gt_chunks, "velocity"),
        stack(gt_chunks, "waypoints"),
        offsets(est_chunks),
        stack(est_chunks, "position"),
        stack(est_chunks, "yaw"),
        stack(est_chunks, "size"),
        stack(est_chunks, "class").astype(np.int32),
        stack(est_chunks, "instance").astype(np.int64),
        stack(est_chunks, "velocity"),
        stack(est_chunks, "confidence"),
        stack(est_chunks, "waypoints"),
        stack(est_chunks, "mode_confidence"),
    )
    _validate(scene)
    return scene


def _validate(scene: Scene) -> None:
    params = scene.params
    # Tie order differs between the libraries, so ties are excluded from the input.
    if np.unique(scene.est_confidence).size != scene.est_confidence.size:
        raise AssertionError("estimation confidences are not unique")
    for row in scene.est_mode_confidence:
        if np.unique(row).size != row.size:
            raise AssertionError("mode confidences within one object are not unique")

    if params.regime != "unambiguous" or not params.check:
        return
    limit = 2.0 * (max(DETECTION_THRESHOLDS) + params.position_noise_max)
    for frame in range(scene.num_frames):
        position = scene.gt(frame).position[:, :2]
        if len(position) < 2:
            continue
        distance = np.linalg.norm(position[:, None, :] - position[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        if distance.min() <= limit:
            raise AssertionError(f"ground truths closer than {limit} m at frame {frame}")
        clutter = scene.est(frame)
        far = clutter.position[clutter.instance >= 20_000][:, :2]
        if len(far):
            gap = np.linalg.norm(far[:, None, :] - position[None, :, :], axis=-1).min()
            if gap <= limit:
                raise AssertionError(f"clutter within {limit} m of a ground truth at frame {frame}")
