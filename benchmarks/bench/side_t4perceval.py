"""The t4perceval side: build a store from a scene, run the pipelines, read the metrics.

Every import of the library happens inside a function so that this module can be imported
under an interpreter that does not have it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from bench.scenes import (
    DETECTION_THRESHOLDS,
    MATCHING_THRESHOLD,
    MISS_TOLERANCE,
    TOP_KS,
    Frame,
    Scene,
    yaw_to_quaternion_xyzw,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

EST = "/estimation/objects"
GT = "/ground_truth/objects"
MATCHING = "/matching/center_distance"
FRAME_ID = "base_link"
ALL = "ALL"

Kind = str  # "detection" | "tracking" | "prediction"


def _archetype(frame: Frame, kind: Kind, time_offset_ns: np.ndarray) -> Any:
    from t4perceval import Detections3D, Predictions3D, Trackings3D

    common = {
        "position": frame.position,
        "quaternion": yaw_to_quaternion_xyzw(frame.yaw),
        "size": frame.size,
        "class_id": frame.class_id,
        "confidence": frame.confidence,
        "velocity": frame.velocity,
    }
    if kind == "detection":
        return Detections3D(**common)
    if kind == "tracking":
        return Trackings3D(**common, instance_id=frame.instance)
    if kind == "prediction":
        return Predictions3D(
            **common,
            instance_id=frame.instance,
            waypoints=frame.waypoints,
            mode_confidence=frame.mode_confidence,
            time_offset=np.tile(time_offset_ns, (len(frame), 1)),
        )
    raise ValueError(f"unknown kind {kind!r}")


def build_frame(scene: Scene, frame: int, kind: Kind) -> tuple[Any, Any]:
    """Turn one frame into an (estimation, ground truth) archetype pair."""
    return (
        _archetype(scene.est(frame), kind, scene.time_offset_ns),
        _archetype(scene.gt(frame), kind, scene.time_offset_ns),
    )


def build_chunks(scene: Scene, kind: Kind) -> list[Any]:
    from t4perceval import TimePoint

    chunks = []
    for frame in range(scene.num_frames):
        est, gt = build_frame(scene, frame, kind)
        at = TimePoint.at(frame=frame)
        chunks.append(est.to_chunk(EST, at=at, frame_id=FRAME_ID))
        chunks.append(gt.to_chunk(GT, at=at, frame_id=FRAME_ID))
    return chunks


def fresh_store(chunks: Sequence[Any]) -> Any:
    from t4perceval import Store

    store = Store()
    for chunk in chunks:
        store.send_chunk(chunk)
    return store


def registry(scene: Scene) -> Any:
    from t4perceval import LabelRegistry

    return LabelRegistry.from_names(scene.class_names)


def _context(store: Any, labels: Any) -> Any:
    from t4perceval import FRAME
    from t4perceval.system import SystemContext

    return SystemContext(store, FRAME, labels=labels)


# -- timed calls ------------------------------------------------------------------------


def memory_builder(scene: Scene) -> Callable[[], tuple[Any, Any]]:
    return lambda: build_frame(scene, 0, "detection")


def matching_call(scene: Scene) -> Callable[[], Any]:
    """One frame of center-distance matching, everything else prebuilt."""
    from t4perceval.system import CenterDistanceMatchingSystem

    est, gt = build_frame(scene, 0, "detection")
    store = fresh_store(build_chunks(_single_frame(scene), "detection"))
    ctx = _context(store, registry(scene))
    matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=MATCHING_THRESHOLD)
    expected = len(est) + len(gt)  # every row is a TP pair, an FP or an FN; pairs count once

    def call() -> Any:
        chunks = tuple(matcher(ctx, 0))
        rows = sum(chunk.num_rows for chunk in chunks)
        if not 0 < rows <= expected:
            raise RuntimeError(f"unexpected match row count {rows}")
        return chunks

    call()
    return call


def _single_frame(scene: Scene) -> Scene:
    """A view of the scene holding frame 0 only (for the single-frame matching phase)."""
    from dataclasses import replace

    gt_hi, est_hi = scene.gt_offsets[1], scene.est_offsets[1]
    return replace(
        scene,
        gt_offsets=scene.gt_offsets[:2],
        est_offsets=scene.est_offsets[:2],
        gt_position=scene.gt_position[:gt_hi],
        gt_yaw=scene.gt_yaw[:gt_hi],
        gt_size=scene.gt_size[:gt_hi],
        gt_class=scene.gt_class[:gt_hi],
        gt_instance=scene.gt_instance[:gt_hi],
        gt_velocity=scene.gt_velocity[:gt_hi],
        gt_waypoints=scene.gt_waypoints[:gt_hi],
        est_position=scene.est_position[:est_hi],
        est_yaw=scene.est_yaw[:est_hi],
        est_size=scene.est_size[:est_hi],
        est_class=scene.est_class[:est_hi],
        est_instance=scene.est_instance[:est_hi],
        est_velocity=scene.est_velocity[:est_hi],
        est_confidence=scene.est_confidence[:est_hi],
        est_waypoints=scene.est_waypoints[:est_hi],
        est_mode_confidence=scene.est_mode_confidence[:est_hi],
    )


def _pipeline_call(scene: Scene, kind: Kind, systems: list[Any]) -> Callable[[], Any]:
    """Time ``Pipeline.run`` over the whole scene against a fresh store each call.

    The store is rebuilt from prebuilt chunks because a second run would append metric rows
    at the same reporting time and make the per-class lookup ambiguous; sending ~2F chunks
    costs microseconds and is stated in the report.
    """
    from t4perceval import TimeRange
    from t4perceval.system import Pipeline

    chunks = build_chunks(scene, kind)
    labels = registry(scene)
    pipeline = Pipeline(systems)
    everything = TimeRange.everything()

    def call() -> Any:
        store = fresh_store(chunks)
        pipeline.run(_context(store, labels), everything)
        return store

    return call


def detection_call(scene: Scene) -> Callable[[], Any]:
    from t4perceval.system import average_precision_sweep

    systems = average_precision_sweep(EST, GT, thresholds=list(DETECTION_THRESHOLDS), heading=True)
    return _pipeline_call(scene, "detection", systems)


def tracking_call(scene: Scene) -> Callable[[], Any]:
    from t4perceval.system import CenterDistanceMatchingSystem, ClearSystem

    systems = [
        CenterDistanceMatchingSystem.between(EST, GT, threshold=MATCHING_THRESHOLD),
        ClearSystem.on(MATCHING, EST, GT),
    ]
    return _pipeline_call(scene, "tracking", systems)


def prediction_call(scene: Scene) -> Callable[[], Any]:
    from t4perceval.system import CenterDistanceMatchingSystem, PathDisplacementSystem

    systems: list[Any] = [
        CenterDistanceMatchingSystem.between(EST, GT, threshold=MATCHING_THRESHOLD)
    ]
    for k in TOP_KS:
        systems.append(
            PathDisplacementSystem.on(
                MATCHING,
                EST,
                GT,
                top_k=k,
                miss_tolerance=MISS_TOLERANCE,
                kernel=None,
                target=f"/metrics/displacement/k{k}",
            ),
        )
    return _pipeline_call(scene, "prediction", systems)


# -- reading metrics --------------------------------------------------------------------


def _values(store: Any, path: str) -> Any:
    from t4perceval import FRAME, MetricValues, TimeRange

    return store.range(path, timeline=FRAME, time_range=TimeRange.everything()).materialize(
        MetricValues,
    )


def _row(values: Any, class_id: int, threshold: float | None) -> float:
    """Return the one value for ``(class_id, threshold)``; ``None`` matches a NaN threshold."""
    thresholds = values.threshold.values
    mask = values.class_id.values == class_id
    mask &= np.isnan(thresholds) if threshold is None else np.isclose(thresholds, threshold)
    (index,) = np.flatnonzero(mask)
    return float(values.value.values[index])


def _clean(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _label(threshold: float) -> str:
    return f"{threshold:g}"


def detection_metrics(store: Any, class_names: Sequence[str]) -> dict[str, float | None]:
    from t4perceval.archetype.metric import ALL_CLASSES

    out: dict[str, float | None] = {}
    for kind in ("ap", "aph"):
        for j, threshold in enumerate(DETECTION_THRESHOLDS):
            values = _values(store, f"/metrics/{kind}/{j}")
            for c, name in enumerate(class_names):
                out[f"detection/{kind}/{_label(threshold)}/{name}"] = _clean(values.of_class(c))
        mean = _values(store, "/metrics/map" if kind == "ap" else "/metrics/maph")
        for c, name in enumerate(class_names):
            out[f"detection/{kind}/mean/{name}"] = _clean(_row(mean, c, None))
        for threshold in DETECTION_THRESHOLDS:
            out[f"detection/{kind}/{_label(threshold)}/{ALL}"] = _clean(
                _row(mean, ALL_CLASSES, threshold),
            )
        out[f"detection/m{kind}/{ALL}"] = _clean(mean.aggregate)
    return out


def tracking_metrics(store: Any, class_names: Sequence[str]) -> dict[str, float | None]:
    from t4perceval import FRAME, TimeRange
    from t4perceval.component import MatchStatus
    from t4perceval.descriptors import CLASS_ID, MATCH_STATUS, MATCHING_SCORE
    from t4perceval.system import MatchJoin

    prefix = f"tracking/{{}}/{_label(MATCHING_THRESHOLD)}/"
    out: dict[str, float | None] = {}
    per_class = {
        name: _values(store, f"/metrics/clear/{name}") for name in ("mota", "motp", "id_switch")
    }
    for c, name in enumerate(class_names):
        for metric, values in per_class.items():
            out[prefix.format(metric) + name] = _clean(values.of_class(c))

    # perception_eval's scene-wide numbers, derived with its formulas: MOTA weighted by
    # ground-truth count, MOTP weighted by true-positive count, switches summed.
    mota = per_class["mota"]
    support = mota.support.values.astype(np.float64)
    finite = np.isfinite(mota.value.values)
    out[prefix.format("mota") + ALL] = _clean(
        max(0.0, float(np.sum(mota.value.values[finite] * support[finite]) / support.sum()))
        if support.sum()
        else np.nan,
    )
    join = MatchJoin.of(store, MATCHING, EST, GT, timeline=FRAME, time_range=TimeRange.everything())
    status = join.match_component(MATCH_STATUS)
    true_positive = (status == int(MatchStatus.TP)) & join.is_label_correct()
    score = join.match_component(MATCHING_SCORE)
    out[prefix.format("motp") + ALL] = _clean(
        float(np.mean(score[true_positive])) if true_positive.any() else np.nan,
    )
    switches = per_class["id_switch"].value.values
    out[prefix.format("id_switch") + ALL] = _clean(float(np.nansum(switches)))
    del CLASS_ID
    return out


def prediction_metrics(store: Any, class_names: Sequence[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in TOP_KS:
        for metric in ("ade", "fde", "miss_rate"):
            values = _values(store, f"/metrics/displacement/k{k}/{metric}")
            per_class = [values.of_class(c) for c in range(len(class_names))]
            for name, value in zip(class_names, per_class):
                out[f"prediction/{metric}/k{k}/{name}"] = _clean(value)
            finite = [value for value in per_class if np.isfinite(value)]
            out[f"prediction/{metric}/k{k}/{ALL}"] = _clean(np.mean(finite) if finite else np.nan)
    return out
