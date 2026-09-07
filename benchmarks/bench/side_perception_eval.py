"""The perception_eval side: build objects from a scene, score them, read the metrics.

Uses the manager-free surface of ``perception_eval`` 1.3.6 -- ``NuscenesObjectMatcher`` /
``get_object_results`` plus ``MetricsScore`` -- so no dataset, result directory or
visualizer is involved. Every import happens inside a function.

Conversions at this boundary: pyquaternion is ``wxyz`` (built from an axis-angle here, so
no element order is assumed); ``Shape.size`` is ``(width, length, height)``, the same order
as the scene; predicted paths are per-mode lists whose timestamps are only used for their
length.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from bench.scenes import (
    DETECTION_THRESHOLDS,
    MATCHING_THRESHOLD,
    MISS_TOLERANCE,
    TOP_KS,
    Frame,
    Scene,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

ALL = "ALL"
FRAME_TIME_US = 500_000

Kind = str


def _labels(class_names: Sequence[str]) -> tuple[list[Any], list[Any]]:
    from perception_eval.common.label import AutowareLabel, Label

    targets = [AutowareLabel(name) for name in class_names]
    labels = [Label(target, name, []) for target, name in zip(targets, class_names)]
    return targets, labels


def _objects(
    frame: Frame, frame_index: int, kind: Kind, labels: list[Any], time_offset_ns: np.ndarray
) -> list[Any]:
    from perception_eval.common.object import DynamicObject
    from perception_eval.common.schema import FrameID
    from perception_eval.common.shape import Shape, ShapeType
    from pyquaternion import Quaternion

    unix_time = frame_index * FRAME_TIME_US
    timestamps = [int(offset // 1000) for offset in time_offset_ns.tolist()]
    positions = frame.position.tolist()
    yaws = frame.yaw.tolist()
    sizes = frame.size.tolist()
    velocities = frame.velocity.tolist()
    confidences = frame.confidence.tolist()
    classes = frame.class_id.tolist()
    instances = frame.instance.tolist()
    waypoints = frame.waypoints.tolist() if kind == "prediction" else None
    mode_confidences = frame.mode_confidence.tolist() if kind == "prediction" else None

    objects = []
    for index in range(len(frame)):
        extra: dict[str, Any] = {}
        if kind != "detection":
            extra["uuid"] = str(instances[index])
        if kind == "prediction":
            modes = waypoints[index]
            extra["relative_timestamps"] = [timestamps] * len(modes)
            extra["predicted_positions"] = [[tuple(p) for p in mode] for mode in modes]
            extra["predicted_scores"] = mode_confidences[index]
        objects.append(
            DynamicObject(
                unix_time=unix_time,
                frame_id=FrameID.BASE_LINK,
                position=tuple(positions[index]),
                orientation=Quaternion(axis=[0.0, 0.0, 1.0], radians=yaws[index]),
                shape=Shape(ShapeType.BOUNDING_BOX, tuple(sizes[index])),
                velocity=tuple(velocities[index]),
                semantic_score=confidences[index],
                semantic_label=labels[classes[index]],
                **extra,
            ),
        )
    return objects


def build_frame(scene: Scene, frame: int, kind: Kind) -> tuple[list[Any], list[Any]]:
    """Turn one frame into (estimations, ground truths) lists."""
    _, labels = _labels(scene.class_names)
    return (
        _objects(scene.est(frame), frame, kind, labels, scene.time_offset_ns),
        _objects(scene.gt(frame), frame, kind, labels, scene.time_offset_ns),
    )


def _frames(scene: Scene, kind: Kind) -> list[tuple[list[Any], list[Any]]]:
    return [build_frame(scene, frame, kind) for frame in range(scene.num_frames)]


# -- timed calls ------------------------------------------------------------------------


def memory_builder(scene: Scene) -> Callable[[], tuple[Any, Any]]:
    return lambda: build_frame(scene, 0, "detection")


def matching_call(scene: Scene) -> Callable[[], Any]:
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.evaluation.matching import MatchingMode
    from perception_eval.evaluation.result.object_result_matching import get_object_results

    targets, _ = _labels(scene.class_names)
    est, gt = build_frame(scene, 0, "detection")
    thresholds = [MATCHING_THRESHOLD] * len(targets)

    def call() -> Any:
        results = get_object_results(
            EvaluationTask.DETECTION,
            est,
            gt,
            target_labels=targets,
            matching_mode=MatchingMode.CENTERDISTANCE,
            matchable_thresholds=thresholds,
        )
        if not 0 < len(results) <= len(est) + len(gt):
            raise RuntimeError(f"unexpected result count {len(results)}")
        return results

    call()
    return call


def _num_ground_truth(
    frames: list[tuple[list[Any], list[Any]]], targets: list[Any]
) -> dict[Any, int]:
    from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num

    return divide_objects_to_num([obj for _, gts in frames for obj in gts], targets)


def detection_call(scene: Scene) -> Callable[[], Any]:
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.evaluation.metrics import MetricsScore, MetricsScoreConfig
    from perception_eval.evaluation.result.object_result_matching import NuscenesObjectMatcher
    from perception_eval.util.aggregation_results import accumulate_nuscene_results

    targets, _ = _labels(scene.class_names)
    frames = _frames(scene, "detection")
    num_gt = _num_ground_truth(frames, targets)
    config = MetricsScoreConfig(
        EvaluationTask.DETECTION,
        target_labels=targets,
        center_distance_thresholds=list(DETECTION_THRESHOLDS),
    )
    matcher = NuscenesObjectMatcher(evaluation_task=EvaluationTask.DETECTION, metrics_config=config)
    used = list(range(scene.num_frames))

    def call() -> Any:
        accumulated: Any = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for mode, per_label in matcher.matching_config_map.items():
            for label, thresholds in per_label.items():
                for threshold in thresholds:
                    accumulated[mode][label][threshold] = []
        for est, gt in frames:
            accumulate_nuscene_results(accumulated, matcher.match(est, gt))
        score = MetricsScore(config, used_frame=used)
        score.evaluate_detection(accumulated, num_gt)
        return score

    return call


def _per_frame_results(
    frames: list[tuple[list[Any], list[Any]]], task: Any, targets: list[Any]
) -> list[dict[Any, list[Any]]]:
    from perception_eval.evaluation.matching import MatchingMode
    from perception_eval.evaluation.matching.objects_filter import divide_objects
    from perception_eval.evaluation.result.object_result_matching import get_object_results

    thresholds = [MATCHING_THRESHOLD] * len(targets)
    divided = []
    for est, gt in frames:
        results = get_object_results(
            task,
            est,
            gt,
            target_labels=targets,
            matching_mode=MatchingMode.CENTERDISTANCE,
            matchable_thresholds=thresholds,
        )
        divided.append(divide_objects(results, targets))
    return divided


def tracking_call(scene: Scene) -> Callable[[], Any]:
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.evaluation.metrics import MetricsScore, MetricsScoreConfig

    targets, _ = _labels(scene.class_names)
    frames = _frames(scene, "tracking")
    num_gt = _num_ground_truth(frames, targets)
    config = MetricsScoreConfig(
        EvaluationTask.TRACKING,
        target_labels=targets,
        center_distance_thresholds=[MATCHING_THRESHOLD],
        center_distance_bev_thresholds=None,
        plane_distance_thresholds=None,
        iou_2d_thresholds=None,
        iou_3d_thresholds=None,
    )
    used = list(range(scene.num_frames))

    def call() -> Any:
        divided = _per_frame_results(frames, EvaluationTask.TRACKING, targets)
        # CLEAR scores frame i against frame i-1, so an empty frame in front makes frame 0
        # count as well (the library's own manager omits it and skips the first frame).
        by_label = {label: [[]] + [d.get(label, []) for d in divided] for label in targets}
        score = MetricsScore(config, used_frame=used)
        score.evaluate_tracking(by_label, num_gt)
        return score

    return call


def prediction_call(scene: Scene) -> Callable[[], Any]:
    from perception_eval.common.evaluation_task import EvaluationTask
    from perception_eval.evaluation.metrics import MetricsScore, MetricsScoreConfig

    targets, _ = _labels(scene.class_names)
    frames = _frames(scene, "prediction")
    num_gt = _num_ground_truth(frames, targets)
    config = MetricsScoreConfig(
        EvaluationTask.PREDICTION,
        target_labels=targets,
        top_ks=list(TOP_KS),
        miss_tolerance=MISS_TOLERANCE,
    )
    used = list(range(scene.num_frames))

    def call() -> Any:
        divided = _per_frame_results(frames, EvaluationTask.PREDICTION, targets)
        by_label = {label: [d.get(label, []) for d in divided] for label in targets}
        score = MetricsScore(config, used_frame=used)
        score.evaluate_prediction(by_label, num_gt)
        return score

    return call


# -- reading metrics --------------------------------------------------------------------


def _clean(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _label(threshold: float) -> str:
    return f"{threshold:g}"


def _nanmean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def detection_metrics(score: Any, class_names: Sequence[str]) -> dict[str, float | None]:
    targets, _ = _labels(class_names)
    (mean_ap,) = score.mean_ap_values
    out: dict[str, float | None] = {}
    for kind, per_label, per_label_mean, overall in (
        ("ap", mean_ap.label_to_aps, mean_ap.label_mean_to_ap, mean_ap.map),
        ("aph", mean_ap.label_to_aphs, mean_ap.label_mean_to_aph, mean_ap.maph),
    ):
        for target, name in zip(targets, class_names):
            aps = per_label[target]
            for j, threshold in enumerate(DETECTION_THRESHOLDS):
                if not np.isclose(aps[j].matching_threshold, threshold):
                    raise RuntimeError("threshold order differs from DETECTION_THRESHOLDS")
                out[f"detection/{kind}/{_label(threshold)}/{name}"] = _clean(aps[j].ap)
            out[f"detection/{kind}/mean/{name}"] = _clean(per_label_mean[target])
        for j, threshold in enumerate(DETECTION_THRESHOLDS):
            out[f"detection/{kind}/{_label(threshold)}/{ALL}"] = _nanmean(
                [_clean(per_label[target][j].ap) for target in targets],
            )
        out[f"detection/m{kind}/{ALL}"] = _clean(overall)
    return out


def tracking_metrics(score: Any, class_names: Sequence[str]) -> dict[str, float | None]:
    (tracking,) = score.tracking_scores
    prefix = f"tracking/{{}}/{_label(MATCHING_THRESHOLD)}/"
    out: dict[str, float | None] = {}
    for clear, name in zip(tracking.clears, class_names):
        out[prefix.format("mota") + name] = _clean(clear.mota)
        out[prefix.format("motp") + name] = _clean(clear.motp)
        out[prefix.format("id_switch") + name] = _clean(clear.id_switch)
    mota, motp, switches = tracking._sum_clear()
    out[prefix.format("mota") + ALL] = _clean(mota)
    out[prefix.format("motp") + ALL] = _clean(motp)
    out[prefix.format("id_switch") + ALL] = _clean(switches)
    return out


def prediction_metrics(score: Any, class_names: Sequence[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k, prediction in zip(TOP_KS, score.prediction_scores):
        if prediction.top_k != k:
            raise RuntimeError("top-k order differs from TOP_KS")
        for displacement, name in zip(prediction.displacements, class_names):
            out[f"prediction/ade/k{k}/{name}"] = _clean(displacement.ade)
            out[f"prediction/fde/k{k}/{name}"] = _clean(displacement.fde)
            out[f"prediction/miss_rate/k{k}/{name}"] = _clean(displacement.miss_rate)
        out[f"prediction/ade/k{k}/{ALL}"] = _clean(prediction.ade)
        out[f"prediction/fde/k{k}/{ALL}"] = _clean(prediction.fde)
        out[f"prediction/miss_rate/k{k}/{ALL}"] = _clean(prediction.miss_rate)
    return out
