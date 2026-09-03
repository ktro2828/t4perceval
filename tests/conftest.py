from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    Detections3D,
    LabelRegistry,
    Predictions3D,
    Store,
    TimePoint,
    Trackings3D,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from t4perceval.typing import ArrayLike


def make_detections(
    positions: Sequence[Sequence[float]],
    class_ids: Sequence[int] | None = None,
    *,
    confidences: Sequence[float] | None = None,
    velocity: ArrayLike | None = None,
) -> Detections3D:
    """Build a detection batch with sensible filler for everything not under test."""
    count = len(positions)
    return Detections3D(
        position=positions,
        quaternion=[[0.0, 0.0, 0.0, 1.0]] * count,
        size=[[1.0, 1.0, 1.0]] * count,
        class_id=[0] * count if class_ids is None else class_ids,
        confidence=[0.9] * count if confidences is None else confidences,
        velocity=velocity,
    )


def make_trackings(
    positions: Sequence[Sequence[float]],
    instance_ids: Sequence[int],
    class_ids: Sequence[int] | None = None,
) -> Trackings3D:
    detection = make_detections(positions, class_ids)
    return Trackings3D(
        position=detection.position,
        quaternion=detection.quaternion,
        size=detection.size,
        class_id=detection.class_id,
        confidence=detection.confidence,
        instance_id=instance_ids,
    )


def make_predictions(
    positions: Sequence[Sequence[float]],
    instance_ids: Sequence[int],
    *,
    num_modes: int = 2,
    num_timesteps: int = 3,
) -> Predictions3D:
    count = len(positions)
    tracking = make_trackings(positions, instance_ids)
    return Predictions3D(
        position=tracking.position,
        quaternion=tracking.quaternion,
        size=tracking.size,
        class_id=tracking.class_id,
        confidence=tracking.confidence,
        instance_id=tracking.instance_id,
        waypoints=np.zeros((count, num_modes, num_timesteps, 3), dtype=np.float64),
        mode_confidence=np.full((count, num_modes), 1.0 / num_modes, dtype=np.float64),
        time_offset=np.tile(np.arange(1, num_timesteps + 1) * 100, (count, 1)),
    )


def make_metric_scene(
    labels: LabelRegistry,
    frames: Sequence[tuple[int, Sequence[tuple[float, str]], Sequence[tuple[float, str, float]]]],
    *,
    ground_truth: str = "/ground_truth/objects",
    estimation: str = "/estimation/objects",
) -> Store:
    """Build a store from a compact description of a scene.

    Each frame is ``(frame_index, [(x, class_name), ...], [(x, class_name, confidence), ...])``
    -- the ground truths first, then the estimations.
    """
    store = Store()
    for frame, gt_objects, est_objects in frames:
        store.log(
            ground_truth,
            make_detections(
                [[x, 0.0, 0.0] for x, _ in gt_objects],
                labels.encode([name for _, name in gt_objects]),
                confidences=[1.0] * len(gt_objects),
            ),
            at=TimePoint.at(frame=frame),
            frame_id="base_link",
        )
        store.log(
            estimation,
            make_detections(
                [[x, 0.0, 0.0] for x, _, _ in est_objects],
                labels.encode([name for _, name, _ in est_objects]),
                confidences=[confidence for _, _, confidence in est_objects],
            ),
            at=TimePoint.at(frame=frame),
            frame_id="base_link",
        )
    return store


@pytest.fixture
def labels() -> LabelRegistry:
    return LabelRegistry.from_names(["car", "truck", "pedestrian"])


@pytest.fixture
def scene_store() -> Store:
    """Two frames of ground truth and estimation, in ``base_link``.

    Frame 0: gt at x=0 and x=10; est at x=0.3 (a hit) and x=50 (far away).
    Frame 1: gt at (1, 1) as a pedestrian; est at (1.1, 1) as a pedestrian and
    (2, 2) as a car, so the second estimation can only fail on its class.
    """
    store = Store()
    store.log(
        "/ground_truth/objects",
        make_detections([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [0, 0]),
        at=TimePoint.at(frame=0, timestamp_ns=1_000),
        frame_id="base_link",
    )
    store.log(
        "/estimation/objects",
        make_detections([[0.3, 0.0, 0.0], [50.0, 0.0, 0.0]], [0, 0]),
        at=TimePoint.at(frame=0, timestamp_ns=1_000),
        frame_id="base_link",
    )
    store.log(
        "/ground_truth/objects",
        make_detections([[1.0, 1.0, 0.0]], [2]),
        at=TimePoint.at(frame=1, timestamp_ns=2_000),
        frame_id="base_link",
    )
    store.log(
        "/estimation/objects",
        make_detections([[1.1, 1.0, 0.0], [2.0, 2.0, 0.0]], [2, 0]),
        at=TimePoint.at(frame=1, timestamp_ns=2_000),
        frame_id="base_link",
    )
    return store


@pytest.fixture
def frame_timeline() -> object:
    return FRAME


T4_DATASET_ROOT = Path(__file__).parent / "data" / "t4dataset"


@pytest.fixture(scope="session")
def t4_dataset_root() -> Path:
    """The vendored minimal T4 dataset. See ``tests/data/t4dataset/README.md``."""
    if not T4_DATASET_ROOT.exists():  # pragma: no cover - the fixture is committed
        pytest.skip(f"missing T4 fixture at {T4_DATASET_ROOT}")
    return T4_DATASET_ROOT


@pytest.fixture
def t4_importer(t4_dataset_root: Path) -> object:
    """A default importer over the vendored dataset."""
    pytest.importorskip("t4_devkit")
    from t4perceval.importer.t4 import T4Importer

    return T4Importer.open(t4_dataset_root)
