"""Refusing to compare geometry across coordinate frames."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    Detections3D,
    LabelRegistry,
    MatchResults,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import BatchConfidence
from t4perceval.descriptors import CONFIDENCE
from t4perceval.system import CenterDistanceMatchingSystem, Pipeline, SystemContext
from t4perceval.system.base import require_same_frame, resolve_frame

EVERYTHING = TimeRange.everything()


@pytest.fixture
def labels() -> LabelRegistry:
    return LabelRegistry.from_names(["car"])


def boxes(labels: LabelRegistry, x: float, count: int = 1) -> Detections3D:
    return Detections3D(
        np.tile([x, 0.0, 0.0], (count, 1)),
        np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)),
        np.tile([2.0, 4.0, 1.5], (count, 1)),
        labels.encode(["car"] * count),
        np.ones(count),
    )


def two_frames(labels: LabelRegistry, est_frame: str | None, gt_frame: str | None) -> Store:
    store = Store()
    store.log(
        "/ground_truth/objects",
        boxes(labels, 0.0),
        at=TimePoint.at(frame=0),
        frame_id=gt_frame,
    )
    store.log(
        "/estimation/objects",
        boxes(labels, 0.1),
        at=TimePoint.at(frame=0),
        frame_id=est_frame,
    )
    return store


def run(store: Store, labels: LabelRegistry, **params: object) -> MatchResults:
    matcher = CenterDistanceMatchingSystem.between(
        "/estimation/objects",
        "/ground_truth/objects",
        threshold=1.0,
        **params,  # type: ignore[arg-type]
    )
    Pipeline([matcher]).run(SystemContext(store, FRAME, labels=labels), EVERYTHING)
    return store.range(matcher.target, timeline=FRAME, time_range=EVERYTHING).materialize(
        MatchResults,
    )


class TestHelper:
    def test_no_views_state_nothing(self) -> None:
        assert resolve_frame() is None
        assert require_same_frame() is None

    def test_unstated_frames_never_raise(self, labels: LabelRegistry) -> None:
        store = two_frames(labels, None, None)
        views = [
            store.range(path, timeline=FRAME, time_range=EVERYTHING)
            for path in ("/estimation/objects", "/ground_truth/objects")
        ]

        assert require_same_frame(*views) is None

    def test_one_stated_frame_wins(self, labels: LabelRegistry) -> None:
        store = two_frames(labels, None, "base_link")
        views = [
            store.range(path, timeline=FRAME, time_range=EVERYTHING)
            for path in ("/estimation/objects", "/ground_truth/objects")
        ]

        assert require_same_frame(*views) == "base_link"

    def test_a_frame_with_no_rows_still_states_its_frame(self, labels: LabelRegistry) -> None:
        # An empty frame is recorded in a frame like any other. Ignoring it would make a
        # system's output frame flicker across a scene, which concat_chunks then rejects.
        store = Store()
        store.log(
            "/estimation/objects",
            boxes(labels, 0.0, count=0),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        view = store.range("/estimation/objects", timeline=FRAME, time_range=EVERYTHING)

        assert len(view) == 0
        assert resolve_frame(view) == "base_link"


class TestMatching:
    def test_disagreeing_frames_raise(self, labels: LabelRegistry) -> None:
        # Without this the subtraction still produces numbers, and the metric built on
        # them looks entirely plausible.
        store = two_frames(labels, "base_link", "map")

        with pytest.raises(ValueError, match="across coordinate frames"):
            run(store, labels)

    def test_the_message_names_both_entities_and_frames(self, labels: LabelRegistry) -> None:
        store = two_frames(labels, "base_link", "map")

        with pytest.raises(ValueError) as error:
            run(store, labels)

        message = str(error.value)
        assert "/estimation/objects in 'base_link'" in message
        assert "/ground_truth/objects in 'map'" in message

    def test_agreeing_frames_match(self, labels: LabelRegistry) -> None:
        store = two_frames(labels, "base_link", "base_link")

        assert run(store, labels).num_tp == 1

    def test_unknown_frames_still_match(self, labels: LabelRegistry) -> None:
        # Most stores predate frame recording; an unstated frame is not a disagreement.
        store = two_frames(labels, None, None)

        assert run(store, labels).num_tp == 1

    def test_a_statically_stated_frame_is_not_a_disagreement(
        self,
        labels: LabelRegistry,
    ) -> None:
        # A static column's frame does not describe the temporal rows -- and a transform
        # edge states its *parent* there -- so it must not reach the geometry guard. This
        # is the regression: matching in `map` against static data logged as `base_link`
        # would otherwise start raising.
        store = two_frames(labels, "map", "map")
        store.log_static_components(
            "/estimation/objects",
            {CONFIDENCE: BatchConfidence([0.5])},
            frame_id="base_link",
        )

        assert run(store, labels).num_tp == 1

    def test_the_check_can_be_opted_out_of(self, labels: LabelRegistry) -> None:
        store = two_frames(labels, "base_link", "map")

        assert run(store, labels, check_frames=False).num_tp == 1

    def test_the_result_records_the_agreed_frame(self, labels: LabelRegistry) -> None:
        store = two_frames(labels, "base_link", "base_link")
        run(store, labels)

        chunk = store.chunks("/matching/center_distance")[0]

        assert chunk.frame_id == "base_link"


class TestMetrics:
    def test_a_metric_over_disagreeing_frames_raises(self, labels: LabelRegistry) -> None:
        # Every geometric metric reaches its inputs through MatchJoin, so the check there
        # covers all of them at once.
        from t4perceval.system.join import MatchJoin

        store = two_frames(labels, "base_link", "map")
        run(store, labels, check_frames=False)

        with pytest.raises(ValueError, match="across coordinate frames"):
            MatchJoin.of(
                store,
                "/matching/center_distance",
                "/estimation/objects",
                "/ground_truth/objects",
                timeline=FRAME,
                time_range=EVERYTHING,
            )
