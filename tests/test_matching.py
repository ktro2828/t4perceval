"""Matching systems: the score of each mode, and the assignment they share."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import make_detections

from t4perceval import (
    FRAME,
    Detections2D,
    Detections3D,
    MatchResults,
    LabelRegistry,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.component import MatchStatus
from t4perceval.descriptors import CLASS_ID
from t4perceval.system import (
    CenterDistanceBEVMatchingSystem,
    CenterDistanceMatchingSystem,
    IoU3DMatchingSystem,
    IoUBEVMatchingSystem,
    IoURoiMatchingSystem,
    MatchingSystem,
    Pipeline,
    PlaneDistanceMatchingSystem,
    SystemContext,
    Thresholds,
)

if TYPE_CHECKING:
    from t4perceval.core.view import EntityView
    from t4perceval.typing import NDArrayF64

EST = "/estimation/objects"
GT = "/ground_truth/objects"

#: The 3D modes, which all accept the same box components.
BOX_MODES: tuple[type[MatchingSystem], ...] = (
    CenterDistanceMatchingSystem,
    CenterDistanceBEVMatchingSystem,
    PlaneDistanceMatchingSystem,
    IoUBEVMatchingSystem,
    IoU3DMatchingSystem,
)

#: Every mode, including the 2D one, for the properties that hold across the family.
ALL_MODES: tuple[type[MatchingSystem], ...] = (*BOX_MODES, IoURoiMatchingSystem)


def yaw(angle: float) -> list[float]:
    return [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]


def boxes(
    positions: list[list[float]],
    labels: LabelRegistry,
    names: list[str] | None = None,
    *,
    yaws: list[float] | None = None,
    sizes: list[list[float]] | None = None,
) -> Detections3D:
    count = len(positions)
    return Detections3D(
        position=positions,
        quaternion=[yaw(angle) for angle in (yaws or [0.0] * count)],
        size=sizes or [[2.0, 4.0, 2.0]] * count,
        class_id=labels.encode(names or ["car"] * count),
        confidence=[0.9] * count,
    )


def rois(
    values: list[list[int]],
    labels: LabelRegistry,
    names: list[str] | None = None,
) -> Detections2D:
    count = len(values)
    return Detections2D(
        roi=values,
        class_id=labels.encode(names or ["car"] * count),
        confidence=[0.9] * count,
    )


@pytest.fixture
def box_store(labels: LabelRegistry) -> Store:
    """One frame: a car ground truth at x=10, and an estimation 0.5 m further out."""
    store = Store()
    store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0), frame_id="base_link")
    store.log(
        EST,
        boxes([[10.5, 0.0, 0.0]], labels),
        at=TimePoint.at(frame=0),
        frame_id="base_link",
    )
    return store


@pytest.fixture
def roi_store(labels: LabelRegistry) -> Store:
    """One frame of 2D detections: a 10x10 ground-truth ROI and one shifted by 2 px."""
    store = Store()
    store.log(GT, rois([[0, 0, 10, 10]], labels), at=TimePoint.at(frame=0))
    store.log(EST, rois([[2, 0, 10, 10]], labels), at=TimePoint.at(frame=0))
    return store


def store_for(mode: type[MatchingSystem], labels: LabelRegistry) -> Store:
    """Return a store whose entities carry the components ``mode`` requires.

    The estimation coincides with the ground truth, so every mode scores a perfect match.
    """
    store = Store()
    if mode is IoURoiMatchingSystem:
        store.log(GT, rois([[0, 0, 10, 10]], labels), at=TimePoint.at(frame=0))
        store.log(EST, rois([[0, 0, 10, 10]], labels), at=TimePoint.at(frame=0))
    else:
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(EST, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
    return store


def imperfect_store_for(mode: type[MatchingSystem], labels: LabelRegistry) -> Store:
    """Like :func:`store_for`, but offset so every mode scores strictly imperfectly."""
    store = Store()
    if mode is IoURoiMatchingSystem:
        store.log(GT, rois([[0, 0, 10, 10]], labels), at=TimePoint.at(frame=0))
        store.log(EST, rois([[2, 0, 10, 10]], labels), at=TimePoint.at(frame=0))
    else:
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(EST, boxes([[10.5, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
    return store


def result_of(
    system: MatchingSystem,
    ctx: SystemContext,
    at: object = 0,
) -> MatchResults:
    (chunk,) = system(ctx, at)
    return MatchResults.from_chunk(chunk)


def counts(result: MatchResults) -> tuple[int, int, int]:
    return result.num_tp, result.num_fp, result.num_fn


class TestFamilyProperties:
    """Properties every matching mode must share, so a new one cannot drift."""

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_declares_two_sources_and_a_match_result(self, mode: type[MatchingSystem]) -> None:
        system = mode.between(EST, GT)

        assert len(system.sources) == 2
        assert system.PROVIDES == MatchResults.required_descriptors()
        assert CLASS_ID in system.REQUIRES

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_each_mode_gets_its_own_target(self, mode: type[MatchingSystem]) -> None:
        system = mode.between(EST, GT)

        assert str(system.target) == f"/matching/{mode.MATCHING_NAME}"

    def test_no_two_modes_share_a_target(self) -> None:
        targets = [str(mode.between(EST, GT).target) for mode in ALL_MODES]

        assert len(set(targets)) == len(targets)

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_the_target_can_be_overridden(self, mode: type[MatchingSystem]) -> None:
        assert str(mode.between(EST, GT, target="/matching/custom").target) == "/matching/custom"

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_an_exact_match_is_a_true_positive(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        store = store_for(mode, labels)

        result = result_of(mode.between(EST, GT), SystemContext(store, FRAME, labels=labels))

        assert counts(result) == (1, 0, 0)

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_unmatched_rows_carry_a_sentinel_and_no_score(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        store = Store()
        if mode is IoURoiMatchingSystem:
            store.log(GT, rois([[0, 0, 10, 10]], labels), at=TimePoint.at(frame=0))
            store.log(EST, rois([[500, 500, 10, 10]], labels), at=TimePoint.at(frame=0))
        else:
            store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
            store.log(EST, boxes([[500.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))

        result = result_of(mode.between(EST, GT), SystemContext(store, FRAME, labels=labels))

        assert counts(result) == (0, 1, 1)
        false_positive = result.match_status.values == int(MatchStatus.FP)
        assert (result.gt_index.values[false_positive] == -1).all()
        false_negative = result.match_status.values == int(MatchStatus.FN)
        assert (result.est_index.values[false_negative] == -1).all()
        assert np.isnan(result.matching_score.values).all()

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_a_class_mismatch_prevents_a_match(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        store = Store()
        if mode is IoURoiMatchingSystem:
            store.log(GT, rois([[0, 0, 10, 10]], labels, ["car"]), at=TimePoint.at(frame=0))
            store.log(EST, rois([[0, 0, 10, 10]], labels, ["truck"]), at=TimePoint.at(frame=0))
        else:
            store.log(GT, boxes([[10.0, 0.0, 0.0]], labels, ["car"]), at=TimePoint.at(frame=0))
            store.log(EST, boxes([[10.0, 0.0, 0.0]], labels, ["truck"]), at=TimePoint.at(frame=0))
        ctx = SystemContext(store, FRAME, labels=labels)

        strict = result_of(mode.between(EST, GT), ctx)
        agnostic = result_of(mode.between(EST, GT, class_agnostic=True), ctx)

        assert counts(strict) == (0, 1, 1)
        assert counts(agnostic) == (1, 0, 0)

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_a_frame_with_only_ground_truth_is_all_false_negatives(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        store = store_for(mode, labels)
        empty = Store()
        for chunk in store.chunks(GT):
            empty.send_chunk(chunk)

        result = result_of(mode.between(EST, GT), SystemContext(empty, FRAME, labels=labels))

        assert counts(result) == (0, 0, 1)

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_an_empty_store_produces_nothing(self, mode: type[MatchingSystem]) -> None:
        assert (
            list(mode.between(EST, GT)(SystemContext(Store(), FRAME), TimeRange.everything())) == []
        )

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_produces_one_partition_per_frame(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        store = store_for(mode, labels)
        for frame in (1, 2):
            for path in (GT, EST):
                source = store.chunks(path)[0]
                archetype = Detections2D if mode is IoURoiMatchingSystem else Detections3D
                store.log(
                    path,
                    archetype.from_chunk(source),
                    at=TimePoint.at(frame=frame),
                )

        (chunk,) = mode.between(EST, GT)(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        assert chunk.num_partitions == 3
        assert chunk.index(FRAME).times.tolist() == [0, 1, 2]

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_a_missing_component_is_reported(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        from t4perceval.archetype import Classifications2D

        store = Store()
        for path in (GT, EST):
            store.log(
                path,
                Classifications2D(class_id=[0], confidence=[1.0]),
                at=TimePoint.at(frame=0),
            )

        with pytest.raises(ValueError, match="missing required component"):
            list(mode.between(EST, GT)(SystemContext(store, FRAME, labels=labels), 0))

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_needs_exactly_two_sources(self, mode: type[MatchingSystem]) -> None:
        with pytest.raises(ValueError, match="needs exactly two sources"):
            mode((EST,), "/matching/out")

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_the_threshold_direction_matches_the_metric(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        """A distance must stay *below* its threshold; an overlap must stay *above* it.

        Measured against a deliberately imperfect pair, since a perfect one satisfies any
        threshold in either direction and would hide an inverted comparison.
        """
        store = imperfect_store_for(mode, labels)
        ctx = SystemContext(store, FRAME, labels=labels)

        score = self._score_of(mode, store, labels)
        assert 0.0 < score < 1.0 or not mode.HIGHER_IS_BETTER

        if mode.HIGHER_IS_BETTER:
            lenient, strict = score / 2.0, (score + 1.0) / 2.0
        else:
            lenient, strict = score * 2.0, score / 2.0

        assert counts(result_of(mode.between(EST, GT, threshold=lenient), ctx)) == (1, 0, 0)
        assert counts(result_of(mode.between(EST, GT, threshold=strict), ctx)) == (0, 1, 1)

    @staticmethod
    def _score_of(mode: type[MatchingSystem], store: Store, labels: LabelRegistry) -> float:
        """Return the score of the single pair, matched with a threshold that admits it."""
        admits_everything = 1e-9 if mode.HIGHER_IS_BETTER else 1e9
        result = result_of(
            mode.between(EST, GT, threshold=admits_everything),
            SystemContext(store, FRAME, labels=labels),
        )
        matched = result.match_status.values == int(MatchStatus.TP)
        return float(result.matching_score.values[matched][0])

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_a_wrongly_shaped_score_matrix_is_reported(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        from attrs import define

        @define(slots=True)
        class BrokenSystem(mode):  # type: ignore[misc,valid-type]
            def score_matrix(self, est_view: EntityView, gt_view: EntityView) -> NDArrayF64:
                del est_view, gt_view
                return np.zeros((2, 2))

        store = store_for(mode, labels)

        with pytest.raises(ValueError, match=r"returned shape \(2, 2\), expected \(1, 1\)"):
            list(BrokenSystem.between(EST, GT)(SystemContext(store, FRAME, labels=labels), 0))


class TestThresholdValidation:
    @pytest.mark.parametrize("mode", [m for m in ALL_MODES if not m.HIGHER_IS_BETTER])
    def test_a_distance_threshold_must_be_positive(self, mode: type[MatchingSystem]) -> None:
        with pytest.raises(ValueError, match="distances and must be positive"):
            mode.between(EST, GT, threshold=0.0)

    @pytest.mark.parametrize("mode", [m for m in ALL_MODES if m.HIGHER_IS_BETTER])
    @pytest.mark.parametrize("bad", [0.0, 1.5, -0.5])
    def test_an_overlap_threshold_must_lie_in_the_unit_interval(
        self,
        mode: type[MatchingSystem],
        bad: float,
    ) -> None:
        with pytest.raises(ValueError, match=r"overlap ratios and must lie in \(0, 1\]"):
            mode.between(EST, GT, threshold=bad)

    def test_per_class_thresholds_are_validated_too(self) -> None:
        with pytest.raises(ValueError, match="overlap ratios"):
            IoUBEVMatchingSystem.between(
                EST,
                GT,
                threshold=Thresholds(0.5, by_class={"car": 2.0}),
            )

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_each_mode_has_a_sensible_default(self, mode: type[MatchingSystem]) -> None:
        threshold = mode.between(EST, GT).threshold

        assert threshold.is_uniform
        assert threshold.default == mode.DEFAULT_THRESHOLD
        if mode.HIGHER_IS_BETTER:
            assert 0.0 < threshold.default <= 1.0
        else:
            assert threshold.default > 0.0


class TestRecordedThreshold:
    """The threshold is the one thing a metric could not recover by following indices."""

    def test_every_row_records_the_threshold_it_was_judged_at(
        self,
        box_store: Store,
        labels: LabelRegistry,
    ) -> None:
        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT, threshold=1.5),
            SystemContext(box_store, FRAME, labels=labels),
        )

        assert result.threshold.values.tolist() == [1.5]

    def test_a_per_class_threshold_is_recorded_per_row(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(
            GT,
            boxes([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], labels, ["car", "pedestrian"]),
            at=TimePoint.at(frame=0),
        )
        store.log(
            EST,
            boxes([[10.1, 0.0, 0.0], [20.1, 0.0, 0.0]], labels, ["car", "pedestrian"]),
            at=TimePoint.at(frame=0),
        )
        system = CenterDistanceMatchingSystem.between(
            EST,
            GT,
            threshold=Thresholds(1.0, by_class={"car": 2.0, "pedestrian": 0.5}),
        )

        result = result_of(system, SystemContext(store, FRAME, labels=labels))

        by_class = dict(
            zip(result.gt_index.values.tolist(), result.threshold.values.tolist()),
        )
        assert by_class == {0: 2.0, 1: 0.5}

    def test_a_missed_object_records_its_own_class_threshold(
        self,
        labels: LabelRegistry,
    ) -> None:
        store = Store()
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels, ["pedestrian"]), at=TimePoint.at(frame=0))
        store.log(EST, boxes([[500.0, 0.0, 0.0]], labels, ["car"]), at=TimePoint.at(frame=0))
        system = CenterDistanceMatchingSystem.between(
            EST,
            GT,
            threshold=Thresholds(1.0, by_class={"car": 2.0, "pedestrian": 0.5}),
        )

        result = result_of(system, SystemContext(store, FRAME, labels=labels))

        # The false positive has no ground truth, so the estimation's class decides;
        # the false negative is judged by its own.
        rows = dict(
            zip(result.match_status.values.tolist(), result.threshold.values.tolist()),
        )
        assert rows[int(MatchStatus.FP)] == 2.0
        assert rows[int(MatchStatus.FN)] == 0.5

    @pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.__name__)
    def test_every_mode_records_its_threshold(
        self,
        mode: type[MatchingSystem],
        labels: LabelRegistry,
    ) -> None:
        store = store_for(mode, labels)

        result = result_of(
            mode.between(EST, GT),
            SystemContext(store, FRAME, labels=labels),
        )

        assert result.threshold.values.tolist() == [mode.DEFAULT_THRESHOLD]


class TestPerClassThresholds:
    def test_a_class_keeps_its_own_tolerance(self, labels: LabelRegistry) -> None:
        """Both objects are 1.5 m out; only the class allowed 2 m matches."""
        store = Store()
        store.log(
            GT,
            boxes([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], labels, ["car", "pedestrian"]),
            at=TimePoint.at(frame=0),
        )
        store.log(
            EST,
            boxes([[11.5, 0.0, 0.0], [21.5, 0.0, 0.0]], labels, ["car", "pedestrian"]),
            at=TimePoint.at(frame=0),
        )
        ctx = SystemContext(store, FRAME, labels=labels)

        per_class = CenterDistanceMatchingSystem.between(
            EST,
            GT,
            threshold=Thresholds(1.0, by_class={"car": 2.0}),
        )
        uniform = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)

        assert counts(result_of(per_class, ctx)) == (1, 1, 1)
        assert counts(result_of(uniform, ctx)) == (0, 2, 2)

    def test_the_ground_truth_class_selects_the_threshold(self, labels: LabelRegistry) -> None:
        """With class-agnostic matching the two sides can disagree, so one must decide."""
        store = Store()
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels, ["car"]), at=TimePoint.at(frame=0))
        store.log(EST, boxes([[11.5, 0.0, 0.0]], labels, ["pedestrian"]), at=TimePoint.at(frame=0))
        ctx = SystemContext(store, FRAME, labels=labels)

        system = CenterDistanceMatchingSystem.between(
            EST,
            GT,
            threshold=Thresholds(1.0, by_class={"car": 2.0}),
            class_agnostic=True,
        )

        # The ground truth is a car, so the 2 m tolerance applies despite the estimation
        # being labelled a pedestrian.
        assert counts(result_of(system, ctx)) == (1, 0, 0)

    def test_names_need_a_registry(self, box_store: Store) -> None:
        system = CenterDistanceMatchingSystem.between(
            EST,
            GT,
            threshold=Thresholds(1.0, by_class={"car": 2.0}),
        )

        with pytest.raises(ValueError, match="require a LabelRegistry"):
            list(system(SystemContext(box_store, FRAME), 0))


class TestAssignment:
    def test_the_assignment_is_globally_optimal(self, labels: LabelRegistry) -> None:
        """A greedy matcher would pair est0 with gt0 and then fail to place est1."""
        store = Store()
        store.log(
            GT,
            boxes([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], labels),
            at=TimePoint.at(frame=0),
        )
        store.log(
            EST,
            boxes([[0.4, 0.0, 0.0], [0.9, 0.0, 0.0]], labels),
            at=TimePoint.at(frame=0),
        )

        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT, threshold=0.5),
            SystemContext(store, FRAME, labels=labels),
        )

        assert result.num_tp == 2
        assert sorted(result.est_index.values.tolist()) == [0, 1]

    def test_it_is_one_to_one(self, labels: LabelRegistry) -> None:
        """Two estimations on one ground truth: the better one wins, the other is an FP."""
        store = Store()
        store.log(GT, boxes([[0.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(
            EST,
            boxes([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], labels),
            at=TimePoint.at(frame=0),
        )

        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0),
            SystemContext(store, FRAME, labels=labels),
        )

        assert counts(result) == (1, 1, 0)
        matched = result.match_status.values == int(MatchStatus.TP)
        assert result.matching_score.values[matched] == pytest.approx(0.1)

    def test_higher_is_better_maximizes_the_total(self, labels: LabelRegistry) -> None:
        """IoU is maximized, not minimized, so the pairing must not invert."""
        store = Store()
        store.log(
            GT,
            boxes([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], labels),
            at=TimePoint.at(frame=0),
        )
        store.log(
            EST,
            boxes([[0.2, 0.0, 0.0], [4.2, 0.0, 0.0]], labels),
            at=TimePoint.at(frame=0),
        )

        result = result_of(
            IoUBEVMatchingSystem.between(EST, GT, threshold=0.3),
            SystemContext(store, FRAME, labels=labels),
        )

        assert result.num_tp == 2
        pairs = set(zip(result.est_index.values.tolist(), result.gt_index.values.tolist()))
        assert pairs == {(0, 0), (1, 1)}

    def test_indices_are_row_numbers_within_the_frame(self, labels: LabelRegistry) -> None:
        store = Store()
        for frame in (0, 1):
            store.log(GT, boxes([[0.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=frame))
            store.log(
                EST,
                boxes([[9.0, 0.0, 0.0], [0.1, 0.0, 0.0]], labels),
                at=TimePoint.at(frame=frame),
            )

        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0),
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        matched = result.match_status.values == int(MatchStatus.TP)
        assert result.est_index.values[matched].tolist() == [1, 1], "row 1 of each frame"

    def test_a_non_finite_position_never_matches(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(GT, boxes([[0.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(EST, boxes([[np.nan, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))

        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT),
            SystemContext(store, FRAME, labels=labels),
        )

        assert counts(result) == (0, 1, 1)


class TestFrameShapes:
    """How a frame's row counts turn into partitions and verdicts."""

    def test_a_multi_frame_scene_is_counted_as_a_whole(
        self,
        scene_store: Store,
        labels: LabelRegistry,
    ) -> None:
        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0),
            SystemContext(scene_store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        # Frame 0: one hit, one far estimation, one missed ground truth.
        # Frame 1: one hit, one estimation of the wrong class.
        assert counts(result) == (2, 2, 1)

    def test_records_the_distance_of_every_hit(
        self,
        scene_store: Store,
        labels: LabelRegistry,
    ) -> None:
        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0),
            SystemContext(scene_store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        hits = result.match_status.values == int(MatchStatus.TP)
        np.testing.assert_allclose(sorted(result.matching_score.values[hits]), [0.1, 0.3])

    def test_a_frame_with_only_estimations_is_all_false_positives(
        self,
        labels: LabelRegistry,
    ) -> None:
        store = Store()
        store.log(EST, boxes([[0.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))

        result = result_of(
            CenterDistanceMatchingSystem.between(EST, GT),
            SystemContext(store, FRAME, labels=labels),
        )

        assert counts(result) == (0, 1, 0)

    def test_an_empty_frame_produces_an_empty_partition(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(EST, make_detections([]), at=TimePoint.at(frame=0))
        store.log(GT, make_detections([]), at=TimePoint.at(frame=0))

        (chunk,) = CenterDistanceMatchingSystem.between(EST, GT)(
            SystemContext(store, FRAME, labels=labels),
            0,
        )

        assert chunk.num_rows == 0
        assert chunk.num_partitions == 1

    def test_a_frame_present_in_only_one_stream_still_produces_a_partition(
        self,
        labels: LabelRegistry,
    ) -> None:
        store = Store()
        store.log(GT, boxes([[0.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(EST, boxes([[0.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=1))

        (chunk,) = CenterDistanceMatchingSystem.between(EST, GT)(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        assert chunk.num_partitions == 2
        assert chunk.index(FRAME).times.tolist() == [0, 1]
        result = MatchResults.from_chunk(chunk)
        assert counts(result) == (0, 1, 1), "one FN in frame 0, one FP in frame 1"


class TestModeScores:
    """Each mode's score, on a pair whose expected value is known by hand."""

    def score(self, system: MatchingSystem, store: Store, labels: LabelRegistry) -> float:
        result = result_of(system, SystemContext(store, FRAME, labels=labels))
        matched = result.match_status.values == int(MatchStatus.TP)
        return float(result.matching_score.values[matched][0])

    def test_centre_distance_is_the_3d_gap(self, box_store: Store, labels: LabelRegistry) -> None:
        system = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)

        assert self.score(system, box_store, labels) == pytest.approx(0.5)

    def test_centre_distance_bev_ignores_z(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(EST, boxes([[10.5, 0.0, 3.0]], labels), at=TimePoint.at(frame=0))

        bev = CenterDistanceBEVMatchingSystem.between(EST, GT, threshold=1.0)
        full = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)

        assert self.score(bev, store, labels) == pytest.approx(0.5)
        assert counts(result_of(full, SystemContext(store, FRAME, labels=labels))) == (0, 1, 1)

    def test_plane_distance_scores_the_near_face(
        self,
        box_store: Store,
        labels: LabelRegistry,
    ) -> None:
        system = PlaneDistanceMatchingSystem.between(EST, GT, threshold=1.0)

        assert self.score(system, box_store, labels) == pytest.approx(0.5)

    def test_plane_distance_forgives_a_far_face_error(self, labels: LabelRegistry) -> None:
        """A 2 m longer box recentred so its near face coincides scores 0."""
        store = Store()
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(
            EST,
            boxes([[11.0, 0.0, 0.0]], labels, sizes=[[2.0, 6.0, 2.0]]),
            at=TimePoint.at(frame=0),
        )
        ctx = SystemContext(store, FRAME, labels=labels)

        plane = PlaneDistanceMatchingSystem.between(EST, GT, threshold=0.1)
        centre = CenterDistanceMatchingSystem.between(EST, GT, threshold=0.1)

        assert self.score(plane, store, labels) == pytest.approx(0.0)
        assert counts(result_of(centre, ctx)) == (0, 1, 1), "centre distance rejects it"

    def test_bev_iou_uses_the_footprint(self, box_store: Store, labels: LabelRegistry) -> None:
        # 4-long boxes offset by 0.5: intersection 3.5 * 2 = 7, union 8 + 8 - 7 = 9.
        system = IoUBEVMatchingSystem.between(EST, GT, threshold=0.5)

        assert self.score(system, box_store, labels) == pytest.approx(7.0 / 9.0)

    def test_3d_iou_also_accounts_for_height(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(
            EST,
            boxes([[10.0, 0.0, 0.0]], labels, sizes=[[2.0, 4.0, 1.0]]),
            at=TimePoint.at(frame=0),
        )

        bev = IoUBEVMatchingSystem.between(EST, GT, threshold=0.4)
        volumetric = IoU3DMatchingSystem.between(EST, GT, threshold=0.4)

        assert self.score(bev, store, labels) == pytest.approx(1.0)
        assert self.score(volumetric, store, labels) == pytest.approx(0.5)

    def test_roi_iou_uses_the_image_plane(self, roi_store: Store, labels: LabelRegistry) -> None:
        # 10x10 ROIs offset by 2 px: intersection 8 * 10 = 80, union 200 - 80 = 120.
        system = IoURoiMatchingSystem.between(EST, GT, threshold=0.6)

        assert self.score(system, roi_store, labels) == pytest.approx(80.0 / 120.0)

    def test_rotated_footprints_need_polygon_clipping(self, labels: LabelRegistry) -> None:
        """An axis-aligned overlap would call these a perfect match."""
        store = Store()
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(
            EST,
            boxes([[10.0, 0.0, 0.0]], labels, yaws=[np.pi / 2]),
            at=TimePoint.at(frame=0),
        )

        system = IoUBEVMatchingSystem.between(EST, GT, threshold=0.3)

        # Crossed rectangles overlap on a 2x2 square: 4 / (8 + 8 - 4).
        assert self.score(system, store, labels) == pytest.approx(4.0 / 12.0)


class TestPipelineIntegration:
    def test_several_modes_run_over_the_same_frame(
        self,
        box_store: Store,
        labels: LabelRegistry,
    ) -> None:
        systems = [mode.between(EST, GT) for mode in BOX_MODES]
        ctx = SystemContext(box_store, FRAME, labels=labels)

        produced = Pipeline(systems).run(ctx, TimeRange.everything())

        assert len(produced) == len(BOX_MODES)
        for system in systems:
            result = box_store.range(
                system.target,
                timeline=FRAME,
                time_range=TimeRange.everything(),
            ).materialize(MatchResults)
            assert counts(result) == (1, 0, 0), f"{type(system).__name__} disagreed"

    def test_a_filter_can_feed_a_matcher(self, labels: LabelRegistry) -> None:
        from t4perceval.descriptors import MASK
        from t4perceval.system import FilterByDistanceSystem, masked_view

        store = Store()
        store.log(GT, boxes([[10.0, 0.0, 0.0]], labels), at=TimePoint.at(frame=0))
        store.log(
            EST,
            boxes([[10.5, 0.0, 0.0], [500.0, 0.0, 0.0]], labels),
            at=TimePoint.at(frame=0),
        )

        near = FilterByDistanceSystem.on(EST, max_distance=100.0)
        matcher = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        ctx = SystemContext(store, FRAME, labels=labels)

        Pipeline([near, matcher]).run(ctx, TimeRange.everything())

        mask = store.range(
            near.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).component(MASK)
        assert mask.values.tolist() == [True, False]

        # The matcher saw the unfiltered stream, so the far estimation is an FP there.
        result = store.range(
            matcher.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MatchResults)
        assert counts(result) == (1, 1, 0)

        # The mask still says which FP the filter would have removed.
        kept = masked_view(
            store,
            EST,
            near.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        )
        assert len(kept) == 1

    def test_a_result_survives_a_parquet_round_trip(
        self,
        box_store: Store,
        labels: LabelRegistry,
        tmp_path: object,
    ) -> None:
        from t4perceval.io import read_parquet, write_parquet

        system = PlaneDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        ctx = SystemContext(box_store, FRAME, labels=labels)
        Pipeline([system]).run(ctx, TimeRange.everything())

        chunk = box_store.range(
            system.target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).to_chunk()
        path = tmp_path / "plane.parquet"  # type: ignore[operator]
        write_parquet(chunk, path, labels=labels)
        restored, restored_labels = read_parquet(path)

        assert restored == chunk
        assert restored_labels == labels
        assert counts(MatchResults.from_chunk(restored)) == (1, 0, 0)
