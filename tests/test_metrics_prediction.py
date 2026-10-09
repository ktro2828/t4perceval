"""Prediction metrics: ADE, FDE and miss rate."""

from __future__ import annotations

import numpy as np
import pytest

from t4perceval import (
    FRAME,
    MetricValues,
    Predictions3D,
    LabelRegistry,
    Store,
    TimePoint,
    TimeRange,
)
from t4perceval.system import (
    CenterDistanceMatchingSystem,
    PathDisplacementSystem,
    Pipeline,
    SystemContext,
)

EST = "/estimation/objects"
GT = "/ground_truth/objects"

#: A ground truth that carries straight on: x = 1, 2, 3 with y = 0.
STRAIGHT_AHEAD = [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]

#: One timestep, in nanoseconds: every helper trajectory is sampled every 100 ms by default.
STEP_NS = 100_000_000


def steps_ns(count: int, step_ns: int = STEP_NS) -> list[int]:
    """Return the offsets of ``count`` timesteps, ``step_ns`` apart, starting one step ahead."""
    return [step_ns * (index + 1) for index in range(count)]


def prediction(
    x: float,
    waypoints: list[list[list[float]]],
    confidences: list[float],
    labels: LabelRegistry,
    name: str = "car",
    *,
    time_offset: list[int] | None = None,
    mode_valid: list[bool] | None = None,
    timestep_valid: list[list[bool]] | None = None,
) -> Predictions3D:
    """One object whose predicted futures are ``waypoints``, shaped ``(M, T, 3)``."""
    array = np.asarray(waypoints, dtype=np.float64)
    optional: dict[str, np.ndarray] = {}
    if mode_valid is not None:
        optional["mode_valid"] = np.asarray([mode_valid], dtype=np.bool_)
    if timestep_valid is not None:
        optional["timestep_valid"] = np.asarray([timestep_valid], dtype=np.bool_)
    return Predictions3D(
        position=[[x, 0.0, 0.0]],
        quaternion=[[0.0, 0.0, 0.0, 1.0]],
        size=[[2.0, 4.0, 2.0]],
        class_id=labels.encode([name]),
        confidence=[0.9],
        instance_id=[1],
        waypoints=array[None, ...],
        mode_confidence=np.asarray([confidences], dtype=np.float64),
        time_offset=np.asarray([time_offset or steps_ns(array.shape[1])], dtype=np.int64),
        **optional,
    )


def displacement(
    labels: LabelRegistry,
    *,
    est_waypoints: list[list[list[float]]],
    est_confidences: list[float],
    gt_waypoints: list[list[list[float]]] | None = None,
    est_x: float = 0.05,
    est_options: dict[str, object] | None = None,
    gt_options: dict[str, object] | None = None,
    **params: object,
) -> dict[str, MetricValues]:
    """Score one object, keyed by metric name (``"ade"``) when one ``top_k`` is given and by
    entity name (``"ade3"``) when several are."""
    store = Store()
    store.log(
        GT,
        prediction(0.0, gt_waypoints or STRAIGHT_AHEAD, [1.0], labels, **(gt_options or {})),  # type: ignore[arg-type]
        at=TimePoint.at(frame=0),
        frame_id="base_link",
    )
    store.log(
        EST,
        prediction(est_x, est_waypoints, est_confidences, labels, **(est_options or {})),  # type: ignore[arg-type]
        at=TimePoint.at(frame=0),
        frame_id="base_link",
    )

    match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
    metric = PathDisplacementSystem.on(match.target, EST, GT, **params)
    Pipeline([match, metric]).run(
        SystemContext(store, FRAME, labels=labels), TimeRange.everything()
    )

    suffix = str(metric.top_k[0]) if len(metric.top_k) == 1 else ""
    return {
        target.name.removesuffix(suffix): store.range(
            target,
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        for target in metric.targets
    }


#: Mode 0 stays 1 m off; mode 1 stays 3 m off. Mode 1 is the more confident.
TWO_MODES = [
    [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
    [[1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
]
TWO_MODE_CONFIDENCES = [0.3, 0.7]


class TestTargets:
    def test_writes_one_entity_per_metric_suffixed_by_k(self) -> None:
        metric = PathDisplacementSystem.on("/m", EST, GT)

        assert [str(target) for target in metric.targets] == [
            "/metrics/path_displacement/ade3",
            "/metrics/path_displacement/fde3",
            "/metrics/path_displacement/miss_rate3",
        ]

    def test_writes_one_entity_per_metric_and_k(self) -> None:
        metric = PathDisplacementSystem.on("/m", EST, GT, top_k=(1, 3))

        assert [target.name for target in metric.targets] == [
            "ade1",
            "ade3",
            "fde1",
            "fde3",
            "miss_rate1",
            "miss_rate3",
        ]

    def test_best_of_k_names_its_minima_apart(self) -> None:
        """minADE_k / minFDE_k are other numbers than the mean; the miss rate keeps its name."""
        metric = PathDisplacementSystem.on("/m", EST, GT, top_k=(1, 3), best_of_k=True)

        assert [target.name for target in metric.targets] == [
            "min_ade1",
            "min_ade3",
            "min_fde1",
            "min_fde3",
            "miss_rate1",
            "miss_rate3",
        ]


class TestBestOfK:
    def test_by_default_every_kept_mode_counts(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
        )

        # The mean over both modes: (1 + 3) / 2. This is an average over modes, not the
        # best-of-k that some benchmarks report.
        assert metrics["ade"].of_class(0) == pytest.approx(2.0)
        assert metrics["fde"].of_class(0) == pytest.approx(2.0)

    def test_best_of_k_takes_the_best_mode(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            best_of_k=True,
        )

        assert metrics["min_ade"].of_class(0) == pytest.approx(1.0)
        assert metrics["min_fde"].of_class(0) == pytest.approx(1.0)

    def test_min_fde_picks_its_own_mode(self, labels: LabelRegistry) -> None:
        """minADE_k and minFDE_k are separate minima, as nuScenes and Waymo define them."""
        on_average = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 2.0, 0.0]]  # ADE 2/3, FDE 2
        at_the_end = [[1.0, 1.5, 0.0], [2.0, 1.5, 0.0], [3.0, 0.5, 0.0]]  # ADE 7/6, FDE 0.5

        metrics = displacement(
            labels,
            est_waypoints=[on_average, at_the_end],
            est_confidences=[0.5, 0.5],
            best_of_k=True,
        )

        assert metrics["min_ade"].of_class(0) == pytest.approx(2.0 / 3.0)
        assert metrics["min_fde"].of_class(0) == pytest.approx(0.5)
        # The miss rate is counted on the minADE mode: only its 2 m final step misses.
        assert metrics["miss_rate"].of_class(0) == pytest.approx(1.0 / 3.0)

    def test_the_most_confident_mode_alone_is_top_k_1(self, labels: LabelRegistry) -> None:
        """Mode 1 is the confident one, and it is also the worse one."""
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            top_k=1,
        )

        assert metrics["ade"].of_class(0) == pytest.approx(3.0)

    def test_the_old_kernel_parameter_is_gone(self) -> None:
        with pytest.raises(TypeError, match="kernel"):
            PathDisplacementSystem.on("/m", EST, GT, kernel="min")


class TestTopK:
    def test_keeps_only_the_most_confident_modes(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            top_k=1,
        )

        assert metrics["ade"].of_class(0) == pytest.approx(3.0), "the confident mode alone"

    def test_asking_for_more_modes_than_exist_is_harmless(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            top_k=9,
        )

        assert metrics["ade"].of_class(0) == pytest.approx(2.0)

    def test_rejects_a_non_positive_top_k(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            PathDisplacementSystem.on("/m", EST, GT, top_k=0)
        with pytest.raises(ValueError, match="at least 1"):
            PathDisplacementSystem.on("/m", EST, GT, top_k=(3, 0))

    def test_a_single_value_becomes_a_tuple(self) -> None:
        assert PathDisplacementSystem.on("/m", EST, GT, top_k=2).top_k == (2,)

    def test_rejects_an_empty_top_k(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            PathDisplacementSystem.on("/m", EST, GT, top_k=())

    def test_rejects_a_repeated_top_k(self) -> None:
        """Both would write the same entities."""
        with pytest.raises(ValueError, match="distinct"):
            PathDisplacementSystem.on("/m", EST, GT, top_k=(3, 3))

    def test_several_values_are_scored_in_one_pass(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            top_k=(1, 2),
        )

        assert metrics["ade1"].of_class(0) == pytest.approx(3.0), "the confident mode alone"
        assert metrics["ade2"].of_class(0) == pytest.approx(2.0), "(1 + 3) / 2"

    @pytest.mark.parametrize(
        ("best_of_k", "expected_ade"),
        [
            # Per object: all modes real / the 3 m mode padded / no real mode (1.95).
            (
                False,
                [
                    (3 + 4 / 3 + 1.95) / 3,
                    (13 / 6 + 11 / 12 + 1.95) / 3,
                    (29 / 18 + 11 / 12 + 1.95) / 3,
                ],
            ),
            (True, [(3 + 4 / 3 + 1.95) / 3, (4 / 3 + 0.5 + 1.95) / 3, (0.5 + 0.5 + 1.95) / 3]),
        ],
    )
    def test_several_values_agree_with_one_system_per_value(
        self,
        labels: LabelRegistry,
        best_of_k: bool,
        expected_ade: list[float],
    ) -> None:
        """Sharing the pass at the largest k must not change what a smaller k scores.

        Three objects stress the slicing: one with more real modes than the smaller k
        keeps, one whose padded mode leaves fewer candidates than k, and one with no real
        mode at all, which is scored as standing still. By confidence the modes rank 3 m
        off (ADE 3, FDE 3), then a swerve (ADE 4/3, FDE 0), then 0.5 m off.
        """
        store = Store()
        modes = [
            [[1.0, 0.5, 0.0], [2.0, 0.5, 0.0], [3.0, 0.5, 0.0]],
            [[1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
            [[1.0, 1.5, 0.0], [2.0, 2.5, 0.0], [3.0, 0.0, 0.0]],
        ]
        mode_valid = [[True, True, True], [True, False, True], [False, False, False]]
        num = len(mode_valid)
        common = {
            "quaternion": [[0.0, 0.0, 0.0, 1.0]] * num,
            "size": [[2.0, 4.0, 2.0]] * num,
            "class_id": labels.encode(["car"] * num),
            "confidence": [0.9] * num,
            "instance_id": list(range(num)),
            "time_offset": np.asarray([steps_ns(3)] * num, dtype=np.int64),
        }
        xs = [0.0, 20.0, 40.0]
        store.log(
            GT,
            Predictions3D(
                position=[[x, 0.0, 0.0] for x in xs],
                waypoints=np.asarray(
                    [[[[x + 1.0, 0.0, 0.0], [x + 2.0, 0.0, 0.0], [x + 3.0, 0.0, 0.0]]] for x in xs]
                ),
                mode_confidence=np.ones((num, 1)),
                **common,
            ),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        store.log(
            EST,
            Predictions3D(
                position=[[x + 0.05, 0.0, 0.0] for x in xs],
                waypoints=np.asarray(
                    [[[[x + p[0], p[1], p[2]] for p in m] for m in modes] for x in xs]
                ),
                mode_confidence=np.asarray([[0.2, 0.5, 0.3]] * num),
                mode_valid=np.asarray(mode_valid, dtype=np.bool_),
                **common,
            ),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )

        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        shared = PathDisplacementSystem.on(
            match.target, EST, GT, top_k=(1, 2, 3), best_of_k=best_of_k
        )
        alone = [
            PathDisplacementSystem.on(
                match.target, EST, GT, top_k=k, best_of_k=best_of_k, target=f"/alone/{k}"
            )
            for k in (1, 2, 3)
        ]
        Pipeline([match, shared, *alone]).run(
            SystemContext(store, FRAME, labels=labels), TimeRange.everything()
        )

        def read(target: object) -> float:
            return (
                store.range(
                    target,  # type: ignore[arg-type]
                    timeline=FRAME,
                    time_range=TimeRange.everything(),
                )
                .materialize(MetricValues)
                .of_class(0)
            )

        separate = [target for system in alone for target in system.targets]
        expected = {target.name: read(target) for target in separate}
        actual = {target.name: read(target) for target in shared.targets}
        assert actual == pytest.approx(expected)
        ade = "min_ade" if best_of_k else "ade"
        assert [actual[f"{ade}{k}"] for k in (1, 2, 3)] == pytest.approx(expected_ade)


class TestHorizonAlignment:
    def test_a_short_prediction_is_held_at_its_last_state(self, labels: LabelRegistry) -> None:
        """Predicting less far ahead is penalised for the gap, not excused from it."""
        short = [[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]]

        metrics = displacement(labels, est_waypoints=short, est_confidences=[1.0])

        # Steps 1 and 2 are 1 m out; the held step 3 sits at (2, 1) against (3, 0).
        assert metrics["ade"].of_class(0) == pytest.approx((1.0 + 1.0 + np.sqrt(2.0)) / 3.0)
        assert metrics["fde"].of_class(0) == pytest.approx(np.sqrt(2.0))

    def test_a_long_prediction_is_truncated(self, labels: LabelRegistry) -> None:
        long = [
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [4.0, 99.0, 0.0],
            ],
        ]

        metrics = displacement(labels, est_waypoints=long, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(1.0), "the fourth step is ignored"


class TestValues:
    def test_a_perfect_prediction_scores_zero(self, labels: LabelRegistry) -> None:
        metrics = displacement(labels, est_waypoints=STRAIGHT_AHEAD, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(0.0)
        assert metrics["fde"].of_class(0) == pytest.approx(0.0)
        assert metrics["miss_rate"].of_class(0) == pytest.approx(0.0)

    def test_fde_looks_only_at_the_final_step(self, labels: LabelRegistry) -> None:
        drifting = [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 5.0, 0.0]]]

        metrics = displacement(labels, est_waypoints=drifting, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(5.0 / 3.0)
        assert metrics["fde"].of_class(0) == pytest.approx(5.0)

    def test_the_error_ignores_the_z_axis(self, labels: LabelRegistry) -> None:
        lifted = [[[1.0, 0.0, 9.0], [2.0, 0.0, 9.0], [3.0, 0.0, 9.0]]]

        metrics = displacement(labels, est_waypoints=lifted, est_confidences=[1.0])

        assert metrics["ade"].of_class(0) == pytest.approx(0.0)

    def test_miss_rate_counts_the_steps_beyond_tolerance(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
        )

        # Mode 0 is 1 m out and within tolerance; mode 1 is 3 m out and beyond it.
        assert metrics["miss_rate"].of_class(0) == pytest.approx(0.5)

    def test_the_tolerance_is_configurable(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            miss_tolerance=0.5,
        )

        assert metrics["miss_rate"].of_class(0) == pytest.approx(1.0)

    def test_rejects_a_non_positive_tolerance(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            PathDisplacementSystem.on("/m", EST, GT, miss_tolerance=0.0)


class TestValidity:
    """Padding is masked, never scored."""

    def test_an_invalid_mode_never_competes_for_top_k(self, labels: LabelRegistry) -> None:
        # The padded third mode claims the highest confidence, but is not a real mode.
        metrics = displacement(
            labels,
            est_waypoints=[*TWO_MODES, STRAIGHT_AHEAD[0]],
            est_confidences=[*TWO_MODE_CONFIDENCES, 0.9],
            est_options={"mode_valid": [True, True, False]},
            top_k=1,
        )

        assert metrics["ade"].of_class(0) == pytest.approx(3.0), "the confident *real* mode"

    def test_best_of_k_never_picks_an_invalid_mode(self, labels: LabelRegistry) -> None:
        # The padded mode lies exactly on the ground truth; picking it would score zero.
        metrics = displacement(
            labels,
            est_waypoints=[*TWO_MODES, STRAIGHT_AHEAD[0]],
            est_confidences=[*TWO_MODE_CONFIDENCES, 0.0],
            est_options={"mode_valid": [True, True, False]},
            best_of_k=True,
        )

        assert metrics["min_ade"].of_class(0) == pytest.approx(1.0)

    def test_a_mode_with_no_valid_step_is_not_a_candidate(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=[*TWO_MODES, STRAIGHT_AHEAD[0]],
            est_confidences=[*TWO_MODE_CONFIDENCES, 0.0],
            est_options={
                "timestep_valid": [[True] * 3, [True] * 3, [False] * 3],
            },
            best_of_k=True,
        )

        assert metrics["min_ade"].of_class(0) == pytest.approx(1.0)

    def test_invalid_estimation_steps_are_interpolated_across(self, labels: LabelRegistry) -> None:
        holey = [[[1.0, 0.0, 0.0], [2.0, 50.0, 0.0], [3.0, 0.0, 0.0]]]

        metrics = displacement(
            labels,
            est_waypoints=holey,
            est_confidences=[1.0],
            est_options={"timestep_valid": [[True, False, True]]},
        )

        # Step 2 is read from the line between steps 1 and 3, not from the garbage.
        assert metrics["ade"].of_class(0) == pytest.approx(0.0)

    def test_invalid_ground_truth_steps_are_not_scored(self, labels: LabelRegistry) -> None:
        observed = [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [99.0, 99.0, 0.0]]]
        predicted = [[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 5.0, 0.0]]]

        metrics = displacement(
            labels,
            est_waypoints=predicted,
            est_confidences=[1.0],
            gt_waypoints=observed,
            gt_options={"timestep_valid": [[True, True, False]]},
        )

        assert metrics["ade"].of_class(0) == pytest.approx(1.0)
        assert metrics["fde"].of_class(0) == pytest.approx(1.0), "taken at the last valid step"
        assert metrics["miss_rate"].of_class(0) == pytest.approx(0.0)

    def test_a_ground_truth_with_no_future_is_left_out(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            gt_options={"timestep_valid": [[False, False, False]]},
        )

        assert np.isnan(metrics["ade"].of_class(0))
        assert metrics["ade"].support.values[0] == 1, "the ground truth still counted"

    @pytest.mark.parametrize(
        "masks",
        [
            {"mode_valid": [False]},
            {"mode_valid": [False], "timestep_valid": [[True, True, True]]},
        ],
        ids=["mode-mask-alone", "mode-mask-overrides-steps"],
    )
    def test_an_invalid_ground_truth_mode_is_left_out(
        self,
        labels: LabelRegistry,
        masks: dict[str, object],
    ) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            gt_options=masks,
        )

        assert np.isnan(metrics["ade"].of_class(0))
        assert np.isnan(metrics["fde"].of_class(0))
        assert np.isnan(metrics["miss_rate"].of_class(0))
        assert metrics["ade"].support.values[0] == 1, "the ground truth still counted"

    def test_an_estimation_with_no_future_stands_still(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            est_options={"mode_valid": [False, False]},
        )

        # Held at its position (0.05, 0) against x = 1, 2, 3: penalised, not excused.
        assert metrics["ade"].of_class(0) == pytest.approx(1.95)
        assert metrics["fde"].of_class(0) == pytest.approx(2.95)


class TestTimeAlignment:
    """The two sides are compared at the same times, not at the same array index."""

    def test_a_coarser_prediction_is_interpolated(self, labels: LabelRegistry) -> None:
        # The ground truth moves at 10 m/s, sampled every 100 ms; the prediction says the
        # same thing every 500 ms. By index the two would be metres apart.
        metrics = displacement(
            labels,
            est_waypoints=[[[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
            est_confidences=[1.0],
            est_x=0.0,
            est_options={"time_offset": steps_ns(2, step_ns=5 * STEP_NS)},
        )

        assert metrics["ade"].of_class(0) == pytest.approx(0.0)
        assert metrics["fde"].of_class(0) == pytest.approx(0.0)

    def test_the_current_position_anchors_offset_zero(self, labels: LabelRegistry) -> None:
        # One waypoint at 300 ms; the steps before it lie between it and the position.
        metrics = displacement(
            labels,
            est_waypoints=[[[3.0, 0.0, 0.0]]],
            est_confidences=[1.0],
            est_x=0.0,
            est_options={"time_offset": [3 * STEP_NS]},
        )

        assert metrics["ade"].of_class(0) == pytest.approx(0.0)

    def test_a_prediction_ending_early_is_held_at_its_last_point(
        self,
        labels: LabelRegistry,
    ) -> None:
        # Right on time until 200 ms, then nothing: the 300 ms step reads (2, 0).
        metrics = displacement(
            labels,
            est_waypoints=[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            est_confidences=[1.0],
        )

        assert metrics["fde"].of_class(0) == pytest.approx(1.0)
        assert metrics["ade"].of_class(0) == pytest.approx(1.0 / 3.0)

    def test_objects_weigh_the_same_whatever_their_horizon(
        self,
        labels: LabelRegistry,
    ) -> None:
        """One object observed for one step 4 m off, one for three steps 1 m off."""

        def two(positions: list[float], waypoints: np.ndarray, valid: list[list[bool]]):  # noqa: ANN202
            return Predictions3D(
                position=[[x, 0.0, 0.0] for x in positions],
                quaternion=[[0.0, 0.0, 0.0, 1.0]] * 2,
                size=[[2.0, 4.0, 2.0]] * 2,
                class_id=labels.encode(["car", "car"]),
                confidence=[0.9, 0.9],
                instance_id=[1, 2],
                waypoints=waypoints,
                mode_confidence=[[1.0], [1.0]],
                timestep_valid=np.asarray(valid, dtype=np.bool_)[:, None, :],
                time_offset=[steps_ns(3), steps_ns(3)],
            )

        observed = np.zeros((2, 1, 3, 3))
        observed[:, 0, :, 0] = [1.0, 2.0, 3.0]
        predicted = observed.copy()
        predicted[0, 0, :, 1] = 4.0
        predicted[1, 0, :, 1] = 1.0
        # The first object's unobserved steps are padded as the importers pad them: held
        # at the last valid point, so scoring them would change the answer.
        observed[0, 0, 1:] = observed[0, 0, 0]

        store = Store()
        store.log(
            GT,
            two([0.0, 50.0], observed, [[True, False, False], [True, True, True]]),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        store.log(
            EST,
            two([0.0, 50.0], predicted, [[True] * 3, [True] * 3]),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = PathDisplacementSystem.on(match.target, EST, GT)
        Pipeline([match, metric]).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )
        ade = store.range(
            metric.targets[0],
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)

        # (4 + 1) / 2 per object, not (4 + 1 + 1 + 1) / 4 over all steps.
        assert ade.of_class(0) == pytest.approx(2.5)

    @pytest.mark.parametrize("side", [EST, GT])
    def test_a_missing_time_axis_is_an_error(self, labels: LabelRegistry, side: str) -> None:
        from attrs import evolve

        store = Store()
        for path, x in ((GT, 0.0), (EST, 0.05)):
            objects = prediction(x, STRAIGHT_AHEAD, [1.0], labels)
            if path == side:
                objects = evolve(objects, time_offset=None)
            store.log(path, objects, at=TimePoint.at(frame=0), frame_id="base_link")

        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0)
        metric = PathDisplacementSystem.on(match.target, EST, GT)
        with pytest.raises(ValueError, match="missing required component.*time_offset"):
            Pipeline([match, metric]).run(
                SystemContext(store, FRAME, labels=labels),
                TimeRange.everything(),
            )


class TestEdges:
    def test_an_unmatched_object_contributes_nothing(self, labels: LabelRegistry) -> None:
        metrics = displacement(
            labels,
            est_waypoints=TWO_MODES,
            est_confidences=TWO_MODE_CONFIDENCES,
            est_x=500.0,
        )

        assert np.isnan(metrics["ade"].of_class(0))
        assert metrics["ade"].support.values[0] == 1, "the ground truth still counted"

    def test_a_class_mismatch_contributes_nothing(self, labels: LabelRegistry) -> None:
        store = Store()
        store.log(
            GT,
            prediction(0.0, STRAIGHT_AHEAD, [1.0], labels, "car"),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        store.log(
            EST,
            prediction(0.05, STRAIGHT_AHEAD, [1.0], labels, "truck"),
            at=TimePoint.at(frame=0),
            frame_id="base_link",
        )
        match = CenterDistanceMatchingSystem.between(EST, GT, threshold=1.0, class_agnostic=True)
        metric = PathDisplacementSystem.on(match.target, EST, GT)
        Pipeline([match, metric]).run(
            SystemContext(store, FRAME, labels=labels),
            TimeRange.everything(),
        )

        ade = store.range(
            metric.targets[0],
            timeline=FRAME,
            time_range=TimeRange.everything(),
        ).materialize(MetricValues)
        assert np.isnan(ade.of_class(labels.class_id("car")))

    def test_an_empty_store_yields_undefined_rows(self, labels: LabelRegistry) -> None:
        metric = PathDisplacementSystem.on("/matching/center_distance", EST, GT)
        store = Store()
        Pipeline([metric]).run(SystemContext(store, FRAME, labels=labels), TimeRange.everything())

        for target in metric.targets:
            row = store.range(
                target,
                timeline=FRAME,
                time_range=TimeRange.everything(),
            ).materialize(MetricValues)
            assert np.isnan(row.value.values).all()

    def test_reports_a_row_for_every_registered_class(self, labels: LabelRegistry) -> None:
        metrics = displacement(labels, est_waypoints=STRAIGHT_AHEAD, est_confidences=[1.0])

        assert metrics["ade"].class_id.values.tolist() == [0, 1, 2]
