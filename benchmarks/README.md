# Benchmarks

`compare.py` feeds `autoware_perception_evaluation` (`perception_eval` 1.3.6) and `t4perceval` the
same synthetic scenes and compares both speed and the metric values themselves. The report is
written to [`results/latest.md`](results/latest.md) and `results/latest.json`; the headline numbers
are copied into the top-level [README](../README.md#benchmark).

## Running it

From the repository root:

```bash
uv run python benchmarks/compare.py --check                          # full run, a few minutes
uv run python benchmarks/compare.py --sizes 10,50 --repeats 2 --check  # quick iteration
uv run python benchmarks/compare.py --skip-perf --check              # agreement only, seconds
uv run python benchmarks/compare.py --skip-agreement                 # timing only
uv run python benchmarks/compare.py --output-dir /tmp/bench          # write elsewhere
```

`--check` exits non-zero when a metric value differs from `perception_eval` in a way that is not a
documented divergence. Other options: `--frames` (10), `--matching-repeats` (15), `--warmups` (2),
`--memory-objects` (20000), `--agreement-objects` (50), `--seed` (0); `--help` lists them all.

## What a run does

The coordinator generates every scene once and writes it to a temporary directory as `.npz`, then
launches one worker per library over the same files, so "the same input" is a file identity rather
than trust that two NumPy majors draw identical random streams.

- The `t4perceval` worker uses the project environment.
- The `perception_eval` worker is started with
  `uv run --no-project --with "perception_eval @ git+...@v1.3.6"`, because that package pins NumPy 1.
  On the first run this downloads and builds the environment into `.cache/uv-benchmark`
  (gitignored) and reuses it afterwards.

Each worker pins itself to one CPU and, per scene size, times five phases -- construction (NumPy
arrays to one frame of the library's objects), single-frame center-distance matching, detection
(mAP + mAPH at four thresholds), tracking (CLEAR) and prediction (ADE / FDE / miss rate for top-k 1
and 3) -- plus the RSS retained by 20,000 + 20,000 detection objects. It then computes the same
metrics for two agreement scenes and prints one JSON line, which the coordinator merges.

## Scenes

`bench/scenes.py` builds two regimes with a seeded generator:

- **unambiguous** -- ground truths sit on a jittered grid whose spacing exceeds twice the largest
  matching threshold plus the largest estimation offset, and every estimate lies within that offset
  of its own ground truth. The feasible graph has degree at most one, so `perception_eval`'s greedy
  assignment and `t4perceval`'s linear-sum assignment pick the same pairs and every metric is
  expected to agree (tolerance `1e-9`; `1e-8` for APH, which `perception_eval` rounds).
- **dense** -- random positions, per-frame noise, label confusion and missed frames. Used for the
  timing tables and for the divergence report, where each difference is classified against a
  documented item in [`docs/development/metric-divergences.md`](../docs/development/metric-divergences.md); anything unclassified is
  a mismatch and fails `--check`.

Both sides of a scene share their frame indices by construction (frame `i` of the estimation is
frame `i` of the ground truth, stamped at `i × 500 ms`), so `t4perceval.align` is not part of
the benchmark: perception_eval's manager-free scoring has no counterpart to pair against, and its
own association (`get_ground_truth_now_frame`, estimation-driven, many-to-one) is a different rule.

## Layout

| File                            | Role                                                                      |
| :------------------------------ | :------------------------------------------------------------------------ |
| `compare.py`                    | CLI, coordinator, and the hidden `--worker` entry both processes run      |
| `bench/harness.py`              | timing, RSS, CPU pinning, worker launch                                   |
| `bench/scenes.py`               | scene generator and `.npz` round-trip                                     |
| `bench/side_t4perceval.py`      | build a `Store`, run the pipelines, read `MetricValues`                   |
| `bench/side_perception_eval.py` | build `DynamicObject`s, run the manager-free scoring, read `MetricsScore` |
| `bench/report.py`               | tolerances, documented divergences, comparison rows, markdown             |

Neither adapter imports its library at module scope; that is what lets one script run under two
incompatible NumPy majors.

## Prerequisites

- `uv` on the path, and network access on the first run.
- A Python 3.10 or 3.11 interpreter available to `uv` for the `perception_eval` worker, since
  `perception_eval` 1.3.6 supports nothing newer.
- The benchmark is not part of `pytest` and is never run in CI; `benchmarks/` is only linted there.
