# Benchmarks

`benchmarks/compare.py` feeds [`autoware_perception_evaluation`](https://github.com/tier4/autoware_perception_evaluation)
(`perception_eval` 1.3.6) and `t4perceval` **the same synthetic scenes** and compares both speed and
the metric values themselves.

The full report is regenerated into
[`benchmarks/results/latest.md`](https://github.com/ktro2828/t4perceval/blob/main/benchmarks/results/latest.md);
see [`benchmarks/README.md`](https://github.com/ktro2828/t4perceval/blob/main/benchmarks/README.md)
for every option and the prerequisites.

```console
uv run python benchmarks/compare.py --check
```

## Results

Scene phases run over 10 frames with 200 objects per frame (median of 5 runs after 2 warm-ups, one
pinned logical CPU); matching is a single frame.

| Workload (200 objects / frame)               | `perception_eval` | `t4perceval` |   Improvement |
| :------------------------------------------- | ----------------: | -----------: | ------------: |
| Data-model construction, one frame           |          6.035 ms |     0.069 ms |  87.2x faster |
| Center-distance matching, one frame          |        339.900 ms |     1.517 ms | 224.1x faster |
| Detection: mAP + mAPH at 4 thresholds        |       6360.713 ms |    94.074 ms |  67.6x faster |
| Tracking: CLEAR (MOTA / MOTP / ID switches)  |       3303.261 ms |    19.573 ms | 168.8x faster |
| Prediction: ADE / FDE / miss rate, top-k 1,3 |       3718.902 ms |    30.701 ms | 121.1x faster |
| Retained RSS, 20,000 est / 20,000 GT         |          83.7 MiB |      5.2 MiB | 16.0x smaller |

Construction is the [columnar data model](../concepts/data-model.md); the rest is what vectorizing
the per-object Python loops buys.

## Numerical agreement

Speed is only interesting if the numbers agree, so the benchmark checks that too, on two scenes.

**Unambiguous scene.** Ground truths sit on a jittered grid whose spacing exceeds twice the largest
matching threshold, and every estimate lies within that offset of its own ground truth. The feasible
graph has degree at most one, so `perception_eval`'s greedy assignment and `t4perceval`'s linear-sum
assignment pick the same pairs.

> All 133 compared values -- per-class and overall AP / APH / mAP / mAPH, MOTA / MOTP / ID switches,
> ADE / FDE / miss rate -- agree to within `1e-9` (`1e-8` for APH, which `perception_eval` rounds).

**Dense scene.** The two differ on 87 values, **every one of which is classified against a documented
divergence**: Hungarian versus confidence-ordered greedy matching, the previous-frame MOTP score,
heading sign in APH, and how ID switches are counted. See
[Metric divergences](metric-divergences.md).

`--check` exits non-zero on any difference that is not on that list. That is the point of the flag:
an unexplained divergence is a regression, an explained one is a design decision.

## How a run works

The coordinator generates every scene **once** and writes it to a temporary directory as `.npz`,
then launches one worker per library over the same files -- so "the same input" is a file identity
rather than trust that two NumPy majors draw identical random streams.

```text
              ┌──────────────┐
              │ coordinator  │  generates scenes → .npz
              └──────┬───────┘
           ┌─────────┴─────────┐
           ▼                   ▼
   t4perceval worker     perception_eval worker
   (project env)         (uv run --no-project --with ...)
           │                   │
           └────────┬──────────┘
                    ▼
            merged JSON + Markdown report
```

The two run in **separate processes** because `perception_eval` pins NumPy 1. Its environment is
built into `.cache/uv-benchmark` on the first run and reused afterwards. Each worker pins itself to
one CPU.

## Quick iteration

```console
uv run python benchmarks/compare.py --skip-perf --check                # agreement only, seconds
uv run python benchmarks/compare.py --sizes 10,50 --repeats 2 --check  # fast timing
uv run python benchmarks/compare.py --skip-agreement                   # timing only
```

## When your change moves a number

Either fix it, or add the divergence to [Metric divergences](metric-divergences.md) with the
reasoning -- and regenerate `benchmarks/results/latest.md` in the same commit. An unclassified
difference fails `--check`, which is what stops a silent metric regression reaching main.

## Where to go next

- [Metric divergences](metric-divergences.md) -- the classified differences.
- [Contributing](contributing.md) -- the rest of the checks.
