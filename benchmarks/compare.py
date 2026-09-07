#!/usr/bin/env python3
"""Compare t4perceval with perception_eval 1.3.6 on speed, memory and metric values.

Run from the repository root::

    uv run python benchmarks/compare.py                 # full run, writes benchmarks/results/latest.*
    uv run python benchmarks/compare.py --skip-perf --check   # numerical agreement only

Each implementation runs in its own process: ``perception_eval`` pins NumPy < 2 while
t4perceval runs on NumPy 2, so the perception_eval worker is launched through
``uv run --no-project --with perception_eval@v1.3.6``. The coordinator generates every scene
once, writes it to ``.npz``, and both workers read the same files.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import harness  # noqa: E402
from bench.report import PHASES, compare, render_markdown, summarize  # noqa: E402
from bench.scenes import (  # noqa: E402
    DETECTION_THRESHOLDS,
    MATCHING_THRESHOLD,
    MISS_TOLERANCE,
    TOP_KS,
    Scene,
    SceneParams,
    make_scene,
)

PERCEPTION_EVAL_SOURCE = (
    "perception_eval @ git+https://github.com/tier4/autoware_perception_evaluation.git@v1.3.6"
)
IMPLEMENTATIONS = ("t4perceval", "perception_eval")
REGIMES = ("unambiguous", "dense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sizes", default="10,50,100,200", help="objects per frame, comma-separated"
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument(
        "--repeats", type=int, default=5, help="timed repetitions of the scene phases"
    )
    parser.add_argument("--matching-repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--memory-objects", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agreement-objects", type=int, default=50)
    parser.add_argument("--check", action="store_true", help="exit 1 on an unexplained mismatch")
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-agreement", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument("--worker", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--scene-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.size_list = tuple(int(item) for item in args.sizes.split(",") if item)
    if not args.size_list or min(args.size_list) <= 0:
        parser.error("--sizes must contain positive integers")
    if args.repeats <= 0 or args.matching_repeats <= 0 or args.memory_objects <= 0:
        parser.error("repeats and memory-objects must be positive")
    if args.warmups < 0 or args.frames < 3 or args.agreement_objects < 6:
        parser.error("warmups must be >= 0, frames >= 3 and agreement-objects >= 6")
    return args


# -- worker -------------------------------------------------------------------------------


def worker(name: str, args: argparse.Namespace) -> dict[str, Any]:
    side = importlib.import_module(f"bench.side_{name}")
    cpu = harness.pin_to_one_cpu()
    result: dict[str, Any] = {"implementation": name, "cpu": cpu, **harness.environment(name)}
    scene_dir: Path = args.scene_dir

    if not args.skip_perf:
        warm = Scene.load(scene_dir / "warm.npz")
        memory_scene = Scene.load(scene_dir / "memory.npz")
        gc.collect()
        result["memory"] = {
            "objects_per_set": args.memory_objects,
            "rss_delta_bytes": harness.retained_memory(
                side.memory_builder(warm),
                side.memory_builder(memory_scene),
                2 * args.memory_objects,
            ),
        }
        del memory_scene

        timings = []
        for size in args.size_list:
            scene = Scene.load(scene_dir / f"perf_{size}.npz")
            calls = {
                "construction": lambda scene=scene: side.build_frame(scene, 0, "detection"),
                "matching": side.matching_call(scene),
                "detection": side.detection_call(scene),
                "tracking": side.tracking_call(scene),
                "prediction": side.prediction_call(scene),
            }
            for phase in PHASES:
                repeats = args.matching_repeats if phase == "matching" else args.repeats
                timing = harness.measure(calls[phase], warmups=args.warmups, repeats=repeats)
                timings.append({"objects": size, "phase": phase, **timing})
        result["timings"] = timings

    if not args.skip_agreement:
        metrics: dict[str, dict[str, float | None]] = {}
        for regime in REGIMES:
            scene = Scene.load(scene_dir / f"agree_{regime}.npz")
            names = scene.class_names
            values: dict[str, float | None] = {}
            values.update(side.detection_metrics(side.detection_call(scene)(), names))
            values.update(side.tracking_metrics(side.tracking_call(scene)(), names))
            values.update(side.prediction_metrics(side.prediction_call(scene)(), names))
            metrics[regime] = values
        result["metrics"] = metrics

    return result


# -- coordinator --------------------------------------------------------------------------


def write_scenes(args: argparse.Namespace, scene_dir: Path) -> None:
    seed = args.seed
    make_scene(SceneParams("unambiguous", 3, 1, recall=1.0, fp_rate=0.0, seed=seed)).save(
        scene_dir / "warm.npz"
    )
    if not args.skip_perf:
        make_scene(
            SceneParams(
                "unambiguous",
                1,
                args.memory_objects,
                recall=1.0,
                fp_rate=0.0,
                seed=seed,
                check=False,
            ),
        ).save(scene_dir / "memory.npz")
        for size in args.size_list:
            make_scene(SceneParams("dense", args.frames, size, confusion_rate=0.1, seed=seed)).save(
                scene_dir / f"perf_{size}.npz",
            )
    if not args.skip_agreement:
        for regime in REGIMES:
            confusion = 0.1 if regime == "dense" else 0.0
            make_scene(
                SceneParams(
                    regime, args.frames, args.agreement_objects, confusion_rate=confusion, seed=seed
                ),
            ).save(scene_dir / f"agree_{regime}.npz")


def coordinate(args: argparse.Namespace) -> dict[str, Any]:
    script = Path(__file__).resolve()
    root = script.parents[1]
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(root / ".cache" / "uv-benchmark"))
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    with tempfile.TemporaryDirectory(prefix="t4perceval-bench-") as tmp:
        scene_dir = Path(tmp)
        write_scenes(args, scene_dir)
        common = [
            f"--sizes={args.sizes}",
            f"--frames={args.frames}",
            f"--repeats={args.repeats}",
            f"--matching-repeats={args.matching_repeats}",
            f"--warmups={args.warmups}",
            f"--memory-objects={args.memory_objects}",
            f"--seed={args.seed}",
            f"--agreement-objects={args.agreement_objects}",
            f"--scene-dir={scene_dir}",
            *(["--skip-perf"] if args.skip_perf else []),
            *(["--skip-agreement"] if args.skip_agreement else []),
        ]
        results = {
            "t4perceval": harness.run_worker(
                [sys.executable, str(script), "--worker=t4perceval", *common],
                env,
            ),
            "perception_eval": harness.run_worker(
                [
                    "uv",
                    "run",
                    "--no-project",
                    "--with",
                    PERCEPTION_EVAL_SOURCE,
                    "python",
                    str(script),
                    "--worker=perception_eval",
                    *common,
                ],
                env,
            ),
        }

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": {"platform": platform.platform(), "cpu_model": harness.cpu_model()},
        "parameters": {
            "sizes": list(args.size_list),
            "frames": args.frames,
            "repeats": args.repeats,
            "matching_repeats": args.matching_repeats,
            "warmups": args.warmups,
            "memory_objects": args.memory_objects,
            "seed": args.seed,
            "agreement_objects": args.agreement_objects,
            "detection_thresholds": list(DETECTION_THRESHOLDS),
            "matching_threshold": MATCHING_THRESHOLD,
            "top_ks": list(TOP_KS),
            "miss_tolerance": MISS_TOLERANCE,
        },
        "implementations": {
            name: {key: results[name][key] for key in ("version", "python", "numpy", "cpu")}
            for name in IMPLEMENTATIONS
        },
    }

    if not args.skip_perf:
        timings = []
        for size in args.size_list:
            for phase in PHASES:
                per_side = {
                    name: next(
                        row
                        for row in results[name]["timings"]
                        if row["objects"] == size and row["phase"] == phase
                    )
                    for name in IMPLEMENTATIONS
                }
                old = per_side["perception_eval"]["median_ms"]
                new = per_side["t4perceval"]["median_ms"]
                timings.append(
                    {
                        "objects": size,
                        "phase": phase,
                        **{
                            name: {k: per_side[name][k] for k in ("median_ms", "min_ms", "max_ms")}
                            for name in IMPLEMENTATIONS
                        },
                        "speedup": old / new if new else None,
                    },
                )
        report["performance"] = {
            "memory": {
                "objects_per_set": args.memory_objects,
                "rss_delta_bytes": {
                    name: results[name]["memory"]["rss_delta_bytes"] for name in IMPLEMENTATIONS
                },
            },
            "timings": timings,
        }

    if not args.skip_agreement:
        agreement = {}
        for regime in REGIMES:
            rows = compare(
                results["t4perceval"]["metrics"][regime],
                results["perception_eval"]["metrics"][regime],
                regime=regime,
            )
            agreement[regime] = {
                "objects": args.agreement_objects,
                "frames": args.frames,
                "rows": rows,
                "summary": summarize(rows),
            }
        report["agreement"] = agreement

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latest.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report)
    (args.output_dir / "latest.md").write_text(markdown)
    return report


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.scene_dir is None:
            raise SystemExit("--worker requires --scene-dir")
        print(json.dumps(worker(args.worker, args)))
        return 0

    report = coordinate(args)
    print(render_markdown(report), end="")
    mismatches = sum(block["summary"]["mismatch"] for block in report.get("agreement", {}).values())
    if args.check and mismatches:
        print(f"\n{mismatches} unexplained mismatch(es)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
