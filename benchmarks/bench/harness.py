"""Timing, memory and process plumbing shared by both sides of the benchmark."""

from __future__ import annotations

import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable


def pin_to_one_cpu() -> int | None:
    """Pin the process to its lowest-numbered available CPU; return it, or ``None``."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    available = os.sched_getaffinity(0)
    if not available:
        return None
    cpu = min(available)
    os.sched_setaffinity(0, {cpu})
    return cpu


def rss_bytes() -> int:
    """Return current RSS on Linux, falling back to peak RSS elsewhere."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024

    import resource

    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def environment(distribution: str) -> dict[str, str]:
    """Describe the interpreter one side runs under."""
    return {
        "version": package_version(distribution),
        "python": platform.python_version(),
        "numpy": package_version("numpy"),
    }


def measure(call: Callable[[], Any], *, warmups: int, repeats: int) -> dict[str, float]:
    """Time ``call`` with the collector disabled; report median, min and max in ms."""
    for _ in range(warmups):
        call()

    samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            call()
            samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    finally:
        gc.enable()

    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def retained_memory(warm: Callable[[], Any], build: Callable[[], Any], expected: int) -> int:
    """Return the RSS a representation retains, excluding imports and inputs.

    ``warm`` builds a one-row representation first so module imports and class
    initialization land in the baseline; the inputs ``build`` reads must already exist.
    """
    warm()
    gc.collect()
    before = rss_bytes()
    retained = build()
    gc.collect()
    after = rss_bytes()
    if sum(len(batch) for batch in retained) != expected:
        raise RuntimeError("builder produced an unexpected object count")
    del retained
    gc.collect()
    return max(0, after - before)


def run_worker(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    """Run one worker process and parse the JSON it prints."""
    completed = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"benchmark worker failed ({' '.join(command)}):\n{completed.stderr or completed.stdout}",
        )
    # Libraries may print warnings on stdout; the report is the last line.
    return json.loads(completed.stdout.strip().splitlines()[-1])


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"
