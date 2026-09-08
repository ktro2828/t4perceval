"""Comparing the two sides' metric dictionaries and rendering the report."""

from __future__ import annotations

from typing import Any

METRICS_DOC = "docs/development/metric-divergences.md"

#: Absolute tolerance by key prefix; the first matching prefix wins.
TOLERANCES: tuple[tuple[str, float], ...] = (
    ("detection/aph", 1e-8),  # perception_eval rounds heading similarity to 10 decimals
    ("detection/maph", 1e-8),
    ("tracking/id_switch", 0.0),
    ("", 1e-9),
)

#: Divergences the two implementations are known to have, by key prefix. Only consulted in
#: the dense regime; in the unambiguous regime the scene is built so none of them applies.
DOCUMENTED: tuple[tuple[str, str, str], ...] = (
    ("detection/ap", "hungarian-vs-greedy", "#3"),
    ("detection/map", "hungarian-vs-greedy", "#3"),
    ("detection/aph", "hungarian-vs-greedy, aph-heading-sign", "#3, #7"),
    ("detection/maph", "hungarian-vs-greedy, aph-heading-sign", "#3, #7"),
    ("tracking/motp", "motp-previous-score", "#6"),
    ("tracking/id_switch", "idsw-across-missed-frame, idsw-estimation-side", "#4, #8"),
    ("tracking/mota", "idsw-across-missed-frame, idsw-estimation-side", "#4, #8"),
    ("prediction/", "hungarian-vs-greedy", "#3"),
)

#: Keys neither library reports directly; one side computes them with the other's formula.
DERIVED_PREFIXES = ("tracking/", "prediction/")
DERIVED_DETECTION = ("detection/ap/", "detection/aph/")

PHASES = ("construction", "matching", "detection", "tracking", "prediction")


def tolerance_of(key: str) -> float:
    for prefix, tolerance in TOLERANCES:
        if key.startswith(prefix):
            return tolerance
    raise AssertionError("unreachable")


def documented_reason(key: str) -> tuple[str, str] | None:
    for prefix, reason, anchor in DOCUMENTED:
        if key.startswith(prefix):
            return reason, anchor
    return None


def is_derived(key: str) -> bool:
    if not key.endswith("/ALL"):
        return False
    if key.startswith(DERIVED_PREFIXES):
        return True
    return key.startswith(DERIVED_DETECTION)


def compare(
    t4perceval: dict[str, float | None],
    perception_eval: dict[str, float | None],
    *,
    regime: str,
) -> list[dict[str, Any]]:
    """Line the two dictionaries up key by key and classify each difference."""
    rows = []
    for key in sorted(set(t4perceval) | set(perception_eval)):
        new = t4perceval.get(key)
        old = perception_eval.get(key)
        if new is None and old is None:
            diff, status = None, "match"
        elif new is None or old is None:
            diff, status = None, "mismatch"
        else:
            diff = abs(new - old)
            status = "match" if diff <= tolerance_of(key) else "mismatch"

        reason = ""
        if status == "mismatch" and regime == "dense":
            documented = documented_reason(key)
            if documented is not None:
                status = "documented"
                reason = f"{documented[0]} ({METRICS_DOC} {documented[1]})"

        parts = key.split("/")
        if len(parts) == 3:  # e.g. detection/map/ALL
            task, metric, class_name = parts
            group = "-"
        else:
            task, metric, group, class_name = parts
        rows.append(
            {
                "key": key,
                "task": task,
                "metric": metric,
                "group": group,
                "class": class_name,
                "t4perceval": new,
                "perception_eval": old,
                "abs_diff": diff,
                "status": status,
                "reason": reason,
                "derived": is_derived(key),
            },
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"match": 0, "documented": 0, "mismatch": 0}
    for row in rows:
        summary[row["status"]] += 1
    return summary


# -- markdown -----------------------------------------------------------------------------


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} ms"


def _num(value: float | None) -> str:
    return "nan" if value is None else f"{value:.6f}"


def _speedup(old: float | None, new: float | None) -> str:
    if old is None or new is None or new == 0:
        return "n/a"
    return f"{old / new:.1f}x"


def render_markdown(report: dict[str, Any]) -> str:
    impl = report["implementations"]
    params = report["parameters"]
    lines = [
        "# t4perceval vs perception_eval",
        "",
        f"Generated: `{report['generated_at']}` by `benchmarks/compare.py`.",
        "",
        "Both libraries are fed the same synthetic scenes (written once to `.npz`, read by both). "
        "Performance uses the `dense` regime; numerical agreement uses the `unambiguous` regime, "
        "in which every estimate has exactly one feasible ground truth so greedy and optimal "
        "assignment coincide, and the `dense` regime again to show where the two implementations "
        "are known to differ.",
        "",
        f"- CPU: `{report['system']['cpu_model']}` (one pinned logical CPU)",
        f"- Frames per scene: {params['frames']}; repetitions: {params['repeats']} "
        f"(matching: {params['matching_repeats']}) after {params['warmups']} warm-ups; median shown",
        f"- Seed: {params['seed']}",
    ]
    for name in ("perception_eval", "t4perceval"):
        info = impl[name]
        lines.append(
            f"- `{name}`: {info['version']}, Python {info['python']}, NumPy {info['numpy']}",
        )

    performance = report.get("performance")
    if performance:
        lines += ["", "## Performance", ""]
        descriptions = {
            "construction": "NumPy arrays -> one frame of each library's object representation "
            "(estimation and ground truth).",
            "matching": "Center-distance matching of one frame at 1.0 m.",
            "detection": f"mAP and mAPH over the scene at center-distance thresholds "
            f"{params['detection_thresholds']} (matching included).",
            "tracking": "CLEAR (MOTA / MOTP / ID switches) over the scene at 1.0 m (matching included).",
            "prediction": f"ADE / FDE / miss rate over the scene for top-k {params['top_ks']} "
            f"(matching included).",
        }
        by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
        for row in performance["timings"]:
            by_phase[row["phase"]].append(row)
        for phase in PHASES:
            rows = by_phase[phase]
            if not rows:
                continue
            lines += [
                f"### {phase.capitalize()}",
                "",
                descriptions[phase],
                "",
                "| Objects / frame | perception_eval | t4perceval | Speedup |",
                "| ---: | ---: | ---: | ---: |",
            ]
            for row in rows:
                old = row["perception_eval"]["median_ms"] if row["perception_eval"] else None
                new = row["t4perceval"]["median_ms"] if row["t4perceval"] else None
                lines.append(
                    f"| {row['objects']} | {_ms(old)} | {_ms(new)} | {_speedup(old, new)} |"
                )
            lines.append("")

        memory = performance["memory"]
        old_mem = memory["rss_delta_bytes"]["perception_eval"] / 2**20
        new_mem = memory["rss_delta_bytes"]["t4perceval"] / 2**20
        ratio = "inf" if new_mem == 0 else f"{old_mem / new_mem:.1f}x"
        lines += [
            "### Retained representation memory",
            "",
            f"RSS increase after constructing {memory['objects_per_set']:,} estimation and "
            f"{memory['objects_per_set']:,} ground-truth objects (detection representation):",
            "",
            "| Implementation | RSS increase | Relative usage |",
            "| :--- | ---: | ---: |",
            f"| perception_eval | {old_mem:.1f} MiB | {ratio} |",
            f"| t4perceval | {new_mem:.1f} MiB | 1.0x |",
        ]

    agreement = report.get("agreement")
    if agreement:
        for regime, title in (
            ("unambiguous", "Numerical agreement (unambiguous scene)"),
            ("dense", "Divergence report (dense scene)"),
        ):
            block = agreement.get(regime)
            if not block:
                continue
            summary = block["summary"]
            lines += [
                "",
                f"## {title}",
                "",
                f"{block['objects']} objects per frame, {block['frames']} frames. "
                f"{summary['match']} match, {summary['documented']} documented divergences, "
                f"{summary['mismatch']} unexplained mismatches.",
                "",
            ]
            if regime == "unambiguous":
                lines += _agreement_table(block["rows"], reason=False)
            else:
                lines += _agreement_table(
                    [row for row in block["rows"] if row["status"] != "match"] or block["rows"],
                    reason=True,
                )

    lines += [
        "",
        "## Notes",
        "",
        "- Timed t4perceval pipeline phases rebuild a `Store` from prebuilt input chunks on every "
        "call (a few microseconds), because a second run would append metric rows at the same "
        "reporting time.",
        "- Rows marked `*` are not reported directly by one library and were derived with the "
        "other's aggregation formula (perception_eval has no per-threshold all-class AP; "
        "t4perceval reports CLEAR and displacement per class only).",
        "- perception_eval's `inf` sentinels (CLEAR with no ground truth or no true positive) are "
        "read as `nan`.",
        f"- Documented divergences refer to the numbered items in `{METRICS_DOC}`. Only the dense "
        "scene may exercise them; a difference in the unambiguous scene is a mismatch.",
        "- perception_eval matches greedily in confidence order and pairs objects across labels "
        "in a second pass; t4perceval solves a linear-sum assignment per frame. The unambiguous "
        "scene is constructed so both pick the same pairs.",
    ]
    return "\n".join(lines) + "\n"


def _agreement_table(rows: list[dict[str, Any]], *, reason: bool) -> list[str]:
    header = "| Task | Metric | Threshold / k | Class | perception_eval | t4perceval | abs diff | Status |"
    rule = "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |"
    if reason:
        header += " Reason |"
        rule += " :--- |"
    lines = [header, rule]
    for row in rows:
        mark = "*" if row["derived"] else ""
        diff = "n/a" if row["abs_diff"] is None else f"{row['abs_diff']:.2e}"
        cells = [
            row["task"],
            row["metric"],
            row["group"],
            row["class"] + mark,
            _num(row["perception_eval"]),
            _num(row["t4perceval"]),
            diff,
            row["status"],
        ]
        if reason:
            cells.append(row["reason"])
        lines.append("| " + " | ".join(cells) + " |")
    return lines
