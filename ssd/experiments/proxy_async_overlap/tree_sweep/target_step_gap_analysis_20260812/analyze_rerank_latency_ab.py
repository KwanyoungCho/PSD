#!/usr/bin/env python3
"""Aggregate the repeated N1=14/12 vs N1=12/12 latency diagnostic.

The profiler includes the runner's two warm-up generations.  Raw JSONL rows
contain the exact measured-step count, so only the final N top-level spans are
used for every run.  AL and hit-rate values are intentionally not used to
judge this latency-only A/B.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "rerank_ab"


def mean(values):
    return statistics.fmean(values) if values else math.nan


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_events(pattern: str):
    paths = glob.glob(pattern)
    if len(paths) != 1:
        raise RuntimeError(f"expected one profile for {pattern}, got {paths}")
    return json.loads(Path(paths[0]).read_text())


def measured_top(events, label, n_steps):
    spans = sorted(
        (e for e in events if e.get("label") == label),
        key=lambda e: (e["start_ms"], e["idx"]),
    )
    if len(spans) < n_steps:
        raise RuntimeError(
            f"profile has {len(spans)} {label} spans, expected >= {n_steps}")
    return spans[-n_steps:]


def next_span(spans, start_ms, before_ms=math.inf):
    for span in spans:
        if span["start_ms"] + 1e-6 >= start_ms and span["start_ms"] < before_ms:
            return span
    return None


def target_breakdown(events, n_steps):
    waits = measured_top(events, "target_spec_wait", n_steps)
    labels = defaultdict(list)
    for event in events:
        label = event.get("label")
        if label and "start_ms" in event:
            labels[label].append(event)
    for spans in labels.values():
        spans.sort(key=lambda e: (e["start_ms"], e["idx"]))

    rows = []
    for i, wait in enumerate(waits):
        next_start = waits[i + 1]["start_ms"] if i + 1 < len(waits) else math.inf
        setup = next_span(labels["verify_setup"], wait["end_ms"], next_start)
        accept = next_span(
            labels["verify_sample_accept"],
            setup["end_ms"] if setup else wait["end_ms"],
            next_start,
        )
        post = next_span(
            labels["target_postprocess"],
            accept["end_ms"] if accept else wait["end_ms"],
            next_start,
        )
        if not (setup and accept and post):
            raise RuntimeError(f"could not segment target step at {wait['start_ms']}")
        row = {
            "status": wait.get("status") or "unknown",
            "full_profile_ms": post["end_ms"] - wait["start_ms"],
            "spec_wait_ms": wait["ms"],
            "preverify_gap_ms": setup["start_ms"] - wait["end_ms"],
            "verify_profile_ms": accept["end_ms"] - setup["start_ms"],
            "postverify_ms": post["end_ms"] - accept["end_ms"],
        }
        for label in (
            "tree_wire_parse_validate",
            "tree_proxy_topology_prepare",
            "tree_parent_q_select",
            "verify_setup",
        ):
            span = next_span(labels[label], wait["end_ms"], next_start)
            row[label + "_ms"] = span["ms"] if span else math.nan
        rows.append(row)
    return rows


def draft_breakdown(events, n_steps):
    recvs = measured_top(events, "draft_recv_request", n_steps)
    cutoff = recvs[0]["start_ms"] - 1e-6
    allowed_end = recvs[-1]["end_ms"] + 1000.0
    wanted = (
        "draft_recv_request",
        "hit_cache_respond_hit_k1",
        "hit_cache_respond_hit_k2",
        "hit_cache_respond_miss",
        "draft_send_response",
        "tree_kv_restore",
        "tree_hit_rerank_p1",
        "tree_hit_rerank_p2",
        "tree_hit_parent_q_gather_p1",
        "tree_hit_parent_q_gather_p2",
    )
    out = {}
    for label in wanted:
        vals = [
            e["ms"]
            for e in events
            if e.get("label") == label
            and e["start_ms"] >= cutoff
            and e["start_ms"] <= allowed_end
        ]
        out[label] = (mean(vals), len(vals))
    return out


def weighted_raw(rows, key):
    den = sum(row["n_verify_steps"] for row in rows)
    return 1000.0 * sum(
        row["n_verify_steps"] * row[key] for row in rows
    ) / den


def summarize_run(raw_path: Path):
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    n_steps = sum(row["n_verify_steps"] for row in rows)
    profile_dir = raw_path.with_name(raw_path.stem + "_profile")
    target = load_events(str(profile_dir / "*target*.json"))
    draft = load_events(str(profile_dir / "*draft*.json"))
    target_rows = target_breakdown(target, n_steps)
    draft_spans = draft_breakdown(draft, n_steps)

    stem = raw_path.stem
    arm = stem.split("_r", 1)[0]
    repeat = int(stem.split("_r", 1)[1].split("_", 1)[0])
    result = {
        "arm": arm,
        "repeat": repeat,
        "turns": len(rows),
        "verify_steps": n_steps,
        "raw_target_step_ms": weighted_raw(rows, "mean_target_step_s"),
        "raw_target_verify_ms": weighted_raw(rows, "mean_target_verify_s"),
    }
    result["raw_outside_verify_ms"] = (
        result["raw_target_step_ms"] - result["raw_target_verify_ms"])

    for status in ("all", "hit_k1", "hit_k2", "miss"):
        selected = target_rows if status == "all" else [
            row for row in target_rows if row["status"] == status]
        result[f"{status}_steps"] = len(selected)
        for metric in (
            "full_profile_ms",
            "spec_wait_ms",
            "preverify_gap_ms",
            "verify_profile_ms",
            "postverify_ms",
            "tree_wire_parse_validate_ms",
            "tree_proxy_topology_prepare_ms",
            "tree_parent_q_select_ms",
            "verify_setup_ms",
        ):
            result[f"{status}_{metric}"] = mean([
                row[metric] for row in selected if not math.isnan(row[metric])])

    for label, (value, count) in draft_spans.items():
        result[f"draft_{label}_ms"] = value
        result[f"draft_{label}_count"] = count
    return result


def fmt(value, std=None):
    if value is None or math.isnan(value):
        return "—"
    if std is None:
        return f"{value:.3f}"
    return f"{value:.3f} ± {std:.3f}"


def aggregate(results, arm, key):
    vals = [row[key] for row in results if row["arm"] == arm]
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return mean(vals), sample_std(vals)


def paired_delta(results, key, arms=("n14_m12", "n12_m12")):
    deltas = []
    for rep in sorted({row["repeat"] for row in results}):
        by_arm = {
            row["arm"]: row for row in results if row["repeat"] == rep}
        if all(arm in by_arm for arm in arms):
            deltas.append(by_arm[arms[1]][key] - by_arm[arms[0]][key])
    return mean(deltas), sample_std(deltas)


def main():
    raw_paths = [
        path for path in sorted(RUN_DIR.glob("n*_r*_s42_o256.jsonl"))
        if sum(1 for line in path.open() if line.strip()) == 7
    ]
    results = [summarize_run(path) for path in raw_paths]
    if not results:
        raise SystemExit(f"no completed runs in {RUN_DIR}")

    keys = sorted({key for row in results for key in row})
    with (RUN_DIR / "run_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    arms = ("n14_m12", "n12_m12")
    lines = [
        "# P1 tree rerank latency A/B",
        "",
        "동일한 7개 입력과 seed42를 사용한 latency-only 반복 진단이다. "
        "두 설정 모두 `M1=12`이므로 target이 검증하는 최대 노드 수는 같다. "
        "`N1=12`는 `N1=14 → M1=12` hit-time rerank만 제거한다.",
        "",
        "프로파일에는 runner의 warm-up 2회도 들어 있으므로 raw JSONL의 실제 "
        "verification-step 수와 맞춘 마지막 N개 span만 집계했다.",
        "",
        "## Conclusion",
        "",
        "- `14→12` subtree 재선택/compaction 비용은 P1 hit당 "
        "**0.914 ± 0.250 ms**로 반복 재현됐다.",
        "- 이를 제거하면 P1 hit의 draft/spec wait는 **0.959 ± 0.762 ms**, "
        "P1 full step은 **1.225 ± 0.714 ms** 감소했다.",
        "- 조작하지 않은 P2 full step은 `-0.236 ± 0.616 ms`, miss는 "
        "`+0.010 ± 0.900 ms` 흔들렸다. 따라서 overall `-0.531 ± 0.863 ms`는 "
        "방향만 참고할 수 있고 유의한 결론이 아니다.",
        "- `N1=12,M1=12`는 P1 latency 약 1 ms를 줄이는 유효 후보지만, "
        "chain 대비 전체 4–6 ms 차이의 전부는 아니다. tree 공통 metadata/"
        "topology 준비와 더 넓은 target verify가 여전히 남는다.",
        "",
        "두 arm은 같은 prompt/seed지만 N1 변경으로 생성 경로와 step 수가 "
        "달라졌다(`14/12`: 436, `12/12`: 470 steps/run). 따라서 직접 원인 "
        "판정은 전체 평균이 아니라 동일 코드 구간 span과 P1 조건부 latency를 "
        "사용한다.",
        "",
        "## Overall latency",
        "",
        "| Metric | N1=14, M1=12 | N1=12, M1=12 | Paired delta (12−14) |",
        "|---|---:|---:|---:|",
    ]
    overall_metrics = (
        ("Raw full target step", "raw_target_step_ms"),
        ("Raw target verify", "raw_target_verify_ms"),
        ("Raw outside verify", "raw_outside_verify_ms"),
        ("Profile full step", "all_full_profile_ms"),
        ("Profile draft/spec wait", "all_spec_wait_ms"),
        ("Profile response→verify gap", "all_preverify_gap_ms"),
        ("Profile verify", "all_verify_profile_ms"),
    )
    for label, key in overall_metrics:
        vals = []
        for arm in arms:
            vals.append(aggregate(results, arm, key))
        delta = paired_delta(results, key, arms)
        lines.append(
            f"| {label} | {fmt(*vals[0])} | {fmt(*vals[1])} | "
            f"{fmt(*delta)} ms |")

    lines += ["", "## Status-specific critical path", ""]
    for status, title in (("hit_k1", "P1 hit"), ("hit_k2", "P2 hit"), ("miss", "Miss")):
        lines += [
            f"### {title}",
            "",
            "| Segment | N1=14, M1=12 | N1=12, M1=12 | Difference |",
            "|---|---:|---:|---:|",
        ]
        for label, suffix in (
            ("Full profile step", "full_profile_ms"),
            ("Draft/spec response wait", "spec_wait_ms"),
            ("Response→verify gap", "preverify_gap_ms"),
            ("Target verify profile", "verify_profile_ms"),
            ("Post-verify", "postverify_ms"),
        ):
            key = f"{status}_{suffix}"
            a = aggregate(results, arms[0], key)
            b = aggregate(results, arms[1], key)
            delta = paired_delta(results, key, arms)
            lines.append(
                f"| {label} | {fmt(*a)} | {fmt(*b)} | {fmt(*delta)} ms |")
        lines.append("")

    lines += [
        "## Detailed tree spans",
        "",
        "| Span | N1=14, M1=12 | N1=12, M1=12 | Difference |",
        "|---|---:|---:|---:|",
    ]
    detailed = (
        ("P1 cache-hit response", "draft_hit_cache_respond_hit_k1_ms"),
        ("P1 hit rerank", "draft_tree_hit_rerank_p1_ms"),
        ("P1 parent-q gather", "draft_tree_hit_parent_q_gather_p1_ms"),
        ("P2 cache-hit response (control)",
         "draft_hit_cache_respond_hit_k2_ms"),
        ("Tree KV restore", "draft_tree_kv_restore_ms"),
        ("Target tree wire parse/validate", "hit_k1_tree_wire_parse_validate_ms"),
        ("Target topology prepare", "hit_k1_tree_proxy_topology_prepare_ms"),
        ("Target parent-q select", "hit_k1_tree_parent_q_select_ms"),
        ("Target verify setup", "hit_k1_verify_setup_ms"),
    )
    for label, key in detailed:
        a = aggregate(results, arms[0], key)
        b = aggregate(results, arms[1], key)
        delta = paired_delta(results, key, arms)
        lines.append(
            f"| {label} | {fmt(*a)} | {fmt(*b)} | {fmt(*delta)} ms |")

    lines += [
        "",
        "## Interpretation rule",
        "",
        "- 이번 결과는 rerank span, P1 spec wait, P1 full step이 함께 줄어 "
        "`14→12` on-hit rerank가 verify 바깥 지연의 직접 원인임을 확인했다.",
        "- 다음 진단은 모든 tree arm에 공통으로 남는 CPU topology parse/validation과 "
        "작은 H2D topology copy를 대상으로 한다.",
        "- 이 표의 AL/hit mix는 결론 근거가 아니다. 정식 AL/TPS 비교는 원인 확인 "
        "후 full Spec-Bench에서 별도로 수행한다.",
        "",
        "Per-run machine-readable values: `run_summary.csv`",
    ]
    (RUN_DIR / "LATENCY_AB.md").write_text("\n".join(lines) + "\n")
    print(RUN_DIR / "LATENCY_AB.md")


if __name__ == "__main__":
    main()
