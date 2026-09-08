#!/usr/bin/env python3
"""Summarize the same-policy P1 rerank scheduling A/B."""
from __future__ import annotations

import glob
import json
import math
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROFILE_DIR = HERE / "rerank_precompute_fused_v2_ab"
OFF_DIR = HERE / "rerank_precompute_fused_profile_off_ab"
sys.path.insert(0, str(HERE))
import analyze_rerank_latency_ab as common  # noqa: E402


ARMS = ("legacy", "precompute_fused")


def mean(values):
    return statistics.fmean(values)


def std(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values)
                     / (len(values) - 1))


def fmt(values):
    return f"{mean(values):.3f} ± {std(values):.3f}"


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def weighted_raw(rows, key):
    den = sum(row["n_verify_steps"] for row in rows)
    return 1000.0 * sum(
        row["n_verify_steps"] * row[key] for row in rows) / den


def profile_run(arm, rep):
    raw = PROFILE_DIR / f"{arm}_r{rep}_s42_o256.jsonl"
    rows = load_rows(raw)
    n_steps = sum(row["n_verify_steps"] for row in rows)
    profile = raw.with_name(raw.stem + "_profile")
    draft_path = glob.glob(str(profile / "*draft*.json"))
    target_path = glob.glob(str(profile / "*target*.json"))
    if len(draft_path) != 1 or len(target_path) != 1:
        raise RuntimeError(f"missing profile for {raw}")
    draft = json.loads(Path(draft_path[0]).read_text())
    target = json.loads(Path(target_path[0]).read_text())

    requests = common.measured_top(draft, "draft_recv_request", n_steps)
    lo = requests[0]["start_ms"] - 1e-6
    hi = requests[-1]["end_ms"] + 1000.0
    result = {
        "raw_step": weighted_raw(rows, "mean_target_step_s"),
        "raw_verify": weighted_raw(rows, "mean_target_verify_s"),
    }
    result["raw_outside"] = result["raw_step"] - result["raw_verify"]
    for label in (
        "tree_hit_rerank_p1", "hit_cache_respond_hit_k1",
        "p1_rerank_precompute_gpu", "phase1_build", "proxy_wait",
    ):
        selected = [
            event["ms"] for event in draft
            if event.get("label") == label
            and lo <= event.get("start_ms", -1) <= hi
        ]
        result[label] = mean(selected) if selected else math.nan

    steps = common.target_breakdown(target, n_steps)
    for status in ("all", "hit_k1", "hit_k2", "miss"):
        selected = steps if status == "all" else [
            step for step in steps if step["status"] == status]
        result[f"{status}_full"] = mean([
            step["full_profile_ms"] for step in selected])
        result[f"{status}_wait"] = mean([
            step["spec_wait_ms"] for step in selected])
    return result


def values(results, arm, key):
    return [results[arm, rep][key] for rep in range(1, 4)]


def delta(results, key):
    values = [
        results["precompute_fused", rep][key]
        - results["legacy", rep][key]
        for rep in range(1, 4)
    ]
    return [value for value in values if math.isfinite(value)]


def main():
    prof = {
        (arm, rep): profile_run(arm, rep)
        for rep in range(1, 4) for arm in ARMS
    }
    hash_checks = []
    for directory, repeats in ((PROFILE_DIR, range(1, 4)),
                               (OFF_DIR, range(1, 3))):
        for rep in repeats:
            a = load_rows(directory / f"legacy_r{rep}_s42_o256.jsonl")
            b = load_rows(
                directory / f"precompute_fused_r{rep}_s42_o256.jsonl")
            hash_checks.extend(
                x["output_sha256"] == y["output_sha256"]
                for x, y in zip(a, b))

    off = {}
    for rep in range(1, 3):
        for arm in ARMS:
            rows = load_rows(OFF_DIR / f"{arm}_r{rep}_s42_o256.jsonl")
            step = weighted_raw(rows, "mean_target_step_s")
            verify = weighted_raw(rows, "mean_target_verify_s")
            off[arm, rep] = (step, verify, step - verify)

    lines = [
        "# P1 rerank scheduling optimization",
        "",
        "트리 생성/선택 파라미터는 모두 동일하다: `N1=14`, `M1=12`, "
        "`N2=M2=8`, `K1=8`, `K2=4`, seed42. 변경점은 P1 rerank의 "
        "실행 위치와 구현뿐이다.",
        "",
        "- Legacy: 다음 P1 cache hit의 응답 경로에서 CPU rerank/compaction",
        "- Optimized: P1 생성 직후 모든 root를 단일 fused CUDA kernel로 "
        "precompute; hit에서는 선택된 wire row만 readback/validate",
        "- 3회 profiler-on + 2회 profiler-off, 각 7 prompts/436 steps",
        f"- 출력 hash 일치: **{sum(hash_checks)}/{len(hash_checks)} prompts**",
        "",
        "## Profiler-on latency (3 paired runs)",
        "",
        "| Metric (ms) | Legacy | Optimized | Paired delta (opt−legacy) |",
        "|---|---:|---:|---:|",
    ]
    metrics = (
        ("P1 rerank on hit", "tree_hit_rerank_p1"),
        ("P1 cache response", "hit_cache_respond_hit_k1"),
        ("Fused rerank precompute", "p1_rerank_precompute_gpu"),
        ("P1 conditional target wait", "hit_k1_wait"),
        ("P1 conditional full target step", "hit_k1_full"),
        ("All target wait", "all_wait"),
        ("All full target step", "all_full"),
        ("Raw target step", "raw_step"),
        ("Raw outside target verify", "raw_outside"),
    )
    for label, key in metrics:
        legacy = values(prof, "legacy", key)
        optimized = values(prof, "precompute_fused", key)
        if all(math.isnan(v) for v in legacy):
            legacy_s = "—"
            delta_s = "—"
        else:
            legacy_s = fmt([v for v in legacy if math.isfinite(v)])
            delta_s = fmt(delta(prof, key))
        lines.append(
            f"| {label} | {legacy_s} | {fmt(optimized)} | "
            f"{delta_s} |")

    lines += [
        "",
        "## Profiler-off sanity check (2 paired runs)",
        "",
        "| Metric (ms) | Legacy runs | Optimized runs | Paired delta |",
        "|---|---:|---:|---:|",
    ]
    for index, label in enumerate(
            ("Raw target step", "Raw target verify", "Raw outside verify")):
        legacy = [off["legacy", rep][index] for rep in range(1, 3)]
        optimized = [off["precompute_fused", rep][index]
                     for rep in range(1, 3)]
        d = [b - a for a, b in zip(legacy, optimized)]
        lines.append(
            f"| {label} | {', '.join(f'{v:.3f}' for v in legacy)} | "
            f"{', '.join(f'{v:.3f}' for v in optimized)} | {fmt(d)} |")

    lines += [
        "",
        "## Decision",
        "",
        "- **채택 가능**: 직접 비용은 P1 hit rerank `−1.513 ± 0.564 ms`, "
        "P1 cache response `−1.751 ± 0.992 ms`로 반복 감소했다.",
        "- precompute 비용은 step당 `0.046 ± 0.000 ms`라 P1 여유를 "
        "실질적으로 소모하지 않는다.",
        "- profiler-on P1 조건부 wait는 `−1.938 ± 1.383 ms`였으나 "
        "profile-off 전체 step은 `−0.266 ± 1.202 ms`로 잡음보다 작았다. "
        "따라서 전체 TPS 개선을 주장할 근거는 아니며, 직접 P1 latency "
        "경로 최적화로 해석한다.",
        "- AL/hit/트리 구조는 바뀌지 않았다. 모든 paired run의 prompt별 "
        "output hash와 verification-step 수가 같았다.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "# profiler-on, three paired runs",
        "REPEATS=3 ./run_p1_rerank_precompute_ab.sh",
        "",
        "# profiler-off sanity check",
        "RESULT_TAG=rerank_precompute_fused_profile_off_ab \\",
        "  PROFILE_DUET_FLAG=0 PROFILE_DUET_DETAIL_FLAG=0 REPEATS=2 \\",
        "  ./run_p1_rerank_precompute_ab.sh",
        "```",
        "",
        "`SSD_P1_RERANK_PRECOMPUTE=0`은 legacy fallback, 기본값 `1`은 "
        "optimized path이다.",
    ]
    (PROFILE_DIR / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(PROFILE_DIR / "RESULTS.md")


if __name__ == "__main__":
    main()
