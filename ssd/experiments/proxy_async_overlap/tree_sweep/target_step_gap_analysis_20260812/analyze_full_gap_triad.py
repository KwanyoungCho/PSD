#!/usr/bin/env python3
"""Decompose chain/P2-tree/full-tree latency on the repeated tiny7 runs."""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "full_gap_triad"
ARMS = ("chain", "p2_tree", "full_tree")
STATUSES = ("all", "hit_k1", "hit_k2", "miss")


def avg(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.fmean(vals) if vals else math.nan


def sd(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def load_one(pattern):
    paths = glob.glob(str(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"expected one file for {pattern}: {paths}")
    return json.loads(Path(paths[0]).read_text())


def index_events(events):
    out = defaultdict(list)
    for event in events:
        if event.get("label") and "start_ms" in event:
            out[event["label"]].append(event)
    for spans in out.values():
        spans.sort(key=lambda e: (e["start_ms"], e["idx"]))
    return out


def measured_tail(index, label, n_steps):
    spans = index[label]
    if len(spans) < n_steps:
        raise RuntimeError(f"{label}: {len(spans)} < measured {n_steps}")
    return spans[-n_steps:]


def in_window(index, labels, start, end):
    if isinstance(labels, str):
        labels = (labels,)
    choices = []
    for label in labels:
        choices.extend(
            e for e in index.get(label, ())
            if e["start_ms"] + 1e-6 >= start and e["start_ms"] < end)
    return min(choices, key=lambda e: (e["start_ms"], e["idx"])) \
        if choices else None


def event_ms(event):
    return event["ms"] if event is not None else math.nan


TARGET_EVENTS = {
    "target_send_request_ms": "target_send_request",
    "target_recv_wait_ms": "target_recv_response_wait",
    "target_recv_fused_ms": "target_recv_fused",
    "target_meta_read_ms": "target_response_meta_read",
    "target_recv_q_ms": (
        "target_recv_chain_q", "target_recv_parent_q_p1",
        "target_recv_parent_q_p2"),
    "target_wire_validate_ms": "tree_wire_parse_validate",
    "target_topology_prepare_ms": "tree_proxy_topology_prepare",
    "target_topology_cpu_pack_ms": "tree_topology_cpu_pack",
    "target_topology_h2d_ms": "tree_topology_h2d_copies",
    "target_parent_q_select_ms": "tree_parent_q_select",
    "verify_setup_ms": "verify_setup",
    "verify_tree_meta_ms": "tree_verify_meta_cpu",
    "verify_tree_mask_ms": "tree_verify_mask_prepare",
    "verify_input_copy_ms": "tree_verify_input_copy",
    "verify_attn_buffers_ms": "tree_verify_attention_buffers",
    "verify_graph_pre_ms": "graph_pre",
    "verify_exit_proxy_ms": "exit_proxy_side",
    "verify_graph_post_ms": "graph_post",
    "verify_final_logits_ms": "final_logits",
    "verify_accept_prep_ms": "verify_accept_prep",
    "verify_sample_accept_ms": "verify_sample_accept",
    "verify_accept_core_ms": ("chain_accept", "tree_accept_walk"),
    "verify_kv_commit_ms": "tree_kv_commit",
}


DRAFT_EVENTS = {
    "draft_kv_restore_ms": "tree_kv_restore",
    "draft_cache_response_ms": (
        "hit_cache_respond_hit_k1", "hit_cache_respond_hit_k2",
        "hit_cache_respond_miss"),
    "draft_rerank_ms": ("tree_hit_rerank_p1", "tree_hit_rerank_p2"),
    "draft_pack_generated_ms": (
        "tree_hit_pack_generated_p1", "tree_hit_pack_generated_p2"),
    "draft_select_subtree_ms": (
        "tree_hit_select_subtree_p1", "tree_hit_select_subtree_p2"),
    "draft_compact_gpu_ms": (
        "tree_hit_compact_gpu_p1", "tree_hit_compact_gpu_p2"),
    "draft_pack_served_ms": (
        "tree_hit_pack_served_p1", "tree_hit_pack_served_p2"),
    "draft_parent_q_gather_ms": (
        "tree_hit_parent_q_gather_p1", "tree_hit_parent_q_gather_p2"),
    "draft_response_pack_ms": "draft_response_pack",
    "draft_send_response_ms": "draft_send_response",
    "draft_send_fused_ms": "draft_send_fused",
    "draft_send_q_ms": (
        "draft_send_chain_q", "draft_send_parent_q_p1",
        "draft_send_parent_q_p2"),
}


def target_steps(events, n_steps):
    index = index_events(events)
    waits = measured_tail(index, "target_spec_wait", n_steps)
    rows = []
    for pos, wait in enumerate(waits):
        stop = waits[pos + 1]["start_ms"] if pos + 1 < len(waits) else math.inf
        setup = in_window(index, "verify_setup", wait["end_ms"], stop)
        accept = in_window(
            index, "verify_sample_accept",
            setup["start_ms"] if setup else wait["end_ms"], stop)
        post = in_window(
            index, "target_postprocess",
            accept["end_ms"] if accept else wait["end_ms"], stop)
        if not (setup and accept and post):
            raise RuntimeError(f"cannot segment target step {pos}")
        row = {
            "status": wait.get("status") or "unknown",
            "target_full_profile_ms": post["end_ms"] - wait["start_ms"],
            "target_spec_wait_ms": wait["ms"],
            "target_preverify_gap_ms": setup["start_ms"] - wait["end_ms"],
            "target_verify_profile_ms": accept["end_ms"] - setup["start_ms"],
            "target_postverify_ms": post["end_ms"] - accept["end_ms"],
        }
        for metric, labels in TARGET_EVENTS.items():
            row[metric] = event_ms(in_window(
                index, labels, wait["start_ms"], stop))
        rows.append(row)
    return rows


def draft_steps(events, n_steps):
    index = index_events(events)
    recvs = measured_tail(index, "draft_recv_request", n_steps)
    response_labels = (
        "hit_cache_respond_hit_k1", "hit_cache_respond_hit_k2",
        "hit_cache_respond_miss")
    rows = []
    for pos, recv in enumerate(recvs):
        stop = recvs[pos + 1]["start_ms"] if pos + 1 < len(recvs) else math.inf
        response = in_window(index, response_labels, recv["end_ms"], stop)
        send = in_window(index, "draft_send_response", recv["end_ms"], stop)
        if not (response and send):
            raise RuntimeError(f"cannot segment draft step {pos}")
        row = {
            "status": response.get("status") or "unknown",
            "draft_recv_request_ms": recv["ms"],
            "draft_service_to_send_end_ms": send["end_ms"] - recv["start_ms"],
            "draft_cache_to_send_gap_ms": send["start_ms"] - response["end_ms"],
        }
        for metric, labels in DRAFT_EVENTS.items():
            row[metric] = event_ms(in_window(
                index, labels, recv["start_ms"], stop))
        rows.append(row)
    return rows


def weighted_raw(raw, key):
    den = sum(row["n_verify_steps"] for row in raw)
    return 1000 * sum(row["n_verify_steps"] * row[key] for row in raw) / den


def summarize_run(path):
    raw = [json.loads(line) for line in path.read_text().splitlines() if line]
    n_steps = sum(row["n_verify_steps"] for row in raw)
    profile = path.with_name(path.stem + "_profile")
    target = target_steps(load_one(profile / "*target*.json"), n_steps)
    draft = draft_steps(load_one(profile / "*draft*.json"), n_steps)
    if len(target) != len(draft):
        raise RuntimeError(f"target/draft length mismatch: {path}")
    for i, (trow, drow) in enumerate(zip(target, draft)):
        if trow["status"] != drow["status"]:
            raise RuntimeError(
                f"status mismatch {path.name} step {i}: "
                f"{trow['status']} != {drow['status']}")
        trow.update({k: v for k, v in drow.items() if k != "status"})

    arm = path.stem.split("_r", 1)[0]
    repeat = int(path.stem.split("_r", 1)[1].split("_", 1)[0])
    overall = {
        "arm": arm,
        "repeat": repeat,
        "turns": len(raw),
        "verify_steps": n_steps,
        "raw_target_step_ms": weighted_raw(raw, "mean_target_step_s"),
        "raw_target_verify_ms": weighted_raw(raw, "mean_target_verify_s"),
    }
    overall["raw_outside_verify_ms"] = (
        overall["raw_target_step_ms"] - overall["raw_target_verify_ms"])
    summaries = []
    metrics = sorted({key for row in target for key in row if key != "status"})
    for status in STATUSES:
        chosen = target if status == "all" else [
            row for row in target if row["status"] == status]
        summary = {
            "arm": arm, "repeat": repeat, "status": status,
            "steps": len(chosen),
        }
        for metric in metrics:
            summary[metric] = avg([row[metric] for row in chosen])
        summaries.append(summary)
    return overall, summaries


def agg(rows, arm, status, metric):
    vals = [row[metric] for row in rows
            if row["arm"] == arm and row["status"] == status]
    return avg(vals), sd(vals)


def agg_overall(rows, arm, metric):
    vals = [row[metric] for row in rows if row["arm"] == arm]
    return avg(vals), sd(vals)


def fmt(pair):
    value, spread = pair
    return "—" if math.isnan(value) else f"{value:.3f} ± {spread:.3f}"


def delta(rows, left, right, status, metric):
    vals = []
    repeats = sorted({row["repeat"] for row in rows})
    for rep in repeats:
        match = {(row["arm"], row["repeat"]): row for row in rows
                 if row["status"] == status}
        if (left, rep) in match and (right, rep) in match:
            vals.append(match[(right, rep)][metric] - match[(left, rep)][metric])
    return avg(vals), sd(vals)


def table(lines, title, status, metrics, summaries):
    lines += [f"### {title}", "",
              "| Segment | Chain | P2 tree only | P1+P2 tree |",
              "|---|---:|---:|---:|"]
    for label, metric in metrics:
        values = [fmt(agg(summaries, arm, status, metric)) for arm in ARMS]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.append("")


def main():
    paths = [path for path in sorted(RUN_DIR.glob("*_r*_s42_o256.jsonl"))
             if sum(1 for line in path.open() if line.strip()) == 7
             and list(path.with_name(path.stem + "_profile").glob("*target*.json"))]
    overalls, summaries = [], []
    for path in paths:
        overall, per_status = summarize_run(path)
        overalls.append(overall)
        summaries.extend(per_status)
    if not overalls:
        raise SystemExit("no complete triad runs")

    with (RUN_DIR / "overall_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(overalls[0]))
        writer.writeheader(); writer.writerows(overalls)
    fields = sorted({key for row in summaries for key in row})
    with (RUN_DIR / "conditional_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summaries)

    lines = [
        "# Chain vs tree full target-step decomposition",
        "",
        "Latency-only diagnostic on the same tiny7 prompts, seed42, output 256. "
        "Warm-up spans are excluded using each raw JSONL's measured step count. "
        "AL and hit rate are not used for the causal conclusions.",
        "The target posts its fused receive before the draft has finished the "
        "response. Therefore `target_recv_fused` is a blocking readiness wait, "
        "not pure metadata transport. The following Q receive is the closest "
        "separately measured payload-transfer span.",
        "",
        "## Raw overall latency",
        "",
        "| Metric | Chain | P2 tree only | P1+P2 tree |",
        "|---|---:|---:|---:|",
    ]
    for label, metric in (
        ("Full target step", "raw_target_step_ms"),
        ("Target verify", "raw_target_verify_ms"),
        ("Outside verify", "raw_outside_verify_ms"),
    ):
        values = [fmt(agg_overall(overalls, arm, metric)) for arm in ARMS]
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    critical = (
        ("Full step", "target_full_profile_ms"),
        ("Draft/spec wait", "target_spec_wait_ms"),
        ("Response→verify gap", "target_preverify_gap_ms"),
        ("Target verify", "target_verify_profile_ms"),
        ("Post-verify", "target_postverify_ms"),
    )
    lines += ["", "## Status-specific critical path", ""]
    table(lines, "P1 hit", "hit_k1", critical, summaries)
    table(lines, "P2 hit", "hit_k2", critical, summaries)
    table(lines, "Miss", "miss", critical, summaries)

    draft_metrics = (
        ("Draft request receive/KV restore envelope", "draft_recv_request_ms"),
        ("Tree KV restore", "draft_kv_restore_ms"),
        ("Cache response compute", "draft_cache_response_ms"),
        ("Tree rerank total", "draft_rerank_ms"),
        ("Generated-tree pack/validate", "draft_pack_generated_ms"),
        ("Subtree selection", "draft_select_subtree_ms"),
        ("GPU compaction", "draft_compact_gpu_ms"),
        ("Served-tree pack/validate", "draft_pack_served_ms"),
        ("Parent-q gather", "draft_parent_q_gather_ms"),
        ("Response wire pack", "draft_response_pack_ms"),
        ("Fused metadata send", "draft_send_fused_ms"),
        ("Q/parent-q send", "draft_send_q_ms"),
        ("Full send envelope", "draft_send_response_ms"),
        ("Draft recv→send end", "draft_service_to_send_end_ms"),
    )
    lines += ["## Draft response decomposition", ""]
    table(lines, "P1 hit", "hit_k1", draft_metrics, summaries)
    table(lines, "P2 hit", "hit_k2", draft_metrics, summaries)

    target_prep = (
        ("Target response receive wait", "target_recv_wait_ms"),
        ("Blocking fused receive (includes draft readiness)",
         "target_recv_fused_ms"),
        ("Tree valid/phase scalar read", "target_meta_read_ms"),
        ("Q/parent-q receive", "target_recv_q_ms"),
        ("Response→verify gap", "target_preverify_gap_ms"),
        ("Wire list/parse/validate", "target_wire_validate_ms"),
        ("Topology total prepare", "target_topology_prepare_ms"),
        ("Topology CPU pack", "target_topology_cpu_pack_ms"),
        ("Topology H2D copies", "target_topology_h2d_ms"),
        ("Parent-q select", "target_parent_q_select_ms"),
    )
    lines += ["## Target receive and pre-verify decomposition", ""]
    table(lines, "P1 hit", "hit_k1", target_prep, summaries)
    table(lines, "P2 hit", "hit_k2", target_prep, summaries)

    verify_metrics = (
        ("Verify total", "target_verify_profile_ms"),
        ("Verify setup", "verify_setup_ms"),
        ("Tree metadata/depth", "verify_tree_meta_ms"),
        ("Tree mask prepare", "verify_tree_mask_ms"),
        ("Input copy", "verify_input_copy_ms"),
        ("Attention buffers", "verify_attn_buffers_ms"),
        ("Target graph pre", "verify_graph_pre_ms"),
        ("Exit/proxy side", "verify_exit_proxy_ms"),
        ("Target graph post", "verify_graph_post_ms"),
        ("Final logits", "verify_final_logits_ms"),
        ("Acceptance prep", "verify_accept_prep_ms"),
        ("Sample/accept envelope", "verify_sample_accept_ms"),
        ("Chain/tree accept core", "verify_accept_core_ms"),
        ("Tree KV commit", "verify_kv_commit_ms"),
    )
    lines += ["## Target verify decomposition", ""]
    table(lines, "P1 hit", "hit_k1", verify_metrics, summaries)
    table(lines, "P2 hit", "hit_k2", verify_metrics, summaries)

    lines += [
        "## Direct deltas against chain",
        "",
        "| Status/comparison | Full step | Draft/spec wait | Pre-verify | Verify |",
        "|---|---:|---:|---:|---:|",
    ]
    for status, label, right in (
        ("hit_k1", "P1: full tree − chain", "full_tree"),
        ("hit_k2", "P2: P2 tree − chain", "p2_tree"),
        ("hit_k2", "P2: full tree − chain", "full_tree"),
        ("miss", "Miss: full tree − chain", "full_tree"),
    ):
        vals = [fmt(delta(summaries, "chain", right, status, metric))
                for metric in (
                    "target_full_profile_ms", "target_spec_wait_ms",
                    "target_preverify_gap_ms", "target_verify_profile_ms")]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", "Machine-readable: `overall_runs.csv`, `conditional_runs.csv`."]
    (RUN_DIR / "FULL_GAP_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    print(RUN_DIR / "FULL_GAP_ANALYSIS.md")


if __name__ == "__main__":
    main()
