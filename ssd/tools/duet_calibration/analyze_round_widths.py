#!/usr/bin/env python3
"""Recommend per-round forward widths and depth budgets from topo traces.

The dynamic P1/P2 selector fills a fixed per-round forward width from the
global score frontier.  This tool asks, per phase and per tree depth:

* how many nodes the unfiltered policy actually built at that depth
  (width utilization of the captured graph lanes);
* when a served tree's accepted path reached that depth, what the accepted
  node's global score rank was among every same-depth node built that step
  (the minimum round width that would still have kept it);
* how deep accepted paths actually go (rounds past the observed tail are
  pure draft latency).

Narrowing analysis only: the trace records nodes that existed under the
collection widths, so ranks are exact for any smaller width but say nothing
about wider rounds.

Inputs are ``SSD_TREE_TOPO_TRACE`` prefixes (``.draft/.serve/.walk``), the
same files consumed by analyze_thresholds.py and analyze_tree_outcomes.py.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_tree_outcomes import _jsonl, _match_served_draft


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def load_joined_steps(prefixes: list[Path]) -> list[dict]:
    """Join draft forests with their served tree and accepted walk.

    Serving happens one step after the build, so serve ``step`` joins draft
    ``trace_seq == step - 1`` (same join as summarize_rerank_caps).
    """
    steps = []
    for prefix in prefixes:
        drafts = list(_jsonl(Path(str(prefix) + ".draft.jsonl")))
        serves = list(_jsonl(Path(str(prefix) + ".serve.jsonl")))
        walks = list(_jsonl(Path(str(prefix) + ".walk.jsonl")))
        if len(serves) != len(walks):
            raise ValueError(f"{prefix}: incomplete serve/walk trace")
        last_trace_seq = -1
        for serve, walk in zip(serves, walks):
            phase = int(serve.get("phase") or walk.get("phase") or 2)
            step = int(serve["step"])
            root_rank = int(serve["root_rank"])
            draft, root, served_to_generated = _match_served_draft(
                drafts, serve, last_trace_seq)
            last_trace_seq = int(draft["trace_seq"])
            served_path = [int(x) for x in walk.get("path", [])]
            steps.append({
                "phase": phase,
                "draft": draft,
                "root_rank": root_rank,
                # The verifier records compact post-rerank ids.  Width ranks
                # are defined on the generated forest, so restore the ids.
                "path": [served_to_generated[x] for x in served_path],
            })
    if not steps:
        raise ValueError("no joined serve/draft steps found")
    return steps


def analyze_phase(steps: list[dict], widths: list[int] | None) -> dict:
    built_per_depth: dict[int, list[int]] = defaultdict(list)
    producing_per_depth: dict[int, list[int]] = defaultdict(list)
    accepted_rank_per_depth: dict[int, list[int]] = defaultdict(list)
    parent_rank_per_depth: dict[int, list[int]] = defaultdict(list)
    accepted_lens = []

    for one in steps:
        draft = one["draft"]
        # Global frontier per depth across every root in this step's forest.
        # ``score`` is exactly the live selector key (root prior * path_conf).
        depth_scores: dict[int, list[float]] = defaultdict(list)
        depth_parents: dict[int, set] = defaultdict(set)
        for r, root in enumerate(draft["roots"]):
            n = int(root["valid"])
            for j in range(n):
                d = int(root["depth"][j])
                depth_scores[d].append(float(root["score"][j]))
                p = int(root["par"][j])
                if p >= 0:
                    depth_parents[int(root["depth"][p])].add((r, p))
        for d, scores in depth_scores.items():
            built_per_depth[d].append(len(scores))
        for d, parents in depth_parents.items():
            producing_per_depth[d].append(len(parents))

        root = draft["roots"][one["root_rank"]]
        path = one["path"]
        accepted_lens.append(len(path))
        for i, j in enumerate(path):
            d = int(root["depth"][j])
            s = float(root["score"][j])
            # Competition rank among every same-depth node built this step.
            rank = 1 + sum(1 for x in depth_scores[d] if x > s)
            accepted_rank_per_depth[d].append(rank)
            if i + 1 < len(path):
                # A non-terminal accepted node had an accepted child, so it
                # HAD to win a round-d forward lane.  This is the exact
                # width requirement; terminal nodes stay servable as leaves.
                parent_rank_per_depth[d].append(rank)

    depths = sorted(built_per_depth)
    trees = len(steps)
    per_depth = {}
    for d in depths:
        built = sorted(built_per_depth[d])
        ranks = sorted(accepted_rank_per_depth.get(d, []))
        pranks = sorted(parent_rank_per_depth.get(d, []))
        producing = sorted(producing_per_depth.get(d, []))
        row = {
            "steps_with_depth": len(built),
            "built_nodes_p50": _quantile(built, 0.5),
            "built_nodes_p95": _quantile(built, 0.95),
            "built_nodes_max": built[-1],
            "producing_parents_p50": _quantile(producing, 0.5),
            "producing_parents_max": producing[-1] if producing else 0,
            "accepted_n": len(ranks),
            "accepted_rank_p50": _quantile(ranks, 0.5),
            "accepted_rank_p95": _quantile(ranks, 0.95),
            "accepted_rank_max": ranks[-1] if ranks else None,
            "expanding_n": len(pranks),
            "expanding_rank_p95": _quantile(pranks, 0.95),
            "expanding_rank_max": pranks[-1] if pranks else None,
        }
        if ranks:
            # Width w keeps a depth-d accepted node iff its rank <= w.  The
            # forward at round d evaluates depth-d nodes, so this is the
            # narrowing guide for round index d (round 0 evaluates roots).
            coverage = {}
            candidates = sorted({1, 2, 4, 6, 8, 10, 12, 14, 16, 20,
                                 row["built_nodes_max"]})
            for w in candidates:
                coverage[str(w)] = sum(1 for r in ranks if r <= w) / len(ranks)
            row["accepted_coverage_by_width"] = coverage
            if pranks:
                pcov = {}
                for w in candidates:
                    pcov[str(w)] = sum(
                        1 for r in pranks if r <= w) / len(pranks)
                row["expanding_coverage_by_width"] = pcov
        per_depth[str(d)] = row

    depth_cdf = {}
    for d in depths:
        depth_cdf[str(d)] = sum(1 for a in accepted_lens if a >= d) / trees
    return {
        "trees": trees,
        "accepted_len_mean": sum(accepted_lens) / trees,
        "accepted_len_max": max(accepted_lens),
        "accept_reaches_depth_rate": depth_cdf,
        "per_depth": per_depth,
        "configured_widths": widths,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-prefix", type=Path, nargs="+", required=True)
    ap.add_argument("--p1-widths",
                    help="comma-separated configured P1 round widths for "
                         "side-by-side reading, e.g. 20,16,16,...")
    ap.add_argument("--p2-widths",
                    help="comma-separated configured P2 round widths")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)

    steps = load_joined_steps(args.trace_prefix)
    result = {}
    for phase in (1, 2):
        one = [x for x in steps if x["phase"] == phase]
        if not one:
            continue
        widths = None
        arg = args.p1_widths if phase == 1 else args.p2_widths
        if arg:
            widths = [int(x) for x in arg.split(",") if x]
        result[f"p{phase}"] = analyze_phase(one, widths)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
