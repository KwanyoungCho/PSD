#!/usr/bin/env python3
"""Explain how much a generated DUET tree helped after target verification.

The topology trace produces two files for every diagnostic run:

* ``PREFIX.serve.jsonl``: the exact root/tree sent by the draft;
* ``PREFIX.walk.jsonl``: the exact path accepted by the target.

This tool joins them in serving order, checks that phase and topology match,
and reports how many accepted paths actually used an alternative sibling.
That is the direct, post-hoc answer to whether branching helped beyond the
first-child chain.  Optional E0 input also joins generated P1 roots to the
next request's actual cache key, so context reach and local root probability
can be evaluated separately instead of trusting their product by assumption.

Trace runs are diagnostic: their TPS is not a performance measurement.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{lineno}: {exc}") \
                    from exc


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _one_tree(serve: dict, walk: dict) -> dict:
    phase_s = int(serve.get("phase", 0))
    phase_w = int(walk.get("phase", 0))
    if phase_s != phase_w:
        raise ValueError(
            f"serve/walk phase mismatch: {phase_s} != {phase_w}")
    valid = int(serve["valid"])
    par_s = [int(x) for x in serve["par"][:valid]]
    sib_s = [int(x) for x in serve["sib"][:valid]]
    par_w = [int(x) for x in walk["par"][:valid]]
    sib_w = [int(x) for x in walk["sib"][:valid]]
    if par_s != par_w or sib_s != sib_w:
        raise ValueError("serve/walk topology mismatch")

    path = [int(x) for x in walk.get("path", [])]
    if any(j < 0 or j >= valid for j in path):
        raise ValueError(f"accepted path outside valid tree: {path}/{valid}")
    ctx = -1
    for j in path:
        if par_s[j] != ctx:
            raise ValueError(
                f"non-contiguous accepted path: node={j} "
                f"parent={par_s[j]} expected={ctx}")
        ctx = j

    alternative_at = next(
        (i for i, node in enumerate(path) if sib_s[node] > 0), None)
    first_child_prefix = (len(path) if alternative_at is None
                          else alternative_at)
    # Once an alternative sibling is accepted, that node and every accepted
    # descendant would not exist in a first-child-only tree.  This is a
    # structural attribution, not a counterfactual full-generation TPS.
    branch_assisted = (0 if alternative_at is None
                       else len(path) - alternative_at)
    internal_nodes = set(x for x in par_s if x >= 0)
    accepted_parents = {-1}
    accepted_parents.update(path[:-1])
    useful_expanded = len(internal_nodes & accepted_parents)

    return {
        "phase": phase_s,
        "valid": valid,
        "accepted": len(path),
        "first_child_prefix": first_child_prefix,
        "used_alternative": alternative_at is not None,
        "branch_assisted": branch_assisted,
        "accepted_siblings": [sib_s[j] for j in path],
        "internal_nodes": len(internal_nodes),
        "useful_expanded_parents": useful_expanded,
        "root_start_score": serve.get("root_start_score"),
        "root_context_id": serve.get("root_context_id"),
        "root_context_reach": serve.get("root_context_reach"),
        "root_local_q": serve.get("root_local_q"),
    }


def summarize_tree_rows(rows: list[dict]) -> dict:
    sibling_counts = Counter(
        sibling for row in rows for sibling in row["accepted_siblings"])
    scores = [float(r["root_start_score"]) for r in rows
              if r["root_start_score"] is not None]
    return {
        "trees": len(rows),
        "generated_nodes": sum(r["valid"] for r in rows),
        "accepted_nodes": sum(r["accepted"] for r in rows),
        "accepted_nodes_per_tree": _mean(
            [float(r["accepted"]) for r in rows]),
        "accepted_node_fraction": (
            sum(r["accepted"] for r in rows) /
            sum(r["valid"] for r in rows)
            if sum(r["valid"] for r in rows) else 0.0),
        "trees_using_alternative_sibling": sum(
            bool(r["used_alternative"]) for r in rows),
        "alternative_tree_rate": (
            sum(bool(r["used_alternative"]) for r in rows) / len(rows)
            if rows else 0.0),
        "branch_assisted_accepted_nodes": sum(
            r["branch_assisted"] for r in rows),
        "branch_assisted_share_of_accepted": (
            sum(r["branch_assisted"] for r in rows) /
            sum(r["accepted"] for r in rows)
            if sum(r["accepted"] for r in rows) else 0.0),
        "accepted_sibling_order_counts": {
            str(k): sibling_counts[k] for k in sorted(sibling_counts)},
        "internal_nodes": sum(r["internal_nodes"] for r in rows),
        "useful_expanded_parents": sum(
            r["useful_expanded_parents"] for r in rows),
        "root_start_score_mean": _mean(scores),
        "root_start_score_median": _median(scores),
        "p1_hit_context_counts": dict(sorted(Counter(
            int(r["root_context_id"]) for r in rows
            if r["phase"] == 1 and r["root_context_id"] is not None
        ).items())),
    }


def load_tree_pairs(prefixes: list[Path]) -> list[dict]:
    rows = []
    for prefix in prefixes:
        serves = list(_jsonl(Path(str(prefix) + ".serve.jsonl")))
        walks = list(_jsonl(Path(str(prefix) + ".walk.jsonl")))
        if len(serves) != len(walks):
            raise ValueError(
                f"{prefix}: serve/walk counts differ "
                f"({len(serves)} != {len(walks)}); use a completed trace")
        rows.extend(_one_tree(s, w) for s, w in zip(serves, walks))
    return rows


def _safe_rerank_indices(par: list[int], sib: list[int],
                         raw_q: list[float], cap: int) -> list[int]:
    """Offline mirror of p2_tree.rerank_tree_indices.

    Keep this script standalone (it is often run on archived traces without a
    model environment), but preserve the production rule: every selected node
    brings its ancestors and all earlier ordered-WOR siblings.
    """
    n = len(par)
    if cap >= n:
        return list(range(n))
    groups = defaultdict(dict)
    conf = [0.0] * n
    for j, (p, s, q) in enumerate(zip(par, sib, raw_q)):
        groups[p][s] = j
        conf[j] = max(0.0, float(q)) * (
            1.0 if p < 0 else conf[p])

    def closure(node: int) -> set[int]:
        need = set()
        while node >= 0:
            parent = par[node]
            for order in range(sib[node] + 1):
                need.add(groups[parent][order])
            node = parent
        return need

    chosen = set()
    for node in sorted(range(n), key=lambda j: (-conf[j], j)):
        expanded = chosen | closure(node)
        if len(expanded) <= cap:
            chosen = expanded
    if len(chosen) < cap:
        for node in range(n):
            expanded = chosen | closure(node)
            if len(expanded) <= cap:
                chosen = expanded
            if len(chosen) == cap:
                break
    return sorted(chosen)


def summarize_rerank_caps(prefixes: list[Path], caps: list[int]) -> dict:
    """Estimate useful final-verify caps from already collected hit traces.

    This is an observed-path retention diagnostic, not a counterfactual AL:
    removing a rejected proposal changes the later residual RNG trajectory.
    It is nevertheless a cheap way to reject caps that discard many nodes
    which the target actually accepted before running live A/B tests.
    """
    rows = []
    for prefix in prefixes:
        drafts = {}
        for rec in _jsonl(Path(str(prefix) + ".draft.jsonl")):
            phase = int(rec.get("phase") or 2)
            drafts[(int(rec["trace_seq"]), phase)] = rec
        serves = list(_jsonl(Path(str(prefix) + ".serve.jsonl")))
        walks = list(_jsonl(Path(str(prefix) + ".walk.jsonl")))
        if len(serves) != len(walks):
            raise ValueError(f"{prefix}: incomplete serve/walk trace")
        for serve, walk in zip(serves, walks):
            phase = int(serve.get("phase") or walk.get("phase") or 2)
            step = int(serve["step"])
            root_rank = int(serve["root_rank"])
            draft = drafts.get((step - 1, phase))
            if draft is None or root_rank >= len(draft["roots"]):
                raise ValueError(
                    f"{prefix}: cannot join served step={step}, "
                    f"phase={phase}, root={root_rank} to draft trace")
            root = draft["roots"][root_rank]
            valid = int(serve["valid"])
            par = [int(x) for x in root["par"][:valid]]
            sib = [int(x) for x in root["sib"][:valid]]
            if par != [int(x) for x in serve["par"][:valid]] \
                    or sib != [int(x) for x in serve["sib"][:valid]]:
                raise ValueError(
                    f"{prefix}: joined draft/serve topology differs at "
                    f"step={step}, phase={phase}, root={root_rank}")
            rows.append({
                "phase": phase, "valid": valid, "par": par, "sib": sib,
                "raw_q": [float(x) for x in root["raw_q"][:valid]],
                "path": [int(x) for x in walk.get("path", [])],
            })

    result = {}
    for phase in sorted({r["phase"] for r in rows}):
        phase_rows = [r for r in rows if r["phase"] == phase]
        accepted_total = sum(len(r["path"]) for r in phase_rows)
        one_phase = {}
        for cap in caps:
            sent = retained = full = 0
            for row in phase_rows:
                keep = set(_safe_rerank_indices(
                    row["par"], row["sib"], row["raw_q"], cap))
                sent += len(keep)
                prefix_len = 0
                for node in row["path"]:
                    if node not in keep:
                        break
                    prefix_len += 1
                retained += prefix_len
                full += int(prefix_len == len(row["path"]))
            original_sent = sum(r["valid"] for r in phase_rows)
            one_phase[str(cap)] = {
                "trees": len(phase_rows),
                "mean_verify_nodes": sent / len(phase_rows),
                "verify_node_reduction": (
                    1.0 - sent / original_sent if original_sent else 0.0),
                "observed_accepted_nodes_retained": (
                    retained / accepted_total if accepted_total else 1.0),
                "observed_full_paths_retained": full / len(phase_rows),
            }
        result[f"p{phase}"] = one_phase
    return result


def _draft_e0_paths(inputs: list[Path]) -> list[Path]:
    out = []
    for value in inputs:
        if value.is_file():
            out.append(value)
            continue
        direct = sorted(value.glob("e0_draft_*.jsonl"))
        nested = sorted(value.glob("e0/e0_draft_*.jsonl"))
        out.extend(direct or nested)
    if not out:
        raise ValueError("no e0_draft_*.jsonl files found")
    return out


def load_p1_root_outcomes(inputs: list[Path]) -> list[dict]:
    rows = []
    for path in _draft_e0_paths(inputs):
        requests = {}
        roots = {}
        for rec in _jsonl(path):
            step = rec.get("step_id")
            if step is None:
                continue
            if rec.get("kind") == "request":
                requests[int(step)] = [tuple(int(x) for x in row)
                                       for row in rec["cache_keys"]]
            elif rec.get("kind") == "p1_roots":
                roots[int(step)] = rec
        for step, rec in roots.items():
            next_req = requests.get(step + 1)
            if not next_req:
                continue
            seq_id = int(rec["seq_id"])
            outcome = next((x for x in next_req if x[0] == seq_id), None)
            if outcome is None:
                continue
            outcome_key = (outcome[1], outcome[2])
            fields = zip(
                rec["context_ids"], rec["root_tokens"],
                rec["start_scores"], rec["context_reach"],
                rec["local_q"], rec["valid"])
            for context_id, token, score, reach, local_q, valid in fields:
                if int(valid) <= 0:
                    continue
                rows.append({
                    "context_id": int(context_id),
                    "token": int(token),
                    "start_score": float(score),
                    "context_reach": float(reach),
                    "local_q": float(local_q),
                    "hit": int((int(context_id), int(token)) == outcome_key),
                })
    return rows


def _rank_auc(rows: list[dict], field: str) -> float | None:
    positives = [float(x[field]) for x in rows if x["hit"]]
    negatives = [float(x[field]) for x in rows if not x["hit"]]
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def summarize_p1_roots(rows: list[dict]) -> dict:
    hits = sum(x["hit"] for x in rows)
    by_context = {}
    for context in sorted({x["context_id"] for x in rows}):
        one = [x for x in rows if x["context_id"] == context]
        n_hit = sum(x["hit"] for x in one)
        by_context[str(context)] = {
            "roots": len(one), "hits": n_hit,
            "hit_rate": n_hit / len(one) if one else 0.0,
        }
    return {
        "servable_roots": len(rows),
        "actual_hits": hits,
        "hit_rate_per_root": hits / len(rows) if rows else 0.0,
        # AUC 0.5 means no ranking signal; >0.5 is useful.  Comparing these
        # three values directly tests whether multiplying by context reach
        # improves the local root probability or makes it worse.
        "ranking_auc": {
            field: _rank_auc(rows, field) for field in
            ("start_score", "context_reach", "local_q")
        },
        "hit_score_means": {
            field: _mean([float(x[field]) for x in rows if x["hit"]])
            for field in ("start_score", "context_reach", "local_q")
        },
        "miss_score_means": {
            field: _mean([float(x[field]) for x in rows if not x["hit"]])
            for field in ("start_score", "context_reach", "local_q")
        },
        "by_context": by_context,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-prefix", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--e0-dir", type=Path, nargs="+",
        help="optional E0 directories/files containing P1 root records")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument(
        "--rerank-caps",
        help="comma-separated final verification caps; also joins the draft "
             "trace and reports observed accepted-path retention")
    args = ap.parse_args(argv)

    rows = load_tree_pairs(args.trace_prefix)
    by_phase = {}
    for phase in (1, 2):
        phase_rows = [x for x in rows if x["phase"] == phase]
        if phase_rows:
            by_phase[f"p{phase}"] = summarize_tree_rows(phase_rows)
    result = {"all": summarize_tree_rows(rows), "by_phase": by_phase}
    if args.e0_dir:
        result["p1_root_prediction"] = summarize_p1_roots(
            load_p1_root_outcomes(args.e0_dir))
    if args.rerank_caps:
        caps = sorted({int(x) for x in args.rerank_caps.split(",") if x})
        if not caps or min(caps) < 1:
            ap.error("--rerank-caps must contain positive integers")
        result["rerank_cap_estimate"] = summarize_rerank_caps(
            args.trace_prefix, caps)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
