#!/usr/bin/env python3
"""Calibrate static DUET P1/P2 dynamic-tree expansion thresholds.

This tool deliberately calibrates *expansion* floors, not token deletion:

* P2 proxy floor: every retained P2 root is still evaluated once; only a
  later forward below a low-proxy root is suppressed;
* P1 start floor: the phase-1 counterpart — a root below the floor keeps its
  round-zero children as verifiable leaves but receives no deeper forwards;
* confidence floor (both phases): the sampled node remains a verifiable
  leaf; only a later forward below a low-q node is suppressed.

Inputs
------
P2 proxy labels are reconstructed from one or more E0 trace directories.
New traces contain ``proxy_piv`` in the draft selector record and are
therefore self-contained.  For historical traces without it, the script
joins the target wire by step/order.

Confidence labels come from one or more ``SSD_TREE_CALIB_TRACE`` JSONL
files.  Records carry a ``phase`` field, so one collection run yields both
the P1 and the P2 confidence axis.  The useful label is whether an accepted
child continued below a node.  Merely accepting the node itself is not
counted as a loss because the production floor keeps that node as a leaf.

P1 start labels come from ``SSD_TREE_TOPO_TRACE`` files: ``.draft.jsonl``
holds every built root's start score (the exact value compared against the
floor), and the ``.serve.jsonl``/``.walk.jsonl`` pair labels each served
tree with its accepted depth.  A start floor only suppresses depth>=2
expansion, so the loss label is a served tree whose accepted path reached
depth two or deeper.

The script reports both a near-lossless ``safe`` recommendation and a more
useful ``balanced`` recommendation.  The latter is the normal serving
candidate, but it should still receive one short engine A/B after changing
models, temperatures, or the tree budget.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PROXY_THRESHOLDS = (
    0.0001, 0.0003, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1,
)
DEFAULT_CONF_THRESHOLDS = (
    0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2,
)
# P1 start scores are ``context reach * root q`` products and live on a
# smaller scale than P2 target proxies, so the scan starts lower.
DEFAULT_START_THRESHOLDS = (
    0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1,
)


@dataclass(frozen=True)
class RiskCriteria:
    proxy_hit_contribution_max: float
    proxy_hit_rate_upper95_max: float
    confidence_use_contribution_max: float
    confidence_use_upper95_max: float
    minimum_tail_count: int = 100


RISK_PROFILES = {
    # Near-lossless diagnostic floor.  It is intentionally conservative and
    # is not automatically the best throughput/AL point.
    "safe": RiskCriteria(
        proxy_hit_contribution_max=0.01,
        proxy_hit_rate_upper95_max=0.002,
        confidence_use_contribution_max=0.01,
        confidence_use_upper95_max=0.005,
    ),
    # Current serving default: discard a large amount of low-value expansion
    # work while bounding the observed useful mass in the tail.  The limits
    # are explicit so a future experiment can audit or override the choice.
    "balanced": RiskCriteria(
        proxy_hit_contribution_max=0.05,
        proxy_hit_rate_upper95_max=0.003,
        confidence_use_contribution_max=0.015,
        confidence_use_upper95_max=0.005,
    ),
}


def wilson_upper(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = successes / n
    den = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    radius = z * math.sqrt(
        p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return min(1.0, (center + radius) / den)


def parse_thresholds(value: str) -> tuple[float, ...]:
    vals = sorted({float(x.strip()) for x in value.split(",") if x.strip()})
    if not vals or any(not 0.0 < x <= 1.0 for x in vals):
        raise argparse.ArgumentTypeError(
            "threshold list must contain comma-separated values in (0,1]")
    return tuple(vals)


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{lineno}: {exc}") from exc


def _clean_wire_token(token: int) -> int:
    """Strip the P_iv wire packing used by newer tree traces, if present."""
    token = int(token)
    if (token >> 31) & 1:
        return token & ((1 << 15) - 1)
    return token


def _trace_drop_count(paths: Iterable[Path]) -> tuple[int, bool]:
    drops = 0
    accounted = False
    for path in paths:
        for rec in _jsonl(path):
            if rec.get("kind") in ("heartbeat", "summary"):
                drops = max(drops, int(rec.get("drops", 0)))
                accounted = True
    return drops, accounted


def _target_wires(paths: list[Path]) -> dict[int, list[dict]]:
    """Return wire rows keyed by explicit wire_id or wire-record ordinal."""
    wires: dict[int, list[dict]] = {}
    ordinal = 0
    for path in sorted(paths):
        for rec in _jsonl(path):
            if rec.get("kind") != "wire":
                continue
            ordinal += 1
            wire_id = int(rec.get("wire_id", ordinal))
            rows = []
            for pos, tok, piv in zip(
                    rec["chosen_pos"], rec["chosen_tok"], rec["piv"]):
                rows.append({
                    "pos": [int(x) for x in pos],
                    "tok": [_clean_wire_token(x) for x in tok],
                    "piv": [float(x) for x in piv],
                })
            wires[wire_id] = rows
    return wires


def _selector_roots(rec: dict, wire_rows: list[dict] | None) -> list[list[dict]]:
    fan_rows = rec["proxy_fan_out"]
    tok_rows = rec["proxy_forked"]
    piv_rows = rec.get("proxy_piv")
    out = []
    for b, (fan, toks) in enumerate(zip(fan_rows, tok_rows)):
        pivs = piv_rows[b] if piv_rows is not None else None
        wire = wire_rows[b] if wire_rows is not None and b < len(wire_rows) \
            else None
        # Historical fallback: consume duplicate (position, token) entries in
        # wire rank order instead of collapsing them into a dict.
        wire_used: set[int] = set()
        roots = []
        i = 0
        for pos, count in enumerate(fan):
            for _ in range(int(count)):
                if i >= len(toks):
                    raise ValueError("selector fan-out exceeds proxy_forked width")
                token = int(toks[i])
                if pivs is not None:
                    score = float(pivs[i])
                elif wire is not None:
                    match = next((j for j, (p, t) in enumerate(zip(
                        wire["pos"], wire["tok"]))
                        if j not in wire_used and p == pos and t == token), None)
                    if match is None:
                        raise ValueError(
                            f"cannot join historical root ({pos},{token}) to target wire")
                    wire_used.add(match)
                    score = float(wire["piv"][match])
                else:
                    raise ValueError(
                        "selector has no proxy_piv and no matching target E0 wire")
                roots.append({"position": pos, "token": token, "score": score})
                i += 1
        out.append(roots)
    return out


def _load_one_e0_dir(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    draft_paths = sorted(path.glob("e0_draft_*.jsonl"))
    target_paths = sorted(path.glob("e0_target_*.jsonl"))
    if not draft_paths:
        # Accept a run directory and find its dedicated e0 subdirectory.
        draft_paths = sorted(path.glob("e0/e0_draft_*.jsonl"))
        target_paths = sorted(path.glob("e0/e0_target_*.jsonl"))
    if len(draft_paths) != 1:
        raise ValueError(
            f"{path}: expected exactly one e0_draft_*.jsonl, found "
            f"{len(draft_paths)}; keep each calibration run in its own directory")

    warnings = []
    drops, accounted = _trace_drop_count(draft_paths + target_paths)
    if drops:
        raise ValueError(f"{path}: E0 trace dropped {drops} records")
    if not accounted:
        warnings.append(
            f"{path}: trace has no heartbeat/summary drop counter; usable, "
            "but a cleanly closed trace is preferable")

    requests: dict[int, list[tuple[int, int, int]]] = {}
    selectors: dict[int, dict] = {}
    for rec in _jsonl(draft_paths[0]):
        kind = rec.get("kind")
        step = rec.get("step_id")
        if step is None:
            continue
        step = int(step)
        if kind == "request":
            requests[step] = [tuple(int(x) for x in row)
                              for row in rec["cache_keys"]]
        elif kind == "selector":
            selectors[step] = rec

    need_wire = any("proxy_piv" not in rec for rec in selectors.values())
    wires = _target_wires(target_paths) if need_wire else {}
    slots = []
    paired_steps = 0
    for step, rec in sorted(selectors.items()):
        build_keys = requests.get(step)
        next_keys = requests.get(step + 1)
        if build_keys is None or next_keys is None:
            continue
        next_by_seq = {row[0]: row for row in next_keys}
        roots_by_batch = _selector_roots(rec, wires.get(step))
        for b, roots in enumerate(roots_by_batch):
            if b >= len(build_keys):
                continue
            seq_id = build_keys[b][0]
            outcome = next_by_seq.get(seq_id)
            if outcome is None:
                continue
            paired_steps += 1
            outcome_seed = (outcome[1], outcome[2])
            for root in roots:
                slots.append({
                    "score": root["score"],
                    "hit": int((root["position"], root["token"])
                               == outcome_seed),
                    "source": str(path),
                    "step": step,
                    "seq_id": seq_id,
                })
    if not slots:
        raise ValueError(f"{path}: no selector -> next-request pairs found")
    warnings.append(
        f"{path}: paired proxy outcomes={paired_steps}, root slots={len(slots)}")
    return slots, warnings


def load_proxy_slots(e0_dirs: list[Path]) -> tuple[list[dict], list[str]]:
    slots, warnings = [], []
    for path in e0_dirs:
        one, notes = _load_one_e0_dir(path)
        slots.extend(one)
        warnings.extend(notes)
    return slots, warnings


def _resolve_confidence_paths(inputs: list[Path]) -> list[Path]:
    out = []
    for path in inputs:
        if path.is_dir():
            direct = sorted(path.glob("*.jsonl"))
            nested = sorted(path.glob("**/conf*.jsonl")) if not direct else []
            out.extend(direct or nested)
        else:
            out.append(path)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(out))


def load_confidence_nodes(paths: list[Path]) -> tuple[int, list[dict]]:
    nodes = []
    records = 0
    for path in _resolve_confidence_paths(paths):
        for rec in _jsonl(path):
            if "nodes" not in rec:
                continue
            records += 1
            policy = rec.get("policy", "unknown")
            phase = int(rec.get("phase", 0))
            accepted_child_parents = {
                int(node["parent"])
                for node in rec["nodes"] if node.get("on_path", False)
            }
            for node in rec["nodes"]:
                row = dict(node)
                row["policy"] = policy
                row["phase"] = phase
                row["source"] = str(path)
                row["expansion_useful"] = (
                    int(node["node"]) in accepted_child_parents)
                nodes.append(row)
    if not records:
        raise ValueError("no confidence records with a 'nodes' field found")
    return records, nodes


def load_p1_start_data(
        prefixes: list[Path]) -> tuple[list[float], list[dict]]:
    """Return (every built P1 root's start score, served P1 tree rows).

    The first list is the population the floor filters (occupancy); the
    second carries the loss label: a served tree whose accepted path reached
    depth >= 2 would lose those tokens under a floor above its start score.
    """
    from analyze_tree_outcomes import load_tree_pairs

    all_scores: list[float] = []
    for prefix in prefixes:
        draft_path = Path(str(prefix) + ".draft.jsonl")
        if not draft_path.exists():
            raise FileNotFoundError(draft_path)
        for rec in _jsonl(draft_path):
            if int(rec.get("phase", 0)) != 1:
                continue
            for root in rec["roots"]:
                all_scores.append(float(root["piv"]))
    served = []
    for row in load_tree_pairs(prefixes):
        if int(row["phase"]) != 1:
            continue
        if row["root_start_score"] is None:
            raise ValueError("P1 serve record lacks root_start_score")
        served.append({
            "score": float(row["root_start_score"]),
            "deep": int(row["accepted"]) >= 2,
        })
    if not all_scores:
        raise ValueError("topo trace contains no phase-1 draft roots")
    if not served:
        raise ValueError("topo trace contains no served phase-1 trees")
    return all_scores, served


def start_table(all_scores: list[float], served: list[dict],
                thresholds: tuple[float, ...]) -> list[dict]:
    """Scan start floors against the built-root population.

    ``deep_rate`` is the unconditional per-built-root probability that this
    root was served *and* accepted to depth >= 2 — the same per-slot
    realization semantics as the P2 proxy ``hit_rate``, so the proxy risk
    limits transfer.  The served-conditional rate is kept as a column for
    reading, not for the decision.
    """
    total_deep = sum(bool(x["deep"]) for x in served)
    rows = []
    for threshold in thresholds:
        low_all = sum(1 for s in all_scores if s < threshold)
        low = [x for x in served if x["score"] < threshold]
        deep = sum(bool(x["deep"]) for x in low)
        rows.append({
            "threshold": threshold,
            "n": low_all,
            "total_n": len(all_scores),
            "occupancy": low_all / len(all_scores) if all_scores else 0.0,
            "served_n": len(low),
            "served_total": len(served),
            "deep": deep,
            "total_deep": total_deep,
            "deep_rate": deep / low_all if low_all else 0.0,
            "deep_rate_upper95": wilson_upper(deep, low_all),
            "deep_rate_served": deep / len(low) if low else 0.0,
            "deep_contribution": deep / total_deep if total_deep else 0.0,
        })
    return rows


def proxy_table(slots: list[dict], thresholds: tuple[float, ...]) -> list[dict]:
    total_hits = sum(int(x["hit"]) for x in slots)
    rows = []
    for threshold in thresholds:
        low = [x for x in slots if float(x["score"]) < threshold]
        hits = sum(int(x["hit"]) for x in low)
        rows.append({
            "threshold": threshold,
            "n": len(low),
            "total_n": len(slots),
            "occupancy": len(low) / len(slots) if slots else 0.0,
            "hits": hits,
            "total_hits": total_hits,
            "hit_rate": hits / len(low) if low else 0.0,
            "hit_rate_upper95": wilson_upper(hits, len(low)),
            "hit_contribution": hits / total_hits if total_hits else 0.0,
        })
    return rows


def confidence_table(nodes: list[dict], depth_cap: int,
                     thresholds: tuple[float, ...]) -> list[dict]:
    candidates = [x for x in nodes if int(x["depth"]) < depth_cap]
    total_path = sum(bool(x["on_path"]) for x in candidates)
    total_useful = sum(bool(x["expansion_useful"]) for x in candidates)
    rows = []
    for threshold in thresholds:
        low = [x for x in candidates if float(x["q"]) < threshold]
        on_path = sum(bool(x["on_path"]) for x in low)
        useful = sum(bool(x["expansion_useful"]) for x in low)
        attempted = [x for x in low if x["attempted"]]
        accepted = sum(bool(x["accepted"]) for x in attempted)
        rows.append({
            "threshold": threshold,
            "n": len(low),
            "candidate_n": len(candidates),
            "total_useful": total_useful,
            "occupancy": len(low) / len(candidates) if candidates else 0.0,
            "on_path_rate": on_path / len(low) if low else 0.0,
            "on_path_upper95": wilson_upper(on_path, len(low)),
            "path_contribution": on_path / total_path if total_path else 0.0,
            "expansion_useful": useful,
            "expansion_use_rate": useful / len(low) if low else 0.0,
            "expansion_use_upper95": wilson_upper(useful, len(low)),
            "expansion_use_contribution": (
                useful / total_useful if total_useful else 0.0),
            "attempted_n": len(attempted),
            "attempt_accept_rate": accepted / len(attempted)
            if attempted else 0.0,
            "attempt_accept_upper95": wilson_upper(accepted, len(attempted)),
        })
    return rows


def choose_proxy(rows: list[dict], criteria: RiskCriteria) -> dict | None:
    good = [r for r in rows
            if r["n"] >= criteria.minimum_tail_count
            and r["hit_contribution"] <= criteria.proxy_hit_contribution_max
            and r["hit_rate_upper95"] <= criteria.proxy_hit_rate_upper95_max]
    return max(good, key=lambda x: x["threshold"], default=None)


def choose_confidence(rows: list[dict], criteria: RiskCriteria) -> dict | None:
    good = [r for r in rows
            if r["n"] >= criteria.minimum_tail_count
            and r["expansion_use_contribution"]
            <= criteria.confidence_use_contribution_max
            and r["expansion_use_upper95"]
            <= criteria.confidence_use_upper95_max]
    return max(good, key=lambda x: x["threshold"], default=None)


def choose_start(rows: list[dict], criteria: RiskCriteria) -> dict | None:
    # The P1 start floor is a root-level filter with the same per-slot
    # realization semantics as the P2 proxy floor, so it reuses the proxy
    # risk limits against the built-root population.
    good = [r for r in rows
            if r["n"] >= criteria.minimum_tail_count
            and r["deep_contribution"] <= criteria.proxy_hit_contribution_max
            and r["deep_rate_upper95"]
            <= criteria.proxy_hit_rate_upper95_max]
    return max(good, key=lambda x: x["threshold"], default=None)


def pct(x: float) -> str:
    return f"{100.0 * x:.3f}%"


def _profile_dict(criteria: RiskCriteria) -> dict:
    return {
        "proxy_hit_contribution_max": criteria.proxy_hit_contribution_max,
        "proxy_hit_rate_upper95_max": criteria.proxy_hit_rate_upper95_max,
        "confidence_use_contribution_max":
            criteria.confidence_use_contribution_max,
        "confidence_use_upper95_max": criteria.confidence_use_upper95_max,
        "minimum_tail_count": criteria.minimum_tail_count,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--e0-dir", type=Path, nargs="+",
        help="one or more E0 run/e0 directories used for proxy outcomes")
    ap.add_argument("--confidence", type=Path, nargs="+", required=True,
                    help="confidence JSONL files or directories")
    ap.add_argument("--depth-cap", type=int, default=4,
                    help="P2 rounds; only nodes with depth < cap can expand")
    ap.add_argument("--topo-prefix", type=Path, nargs="*", default=[],
                    help="SSD_TREE_TOPO_TRACE prefixes (.draft/.serve/.walk) "
                         "enabling the P1 start axis; omit for legacy "
                         "P2-only calibration")
    ap.add_argument("--p1-depth-cap", type=int, default=9,
                    help="P1 rounds (K1); only nodes with depth < cap "
                         "can expand")
    ap.add_argument("--risk-profile", choices=tuple(RISK_PROFILES),
                    default="balanced")
    ap.add_argument("--proxy-thresholds", type=parse_thresholds,
                    default=DEFAULT_PROXY_THRESHOLDS)
    ap.add_argument("--confidence-thresholds", type=parse_thresholds,
                    default=DEFAULT_CONF_THRESHOLDS)
    ap.add_argument("--start-thresholds", type=parse_thresholds,
                    default=DEFAULT_START_THRESHOLDS)
    ap.add_argument("--min-proxy-slots", type=int, default=10000)
    ap.add_argument("--min-proxy-hits", type=int, default=100)
    ap.add_argument("--min-confidence-candidates", type=int, default=1000)
    ap.add_argument("--min-useful-expansions", type=int, default=100)
    ap.add_argument("--min-p1-roots", type=int, default=10000)
    ap.add_argument("--min-p1-deep-trees", type=int, default=100)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--config-out", type=Path,
                    help="write sourceable TREE_*_THRESHOLD assignments")
    ap.add_argument("--strict", action="store_true",
                    help="return non-zero on insufficient data/recommendation")
    args = ap.parse_args(argv)

    if args.depth_cap <= 1:
        ap.error("--depth-cap must be greater than one")
    e0_dirs = args.e0_dir
    if not e0_dirs:
        # Backward compatibility for the original one-off analysis.  New
        # runs must pass --e0-dir explicitly so settings cannot be confused.
        repo_ssd = Path(__file__).resolve().parents[3]
        e0_dirs = [repo_ssd / "experiments/proxy_async_overlap/e0_collect/run1"]
        print("[warning] --e0-dir omitted; using legacy E0 run1", file=sys.stderr)

    slots, trace_notes = load_proxy_slots(e0_dirs)
    records, nodes = load_confidence_nodes(args.confidence)
    # Phase-split confidence: phase==1 nodes come from served P1 trees; any
    # other value (2, or 0 in historical P2-only traces) is the P2 axis.
    nodes_p1 = [x for x in nodes if int(x.get("phase", 0)) == 1]
    nodes_p2 = [x for x in nodes if int(x.get("phase", 0)) != 1]
    proxy = proxy_table(slots, args.proxy_thresholds)
    confidence = confidence_table(nodes_p2, args.depth_cap,
                                  args.confidence_thresholds)
    p1_enabled = bool(args.topo_prefix)
    p1_confidence = (confidence_table(nodes_p1, args.p1_depth_cap,
                                      args.confidence_thresholds)
                     if p1_enabled else [])
    if p1_enabled:
        p1_scores, p1_served = load_p1_start_data(args.topo_prefix)
        p1_start = start_table(p1_scores, p1_served, args.start_thresholds)
    else:
        p1_scores, p1_served, p1_start = [], [], []

    data_warnings = list(trace_notes)
    proxy_hits = sum(int(x["hit"]) for x in slots)
    confidence_candidates = confidence[0]["candidate_n"] if confidence else 0
    useful_expansions = confidence[0]["total_useful"] if confidence else 0
    enough = True
    checks = [
        (len(slots) >= args.min_proxy_slots,
         f"proxy slots {len(slots)} < {args.min_proxy_slots}"),
        (proxy_hits >= args.min_proxy_hits,
         f"proxy hits {proxy_hits} < {args.min_proxy_hits}"),
        (confidence_candidates >= args.min_confidence_candidates,
         f"confidence candidates {confidence_candidates} "
         f"< {args.min_confidence_candidates}"),
        (useful_expansions >= args.min_useful_expansions,
         f"useful expansions {useful_expansions} "
         f"< {args.min_useful_expansions}"),
    ]
    p1_conf_candidates = p1_confidence[0]["candidate_n"] \
        if p1_confidence else 0
    p1_useful = p1_confidence[0]["total_useful"] if p1_confidence else 0
    p1_deep_trees = p1_start[0]["total_deep"] if p1_start else 0
    if p1_enabled:
        checks += [
            (len(p1_scores) >= args.min_p1_roots,
             f"P1 built roots {len(p1_scores)} < {args.min_p1_roots}"),
            (p1_deep_trees >= args.min_p1_deep_trees,
             f"P1 deep-accepted trees {p1_deep_trees} "
             f"< {args.min_p1_deep_trees}"),
            (p1_conf_candidates >= args.min_confidence_candidates,
             f"P1 confidence candidates {p1_conf_candidates} "
             f"< {args.min_confidence_candidates}"),
            (p1_useful >= args.min_useful_expansions,
             f"P1 useful expansions {p1_useful} "
             f"< {args.min_useful_expansions}"),
        ]
    for ok, message in checks:
        if not ok:
            enough = False
            data_warnings.append("INSUFFICIENT: " + message)

    recommendations = {}
    for name, criteria in RISK_PROFILES.items():
        p_rec = choose_proxy(proxy, criteria) if enough else None
        q_rec = choose_confidence(confidence, criteria) if enough else None
        s_rec = (choose_start(p1_start, criteria)
                 if enough and p1_enabled else None)
        q1_rec = (choose_confidence(p1_confidence, criteria)
                  if enough and p1_enabled else None)
        recommendations[name] = {
            "proxy_min": p_rec["threshold"] if p_rec else None,
            "confidence_min": q_rec["threshold"] if q_rec else None,
            "p1_start_min": s_rec["threshold"] if s_rec else None,
            "p1_confidence_min": q1_rec["threshold"] if q1_rec else None,
        }

    selected = recommendations[args.risk_profile]
    print(f"[data] proxy slots={len(slots)} hits={proxy_hits}; "
          f"confidence records={records} nodes={len(nodes)} "
          f"(P1 {len(nodes_p1)} / P2 {len(nodes_p2)}) "
          f"P2 expandable={confidence_candidates} useful={useful_expansions}")
    if p1_enabled:
        print(f"[data] P1 built roots={len(p1_scores)} "
              f"served trees={len(p1_served)} deep trees={p1_deep_trees}; "
              f"P1 expandable={p1_conf_candidates} useful={p1_useful}")
    for note in data_warnings:
        print(f"[note] {note}")

    print("\n[proxy: score below threshold]")
    print("threshold  occupancy  hit-rate  upper95  hit-contribution")
    for r in proxy:
        print(f"{r['threshold']:9g}  {pct(r['occupancy']):>9}  "
              f"{pct(r['hit_rate']):>8}  {pct(r['hit_rate_upper95']):>7}  "
              f"{pct(r['hit_contribution']):>16}")

    def _print_confidence(rows, title):
        print(f"\n[{title}: q below threshold, expandable nodes only]")
        print("threshold  occupancy  useful  upper95  useful-contrib  "
              "on-path  attempted  accept|attempt")
        for r in rows:
            print(f"{r['threshold']:9g}  {pct(r['occupancy']):>9}  "
                  f"{pct(r['expansion_use_rate']):>7}  "
                  f"{pct(r['expansion_use_upper95']):>7}  "
                  f"{pct(r['expansion_use_contribution']):>14}  "
                  f"{pct(r['on_path_rate']):>7}  "
                  f"{r['attempted_n']:9d}  "
                  f"{pct(r['attempt_accept_rate']):>14}")

    _print_confidence(confidence, "P2 confidence")
    if p1_enabled:
        print("\n[P1 start: built-root score below threshold]")
        print("threshold  occupancy  deep-rate  upper95  deep-contrib  "
              "served-n  deep|served")
        for r in p1_start:
            print(f"{r['threshold']:9g}  {pct(r['occupancy']):>9}  "
                  f"{pct(r['deep_rate']):>9}  "
                  f"{pct(r['deep_rate_upper95']):>7}  "
                  f"{pct(r['deep_contribution']):>12}  "
                  f"{r['served_n']:8d}  "
                  f"{pct(r['deep_rate_served']):>11}")
        _print_confidence(p1_confidence, "P1 confidence")

    print("\n[static recommendations]")
    for name in ("safe", "balanced"):
        rec = recommendations[name]
        line = (f"{name:8s} proxy={rec['proxy_min']} "
                f"confidence={rec['confidence_min']}")
        if p1_enabled:
            line += (f" p1_start={rec['p1_start_min']} "
                     f"p1_confidence={rec['p1_confidence_min']}")
        print(line)
    print(f"selected profile = {args.risk_profile}")
    if p1_enabled and enough:
        for axis in ("p1_start_min", "p1_confidence_min"):
            if selected[axis] is None:
                # P1's production default is 0.0; "no floor passes the risk
                # limits" is a valid calibration answer, not missing data.
                print(f"[note] {axis}: no candidate passed the risk "
                      "limits; recommending 0.0 (keep unfiltered)")
                selected[axis] = 0.0

    result = {
        "data": {
            "e0_dirs": [str(x) for x in e0_dirs],
            "confidence_inputs": [str(x) for x in args.confidence],
            "topo_prefixes": [str(x) for x in args.topo_prefix],
            "proxy_slots": len(slots),
            "proxy_hits": proxy_hits,
            "confidence_records": records,
            "confidence_nodes": len(nodes),
            "confidence_nodes_p1": len(nodes_p1),
            "confidence_candidates": confidence_candidates,
            "useful_expansions": useful_expansions,
            "p1_built_roots": len(p1_scores),
            "p1_served_trees": len(p1_served),
            "p1_deep_trees": p1_deep_trees,
            "p1_confidence_candidates": p1_conf_candidates,
            "p1_useful_expansions": p1_useful,
            "depth_cap": args.depth_cap,
            "p1_depth_cap": args.p1_depth_cap,
            "sufficient": enough,
            "notes": data_warnings,
        },
        "criteria": {name: _profile_dict(value)
                     for name, value in RISK_PROFILES.items()},
        "proxy": proxy,
        "confidence": confidence,
        "p1_start": p1_start,
        "p1_confidence": p1_confidence,
        "recommendations": recommendations,
        "selected_profile": args.risk_profile,
        "recommendation": selected,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")

    has_rec = (selected["proxy_min"] is not None
               and selected["confidence_min"] is not None)
    if p1_enabled:
        has_rec = has_rec and (selected["p1_start_min"] is not None
                               and selected["p1_confidence_min"] is not None)
    if args.config_out:
        if not enough or not has_rec:
            print("[warning] config not written: data/recommendation gate failed",
                  file=sys.stderr)
        else:
            args.config_out.parent.mkdir(parents=True, exist_ok=True)
            # Variable names match run_p1_p2_tree_formal_20260807.sh inputs.
            lines = [
                "# Generated by analyze_thresholds.py",
                f"# risk_profile={args.risk_profile}",
                f"TREE_PROXY_THRESHOLD={selected['proxy_min']:.9g}",
                f"TREE_CONF_THRESHOLD={selected['confidence_min']:.9g}",
            ]
            if p1_enabled:
                lines += [
                    f"P1_START_THRESHOLD={selected['p1_start_min']:.9g}",
                    f"P1_CONF_THRESHOLD={selected['p1_confidence_min']:.9g}",
                ]
            args.config_out.write_text("\n".join(lines) + "\n")
            print(f"config written: {args.config_out}")

    if args.strict and (not enough or not has_rec):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
