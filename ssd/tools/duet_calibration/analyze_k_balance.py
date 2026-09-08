#!/usr/bin/env python3
"""Measure and recommend DUET K1/K2 rendezvous points from timeline traces.

K1 gap (positive means draft finished P1 too early):

    estimated proxy arrival on draft - draft P1 ready

The proxy transport delay is estimated from steps where ``proxy_wait`` really
blocked.  K2 gap (positive means draft cache finished too early):

    target next-request ready - draft cache ready

Negative K2 gap means the target issued the next request before the draft
cache was ready and therefore exposes target-side waiting.  The recommender
minimizes the median absolute signed gap; candidates within a small tie window
prefer the larger K to retain proposal quality.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    k1: int
    k2: int
    path: Path


def parse_run(value: str) -> RunSpec:
    try:
        k1_s, k2_s, path = value.split(",", 2)
        result = RunSpec(int(k1_s), int(k2_s), Path(path))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "--run must be K1,K2,/path/to/profile_dir") from exc
    if result.k1 < 1 or result.k2 < 1 or result.k2 > result.k1:
        raise argparse.ArgumentTypeError("require K1>=K2>=1")
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return x[lo]
    return x[lo] * (hi - pos) + x[hi] * (pos - lo)


def _wall_start(event: dict) -> float:
    if "wall_start_ns" not in event:
        raise ValueError("profile lacks wall_start_ns; cross-process balance is unavailable")
    return float(event["wall_start_ns"]) / 1e6


def _wall_end(event: dict) -> float:
    if "wall_end_ns" not in event:
        raise ValueError("profile lacks wall_end_ns; cross-process balance is unavailable")
    return float(event["wall_end_ns"]) / 1e6


def _find_profile(path: Path, role: str) -> Path:
    matches = sorted(path.glob(f"duet_profile_{role}_*.json"))
    if len(matches) != 1:
        raise ValueError(
            f"{path}: expected one duet_profile_{role}_*.json, found {len(matches)}")
    return matches[0]


def _events_with_keys(events: list[dict], label: str,
                      skip_steps: int) -> list[dict]:
    """Attach a request-epoch key to one label's chronological events.

    ``bench.py`` normally emits one monotonically increasing step sequence,
    while ``run_duet.py`` calls ``generate()`` repeatedly and resets step_id
    to one for every warmup/request.  A plain ``dict[step_id]`` silently
    paired the last draft occurrence with a different target request.  The
    profiler preserves event order and both processes observe identical
    resets, so ``(epoch, step_id)`` is the stable cross-process key.
    """
    out = []
    epoch = 0
    previous = None
    for event in events:
        sid = event.get("step_id")
        if sid is None or event.get("label") != label:
            continue
        sid = int(sid)
        if previous is not None and sid < previous:
            epoch += 1
        previous = sid
        if sid <= skip_steps:
            continue
        tagged = dict(event)
        tagged["_profile_key"] = (epoch, sid)
        out.append(tagged)
    return out


def _events_by_step(events: list[dict], label: str,
                    skip_steps: int) -> dict[tuple[int, int], dict]:
    out = {}
    for event in _events_with_keys(events, label, skip_steps):
        key = event["_profile_key"]
        # The selected labels should occur once.  If a detailed profiler emits
        # a nested duplicate, retain the widest enclosing event.
        prev = out.get(key)
        if prev is None or (_wall_end(event) - _wall_start(event)) > (
                _wall_end(prev) - _wall_start(prev)):
            out[key] = event
    return out


def _load_raw(spec: RunSpec, skip_steps: int) -> dict:
    draft_path = _find_profile(spec.path, "draft")
    target_path = _find_profile(spec.path, "target_rank0")
    draft = json.loads(draft_path.read_text())
    target = json.loads(target_path.read_text())
    proxy_send = _events_by_step(
        target, "proxy_send_enqueue", skip_steps)
    # Profiles written before the proxy split used one enclosing label.
    for sid, event in _events_by_step(
            target, "proxy_compute_send", skip_steps).items():
        proxy_send.setdefault(sid, event)
    return {
        "spec": spec,
        "draft_path": draft_path,
        "target_path": target_path,
        "proxy_wait": _events_by_step(draft, "proxy_wait", skip_steps),
        "phase1_build": _events_by_step(draft, "phase1_build", skip_steps),
        "phase1_replay_all": _events_with_keys(
            draft, "phase1_replay", skip_steps),
        "phase2_replay_all": _events_with_keys(
            draft, "phase2_replay", skip_steps),
        "p1_graph": _events_by_step(
            draft, "p1_graph_replay", skip_steps),
        "p2_graph": _events_by_step(draft, "p2_graph_replay", skip_steps),
        "merge": _events_by_step(draft, "merge_cache", skip_steps),
        "proxy_send": proxy_send,
        "next_send": _events_by_step(target, "target_send_request", skip_steps),
    }


def estimate_proxy_transport(raw_runs: list[dict], fallback_ms: float) -> tuple[float, int]:
    samples = []
    for raw in raw_runs:
        for sid, wait in raw["proxy_wait"].items():
            send = raw["proxy_send"].get(sid)
            if send is None:
                continue
            blocked = _wall_end(wait) - _wall_start(wait)
            lag = _wall_end(wait) - _wall_end(send)
            # Only a genuinely blocked wait exposes arrival time.  Very large
            # values are misalignment/startup artifacts, not transport.
            if blocked >= 0.2 and 0.0 <= lag <= 10.0:
                samples.append(lag)
    return (st.median(samples), len(samples)) if samples else (fallback_ms, 0)


def _metric(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _log_metrics(path: Path) -> dict:
    log = path / "run.log"
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    return {
        "tps": _metric(text, r"Final Decode Throughput:\s*([\d.]+)"),
        "tokens_per_step": _metric(
            text, r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)"),
        "p1_hit": _metric(text, r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)"),
        "p2_hit": _metric(text, r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)"),
        "p1_al": _metric(text, r"Avg Phase 1 Accepted Len:\s*([\d.]+)"),
        "p2_al": _metric(text, r"Avg Phase 2 Accepted Len:\s*([\d.]+)"),
    }


def analyze_run(raw: dict, proxy_transport_ms: float) -> dict:
    spec: RunSpec = raw["spec"]
    k1_gaps = []
    wait_durations = []
    for sid, wait in raw["proxy_wait"].items():
        send = raw["proxy_send"].get(sid)
        if send is None:
            continue
        arrival = _wall_end(send) + proxy_transport_ms
        k1_gaps.append(arrival - _wall_start(wait))
        wait_durations.append(max(0.0, _wall_end(wait) - _wall_start(wait)))

    k2_gaps = []
    for (epoch, sid), merge in raw["merge"].items():
        send = raw["next_send"].get((epoch, sid + 1))
        if send is None:
            continue
        k2_gaps.append(_wall_start(send) - _wall_end(merge))

    # Median effective time added by one more P1/P2 round.  This is used only
    # to predict a small local candidate set; final selection uses measured
    # gaps, not the linear model.
    if raw["p1_graph"]:
        p1_round = [(_wall_end(x) - _wall_start(x)) / spec.k1
                    for x in raw["p1_graph"].values()]
    else:
        replay_by_step: dict[tuple[int, int], list[dict]] = {}
        for event in raw["phase1_replay_all"]:
            replay_by_step.setdefault(
                event["_profile_key"], []).append(event)
        p1_round = []
        for events in replay_by_step.values():
            if events:
                span = max(_wall_end(x) for x in events) - min(
                    _wall_start(x) for x in events)
                p1_round.append(span / spec.k1)
    # The optimized tree path records the complete P2 executor as one graph.
    # Chain records K2 individual phase2_replay events.  Supporting both here
    # is what makes the same rendezvous calibration valid for either policy.
    if raw["p2_graph"]:
        p2_round = [(_wall_end(x) - _wall_start(x)) / spec.k2
                    for x in raw["p2_graph"].values()]
        p2_timing_source = "p2_graph_replay"
    else:
        phase2_by_step: dict[tuple[int, int], list[dict]] = {}
        for event in raw["phase2_replay_all"]:
            phase2_by_step.setdefault(
                event["_profile_key"], []).append(event)
        p2_round = []
        for events in phase2_by_step.values():
            if events:
                span = max(_wall_end(x) for x in events) - min(
                    _wall_start(x) for x in events)
                p2_round.append(span / spec.k2)
        p2_timing_source = "phase2_replay"

    def summary(values: list[float]) -> dict:
        overruns = [max(-x, 0.0) for x in values]
        late = [x for x in values if x < 0.0]
        return {
            "n": len(values),
            "p01_ms": percentile(values, 0.01),
            "p05_ms": percentile(values, 0.05),
            "p10_ms": percentile(values, 0.10),
            "p50_ms": percentile(values, 0.50),
            "p90_ms": percentile(values, 0.90),
            "p95_ms": percentile(values, 0.95),
            "p99_ms": percentile(values, 0.99),
            "p50_abs_ms": percentile([abs(x) for x in values], 0.50),
            "p90_abs_ms": percentile([abs(x) for x in values], 0.90),
            # Negative signed gap means the producer missed its overlap
            # deadline: P1 ended after proxy arrival, or P2 cache became
            # ready after the target issued the next request.  Tail frequency
            # is the deployment gate; a positive mean can hide rare stalls.
            "late_n": len(late),
            "late_rate": len(late) / len(values) if values else math.nan,
            "overrun_p90_ms": percentile(overruns, 0.90),
            "overrun_p95_ms": percentile(overruns, 0.95),
            "overrun_p99_ms": percentile(overruns, 0.99),
            "overrun_max_ms": max(overruns) if overruns else math.nan,
            "draft_wait_mean_ms": st.mean(max(x, 0.0) for x in values)
            if values else math.nan,
            "target_wait_mean_ms": st.mean(max(-x, 0.0) for x in values)
            if values else math.nan,
        }

    k1_summary = summary(k1_gaps)
    k2_summary = summary(k2_gaps)
    p1_round_ms = percentile(p1_round, 0.50)
    p2_round_ms = percentile(p2_round, 0.50)
    pred_k1 = spec.k1
    pred_k2 = spec.k2
    if math.isfinite(p1_round_ms) and p1_round_ms > 0 and k1_gaps:
        pred_k1 = max(spec.k2, round(
            spec.k1 + k1_summary["p50_ms"] / p1_round_ms))
    if math.isfinite(p2_round_ms) and p2_round_ms > 0 and k2_gaps:
        pred_k2 = min(spec.k1, max(1, round(
            spec.k2 + k2_summary["p50_ms"] / p2_round_ms)))

    return {
        "path": str(spec.path),
        "k1": spec.k1,
        "k2": spec.k2,
        "proxy_transport_ms": proxy_transport_ms,
        "k1_gap": k1_summary,
        "k2_gap": k2_summary,
        "p1_round_ms": p1_round_ms,
        "p2_round_ms": p2_round_ms,
        "p2_timing_source": p2_timing_source,
        "predicted_k1": pred_k1,
        "predicted_k2": pred_k2,
        "observed_proxy_wait_p50_ms": percentile(wait_durations, 0.50),
        "quality": _log_metrics(spec.path),
    }


def choose(rows: list[dict], dim: str, min_steps: int,
           tie_ms: float, preferred_k1: int | None = None) -> dict | None:
    key = f"{dim}_gap"
    candidates = [x for x in rows if x[key]["n"] >= min_steps]
    if dim == "k2" and preferred_k1 is not None:
        candidates = [x for x in candidates if x["k1"] == preferred_k1]
    if not candidates:
        return None
    best_abs = min(x[key]["p50_abs_ms"] for x in candidates)
    near = [x for x in candidates
            if x[key]["p50_abs_ms"] <= best_abs + tie_ms]
    # Preserve proposal depth inside a true measurement tie.
    return max(near, key=lambda x: (x[dim], -abs(x[key]["p50_ms"])))


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", type=parse_run, required=True,
                    help="repeat K1,K2,/profile/directory")
    ap.add_argument("--skip-steps", type=int, default=10)
    ap.add_argument("--min-steps", type=int, default=30)
    ap.add_argument("--proxy-transport-fallback-ms", type=float, default=1.5)
    ap.add_argument("--tie-ms", type=float, default=0.25)
    ap.add_argument("--preferred-k1", type=int)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    raw = [_load_raw(x, args.skip_steps) for x in args.run]
    proxy_transport, transport_n = estimate_proxy_transport(
        raw, args.proxy_transport_fallback_ms)
    rows = [analyze_run(x, proxy_transport) for x in raw]
    k1_rec = choose(rows, "k1", args.min_steps, args.tie_ms)
    selected_k1 = args.preferred_k1 or (k1_rec["k1"] if k1_rec else None)
    k2_rec = choose(rows, "k2", args.min_steps, args.tie_ms, selected_k1)

    print(f"[proxy transport] {proxy_transport:.3f} ms "
          f"(blocked-wait samples={transport_n})")
    print("K1 K2 | K1 gap p50 / |gap|p50 / n | "
          "K2 gap p50 / |gap|p50 / n | round ms | P1AL P2AL TPS")
    for row in sorted(rows, key=lambda x: (x["k1"], x["k2"])):
        a, b, q = row["k1_gap"], row["k2_gap"], row["quality"]
        print(f"{row['k1']:2d} {row['k2']:2d} | "
              f"{a['p50_ms']:+7.3f} {a['p50_abs_ms']:7.3f} {a['n']:4d} | "
              f"{b['p50_ms']:+7.3f} {b['p50_abs_ms']:7.3f} {b['n']:4d} | "
              f"{row['p1_round_ms']:.3f}/{row['p2_round_ms']:.3f} | "
              f"{q.get('p1_al')} {q.get('p2_al')} {q.get('tps')}")
        print(
            "      tails | "
            f"P1 late={a['late_n']}/{a['n']} ({a['late_rate']:.3%}) "
            f"p01={a['p01_ms']:+.3f} overrun-p99={a['overrun_p99_ms']:.3f} | "
            f"P2 late={b['late_n']}/{b['n']} ({b['late_rate']:.3%}) "
            f"p01={b['p01_ms']:+.3f} overrun-p99={b['overrun_p99_ms']:.3f}")
    print("\n[recommendation]")
    print("K1 =", selected_k1 if selected_k1 is not None else "NONE")
    print("K2 =", k2_rec["k2"] if k2_rec else "NONE")
    if len(rows) == 1:
        print("local prediction from this run: K1=", rows[0]["predicted_k1"],
              "K2=", rows[0]["predicted_k2"])

    result = {
        "proxy_transport_ms": proxy_transport,
        "proxy_transport_samples": transport_n,
        "runs": rows,
        "recommendation": {
            "k1": selected_k1,
            "k2": k2_rec["k2"] if k2_rec else None,
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    ok = selected_k1 is not None and k2_rec is not None
    return 2 if args.strict and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
