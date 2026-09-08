import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1]
          / "tools/duet_calibration/analyze_k_balance.py")
SPEC = importlib.util.spec_from_file_location("k_balance", SCRIPT)
KB = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = KB
SPEC.loader.exec_module(KB)


def _event(step, label, start_ms, end_ms):
    return {
        "step_id": step, "label": label, "status": "hit_k1",
        "wall_start_ns": int(start_ms * 1e6),
        "wall_end_ns": int(end_ms * 1e6),
    }


def _profile(tmp_path, name, k1, k2, k1_gap, k2_gap):
    path = tmp_path / name
    path.mkdir()
    draft, target = [], []
    transport = 1.5
    for step in range(11, 71):
        base = step * 100.0
        proxy_end = base + 40.0
        p1_ready = proxy_end + transport - k1_gap
        proxy_arrival = proxy_end + transport
        wait_end = max(p1_ready, proxy_arrival)
        target.append(_event(step, "proxy_compute_send",
                             proxy_end - 0.1, proxy_end))
        draft.append(_event(step, "proxy_wait", p1_ready, wait_end))
        for r in range(k1):
            draft.append(_event(step, "phase1_replay",
                                base + 5 + 2 * r, base + 7 + 2 * r))
        draft.append(_event(step, "p2_graph_replay",
                            base + 50, base + 50 + 3 * k2))
        merge_end = base + 80.0
        draft.append(_event(step, "merge_cache", merge_end - 0.1, merge_end))
        target.append(_event(step + 1, "target_send_request",
                             merge_end + k2_gap, merge_end + k2_gap + 0.1))
    (path / "duet_profile_draft_test.json").write_text(json.dumps(draft))
    (path / "duet_profile_target_rank0_test.json").write_text(json.dumps(target))
    (path / "run.log").write_text(
        "Final Decode Throughput: 70.0\n"
        "Avg Tokens per step (incl recovery): 4.1\n"
        "Avg Phase 1 (draft) Hit Rate: 0.58\n"
        "Avg Phase 2 (proxy) Hit Rate: 0.24\n"
        "Avg Phase 1 Accepted Len: 4.0\n"
        "Avg Phase 2 Accepted Len: 1.8\n")
    return path


def test_signed_gaps_and_recommendations(tmp_path):
    specs = []
    for k1, gap in ((8, 3.0), (9, 0.0), (10, -3.0)):
        path = _profile(tmp_path, f"k1_{k1}", k1, 4, gap, 0.0)
        specs.append(KB.RunSpec(k1, 4, path))
    for k2, gap in ((3, 2.0), (5, -2.0)):
        path = _profile(tmp_path, f"k2_{k2}", 9, k2, 0.0, gap)
        specs.append(KB.RunSpec(9, k2, path))

    raw = [KB._load_raw(x, skip_steps=10) for x in specs]
    transport, n = KB.estimate_proxy_transport(raw, fallback_ms=9.0)
    assert transport == 1.5
    assert n > 0
    rows = [KB.analyze_run(x, transport) for x in raw]

    k1 = KB.choose([x for x in rows if x["k2"] == 4],
                   "k1", min_steps=30, tie_ms=0.1)
    k2 = KB.choose(rows, "k2", min_steps=30, tie_ms=0.1,
                   preferred_k1=9)
    assert k1["k1"] == 9
    assert k2["k2"] == 4
    balanced = next(x for x in rows if x["k1"] == 9 and x["k2"] == 4)
    assert abs(balanced["k1_gap"]["p50_ms"]) < 1e-6
    assert abs(balanced["k2_gap"]["p50_ms"]) < 1e-6


def test_chain_phase2_replays_supply_round_time(tmp_path):
    path = _profile(tmp_path, "chain", 9, 4, 0.0, 0.0)
    draft_file = path / "duet_profile_draft_test.json"
    draft = json.loads(draft_file.read_text())
    rewritten = []
    for event in draft:
        if event["label"] != "p2_graph_replay":
            rewritten.append(event)
            continue
        start = event["wall_start_ns"] / 1e6
        for round_idx in range(4):
            rewritten.append(_event(
                event["step_id"], "phase2_replay",
                start + 3 * round_idx, start + 3 * (round_idx + 1)))
    draft_file.write_text(json.dumps(rewritten))

    raw = KB._load_raw(KB.RunSpec(9, 4, path), skip_steps=10)
    row = KB.analyze_run(raw, proxy_transport_ms=1.5)
    assert row["p2_timing_source"] == "phase2_replay"
    assert row["p2_round_ms"] == 3.0


def test_repeated_generate_step_ids_are_partitioned_by_epoch(tmp_path):
    """run_duet profiles reset step ids; occurrences must not overwrite."""
    path = _profile(tmp_path, "multi_generate", 8, 4, 2.0, -1.0)
    draft_file = path / "duet_profile_draft_test.json"
    target_file = path / "duet_profile_target_rank0_test.json"
    draft = json.loads(draft_file.read_text())
    target = json.loads(target_file.read_text())

    # Append a second generate() epoch with the same ids but a far-away wall
    # clock.  Overwriting by raw step id would cross-pair these two epochs.
    shift_ns = int(1_000_000 * 1e6)
    for source, file in ((draft, draft_file), (target, target_file)):
        second = []
        for event in source:
            copied = dict(event)
            copied["wall_start_ns"] += shift_ns
            copied["wall_end_ns"] += shift_ns
            second.append(copied)
        file.write_text(json.dumps(source + second))

    raw = KB._load_raw(KB.RunSpec(8, 4, path), skip_steps=10)
    row = KB.analyze_run(raw, proxy_transport_ms=1.5)
    assert row["k1_gap"]["n"] == 120
    assert row["k2_gap"]["n"] == 120
    assert abs(row["k1_gap"]["p50_ms"] - 2.0) < 1e-6
    assert abs(row["k2_gap"]["p50_ms"] + 1.0) < 1e-6
