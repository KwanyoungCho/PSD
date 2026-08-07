import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1]
          / "tools/duet_calibration/analyze_thresholds.py")
SPEC = importlib.util.spec_from_file_location("threshold_calibration", SCRIPT)
CAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CAL
SPEC.loader.exec_module(CAL)


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(x) + "\n" for x in records))


def test_new_e0_selector_trace_is_self_contained(tmp_path):
    e0 = tmp_path / "e0"
    e0.mkdir()
    _write_jsonl(e0 / "e0_draft_1.jsonl", [
        {"kind": "request", "step_id": 1,
         "cache_keys": [[7, 0, 99]]},
        {"kind": "selector", "step_id": 1,
         "proxy_fan_out": [[1, 1]],
         "proxy_forked": [[101, 102]],
         "proxy_piv": [[0.001, 0.2]]},
        # The next request is the actual cache lookup outcome for the tree
        # built by selector step 1.  Batch order need not be used: seq id is.
        {"kind": "request", "step_id": 2,
         "cache_keys": [[7, 1, 102]]},
        {"kind": "summary", "written": 3, "drops": 0},
    ])

    slots, notes = CAL.load_proxy_slots([e0])
    assert len(slots) == 2
    assert [(x["score"], x["hit"]) for x in slots] == [
        (0.001, 0), (0.2, 1)]
    assert any("paired proxy outcomes=1" in x for x in notes)


def test_expansion_label_keeps_accepted_leaf_out_of_loss(tmp_path):
    conf = tmp_path / "confidence.jsonl"
    _write_jsonl(conf, [{
        "policy": "eagle",
        "nodes": [
            {"node": 0, "parent": -1, "depth": 1, "q": 0.02,
             "attempted": True, "accepted": True, "on_path": True},
            {"node": 1, "parent": 0, "depth": 2, "q": 0.01,
             "attempted": True, "accepted": True, "on_path": True},
            {"node": 2, "parent": 1, "depth": 4, "q": 0.001,
             "attempted": False, "accepted": False, "on_path": False},
        ],
    }])
    records, nodes = CAL.load_confidence_nodes([conf])
    assert records == 1
    useful = {x["node"]: x["expansion_useful"] for x in nodes}
    assert useful == {0: True, 1: False, 2: False}

    rows = CAL.confidence_table(nodes, depth_cap=4, thresholds=(0.015,))
    # node 1 is itself accepted but has no accepted child below it in this
    # synthetic topology, so it would normally be a leaf-only non-loss.  Here
    # node 1 *does* parent accepted node 2 only if node 2 is on_path; it is not.
    assert rows[0]["expansion_useful"] == 0
    assert rows[0]["candidate_n"] == 2  # final-depth node excluded


def test_risk_profiles_select_safe_and_balanced_knees():
    proxy_rows = [
        {"threshold": 0.003, "n": 1000,
         "hit_contribution": 0.008, "hit_rate_upper95": 0.0018},
        {"threshold": 0.01, "n": 4000,
         "hit_contribution": 0.04, "hit_rate_upper95": 0.0028},
        {"threshold": 0.03, "n": 6000,
         "hit_contribution": 0.15, "hit_rate_upper95": 0.006},
    ]
    conf_rows = [
        {"threshold": 0.01, "n": 2000,
         "expansion_use_contribution": 0.005,
         "expansion_use_upper95": 0.003},
        {"threshold": 0.03, "n": 4000,
         "expansion_use_contribution": 0.013,
         "expansion_use_upper95": 0.0046},
        {"threshold": 0.05, "n": 5000,
         "expansion_use_contribution": 0.025,
         "expansion_use_upper95": 0.006},
    ]
    safe = CAL.RISK_PROFILES["safe"]
    balanced = CAL.RISK_PROFILES["balanced"]
    assert CAL.choose_proxy(proxy_rows, safe)["threshold"] == 0.003
    assert CAL.choose_confidence(conf_rows, safe)["threshold"] == 0.01
    assert CAL.choose_proxy(proxy_rows, balanced)["threshold"] == 0.01
    assert CAL.choose_confidence(conf_rows, balanced)["threshold"] == 0.03
