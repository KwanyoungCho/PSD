"""Unit tests for the E0 calibration trace gate (P0 — docs/duet/internal/17 §2).

CPU-only. The module reads its env gate at import time, so each test
(re)imports it under a controlled env via importlib.

Run from project root (/home/chokwans99/PSD/ssd):
    python -m unittest tests.test_e0_trace
"""
import importlib
import json
import os
import sys
import tempfile
import unittest

import torch


def _fresh_module(env):
    saved = {k: os.environ.get(k) for k in
             ("SSD_DUET_E0_TRACE", "SSD_DUET_E0_DIR", "SSD_DUET_E0_SUBSAMPLE")}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("ssd.engine.helpers.e0_trace", None)
    mod = importlib.import_module("ssd.engine.helpers.e0_trace")
    # Restore env (module already captured its values).
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return mod


class TestGateDefault(unittest.TestCase):
    def test_default_off(self):
        mod = _fresh_module({})
        self.assertFalse(mod.E0_TRACE)


class TestRecording(unittest.TestCase):
    def _run(self, extra_env, n_records):
        tmp = tempfile.mkdtemp(prefix="e0_test_")
        mod = _fresh_module({"SSD_DUET_E0_TRACE": "1",
                             "SSD_DUET_E0_DIR": tmp, **extra_env})
        self.assertTrue(mod.E0_TRACE)
        for i in range(n_records):
            mod.record_draft_request(
                step_id=i,
                cache_keys=torch.tensor([[7, i, 100 + i]]),
                temps=torch.tensor([7000]),
                num_tokens=torch.tensor([2040 + i]))
        mod.close_all()
        files = [f for f in os.listdir(tmp) if f.startswith("e0_draft")]
        self.assertEqual(len(files), 1)
        with open(os.path.join(tmp, files[0])) as f:
            recs = [json.loads(line) for line in f]
        return recs

    def test_schema_and_summary(self):
        recs = self._run({}, 3)
        body = [r for r in recs if r["kind"] == "request"]
        summ = [r for r in recs if r["kind"] == "summary"]
        self.assertEqual(len(body), 3)
        self.assertEqual(len(summ), 1)
        self.assertEqual(summ[0]["drops"], 0)          # E0 위생: drop=0
        self.assertEqual(summ[0]["written"], 3)
        r = body[0]
        self.assertEqual(r["step_id"], 0)
        self.assertEqual(r["cache_keys"], [[7, 0, 100]])
        self.assertEqual(r["temps"], [7000])
        self.assertEqual(r["num_tokens"], [2040])

    def test_subsample(self):
        recs = self._run({"SSD_DUET_E0_SUBSAMPLE": "2"}, 4)
        body = [r for r in recs if r["kind"] == "request"]
        self.assertEqual(len(body), 2)                 # 매 2번째만 기록

    def test_target_wire_schema(self):
        tmp = tempfile.mkdtemp(prefix="e0_test_")
        mod = _fresh_module({"SSD_DUET_E0_TRACE": "1", "SSD_DUET_E0_DIR": tmp})
        B, K, V, top_k, N = 1, 4, 64, 6, 9
        exit_logits = torch.randn(B * (K + 1), V)
        logits_q = torch.randn(B, K, V)
        draft_tokens = torch.randint(0, V, (B, K))
        P_iv = torch.rand(B, K + 1, top_k)
        top_idx = torch.topk(P_iv.flatten(1), N, dim=-1).indices
        chosen_pos = top_idx // top_k
        chosen_tok = torch.randint(0, V, (B, N))
        h = torch.rand(B, K + 1)
        mod.record_target_wire(None, exit_logits, logits_q, draft_tokens,
                               B, K, None, None, P_iv, top_idx,
                               chosen_pos, chosen_tok, h)
        mod.close_all()
        files = [f for f in os.listdir(tmp) if f.startswith("e0_target")]
        with open(os.path.join(tmp, files[0])) as f:
            recs = [json.loads(line) for line in f]
        wire = [r for r in recs if r["kind"] == "wire"][0]
        # 충분통계 필드 전수 + shape 확인
        self.assertEqual(wire["K"], K)
        self.assertEqual(len(wire["piv"][0]), N)
        self.assertEqual(len(wire["h"][0]), K + 1)
        self.assertEqual(len(wire["y_logit_E"][0]), K)
        self.assertEqual(len(wire["lse1_E"][0]), K + 1)
        self.assertEqual(len(wire["exit_top_ids"][0]), K + 1)
        self.assertEqual(len(wire["exit_top_ids"][0][0]), min(32, V))
        self.assertEqual(len(wire["cand_logit_E"][0]), N)
        # P_iv 값 정합: 기록된 piv == P_iv.flatten.gather(top_idx)
        expect = P_iv.flatten(1).gather(1, top_idx)[0].tolist()
        for a, b in zip(wire["piv"][0], expect):
            self.assertAlmostEqual(a, b, places=5)
        # cand_logit_E 정합: exit_view[0, pos, tok]
        ev = exit_logits.view(B, K + 1, V)
        for j in range(N):
            self.assertAlmostEqual(
                wire["cand_logit_E"][0][j],
                float(ev[0, chosen_pos[0, j], chosen_tok[0, j]]), places=4)


if __name__ == "__main__":
    unittest.main()
