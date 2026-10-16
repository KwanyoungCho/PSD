"""Captured diagnostic preserves proposal outputs and records each replay."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlsys_phase_probe import instrument, drain
import test_p2_executor_parity as parity


@unittest.skipUnless(parity.HAS_FI and torch.cuda.is_available(), 'CUDA/FlashInfer required')
class PhaseProbe(unittest.TestCase):
    def test_native_rng_consumption(self):
        helper = parity.TestExecutorModuleParity()
        ex, cfg, page, vocab = helper._mk('cuda:0', policy='dynamic')
        ctx=page+21; bucket=(ctx+page-1)//page
        ex.prepare_bucket(bucket);helper._fill_inputs(ex,page,ctx)
        torch.cuda.manual_seed_all(87)
        ex.gen.manual_seed(87)
        ex.capture(bucket);ex.model.cache.zero_();ex.replay(bucket)
        reference=ex.view_tok.clone();valid=ex.out_valid.clone()
        instrument(ex)
        torch.cuda.manual_seed_all(87)
        ex.gen.manual_seed(87)
        ex.capture(bucket);ex.model.cache.zero_();ex.replay(bucket)
        self.assertTrue(torch.equal(reference,ex.view_tok))
        self.assertTrue(torch.equal(valid,ex.out_valid))

    def test_captured_outputs_and_no_replay_loss(self):
        helper = parity.TestExecutorModuleParity()
        ex, cfg, page, vocab = helper._mk('cuda:0', policy='dynamic')
        ctx = page + 21
        bucket = (ctx + page - 1) // page
        ex.prepare_bucket(bucket)
        helper._fill_inputs(ex, page, ctx)
        rng = torch.Generator().manual_seed(33)
        ex.parity_noise = [torch.empty(ex.W, vocab).exponential_(generator=rng).cuda()
                           for _ in range(ex.F)]
        ex.capture(bucket)
        ex.model.cache.zero_()
        ex.replay(bucket)
        names = ('view_tok', 'view_par', 'view_sib', 'view_rawq', 'out_valid', 'cell_logits')
        reference = {name: getattr(ex, name).clone() for name in names}
        instrument(ex, limit=4)
        ex.capture(bucket)
        ex.model.cache.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ex.replay(bucket)
        end.record()
        torch.cuda.synchronize()
        for name in names:
            self.assertTrue(torch.equal(getattr(ex, name), reference[name]), name)
        rows = drain(ex)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['forward']), ex.F)
        self.assertEqual(rows[0]['nodes'], ex.out_valid[:ex.R].cpu().tolist())
        self.assertGreater(start.elapsed_time(end), sum(r['forward_ms'] for r in rows[0]['forward']))
        ex.replay(bucket)
        ex.replay(bucket)
        self.assertEqual(len(drain(ex)), 2)
        self.assertEqual(drain(ex), [])


if __name__ == '__main__':
    unittest.main()
