"""Verify direct proposal inputs against the former padded staging contract."""
from collections import defaultdict
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from ssd.engine.helpers import batch_tree_verify as runtime
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy
from ssd.engine.helpers.flat_tree_sampling import FlatProxy


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ProposalPayload(unittest.TestCase):
    @torch.inference_mode()
    def check(self, graph, temperature):
        n, v = 8, 37
        cfg = NS(duet_response_token_width=n, duet_phase1_k=6, duet_phase2_k=2,
                 duet_tree_wire_nodes=n, duet_tree_c_tensor=3,
                 hf_config=NS(vocab_size=v, torch_dtype=torch.bfloat16),
                 duet_proxy_wire_N=15, duet_proxy_top_k=13,
                 stochastic_only=temperature > 0)
        r = NS(config=cfg, device='cuda', block_size=256, async_pg=None, draft_rank=1)
        delivered = []
        logits = None
        def call(name, *args):
            if name == 'commit_tree_kv': return
            self.assertEqual(name, 'run_batched_tree')
            r._duet_proxy_fn(logits, len(args[0]))
            return logits
        r.call = call
        verifier = NS(target_model_runner=r, sampler_x=None, async_fan_out=1,
                      metrics=defaultdict(list),
                      _send_proxy_wire=lambda *a: delivered.append(tuple(x.clone() for x in a[-2:])))
        prepare = {c: c.prepare for c in (FlatProxy, BatchedTreeProxy)}
        observed = []
        def padded(self, parents, siblings, tokens, q, *args):
            # Reference is the exact pre-change staging shape and values.
            b = len(parents)
            t = torch.zeros(b, n, device=tokens.device, dtype=tokens.dtype)
            z = torch.zeros(b, n, v, device=q.device, dtype=q.dtype)
            t[:, :tokens.shape[1]].copy_(tokens)
            z[:, :q.shape[1]].copy_(q)
            observed.append((type(self).__name__, tokens.shape[1], q.shape[1]))
            return prepare[type(self)](self, parents, siblings, t, z, *args)
        with patch.dict('os.environ', {'SSD_BATCH_TREE_FIXED_VERIFY':'1',
                        'SSD_BATCH_TREE_ACCEPT_GRAPH':str(int(graph))}):
          # Large -> short -> empty -> large reuses buffers across shrinkage.
          for b, width, qwidth in ((4,8,8), (4,4,4), (3,4,4), (4,0,0), (4,4,2), (4,8,8)):
            torch.manual_seed(409)
            logits = torch.randn(b, n+1, v, device='cuda', dtype=torch.bfloat16)
            spec = torch.arange(width+1, device='cuda').expand(b,-1)
            q = torch.randn(b, qwidth, v, device='cuda', dtype=torch.float32)
            valid = [width if i % 2 == 0 else min(2,width) for i in range(b)]
            wire = torch.zeros(b, 3+4*n, dtype=torch.int64)
            for i, length in enumerate(valid):
                if length:
                    wire[i, :3] = torch.tensor([length, 1, 2])
                    wire[i, 3:3+length] = spec[i, 1:1+length].cpu()
                    wire[i, 3+n:3+n+length] = torch.tensor([-1,-1,0,0,1,2,3,4][:length])
                    wire[i, 3+2*n:3+2*n+length] = torch.tensor([0,1,0,1,0,0,0,0][:length])
            result = NS(speculations=spec, logits_q=q, tree_ints=wire.cuda(),
                        phase_source=torch.ones(b, device='cuda', dtype=torch.int64),
                        cache_hits=torch.ones(b, device='cuda', dtype=torch.bool), step_id=1)
            def run():
                seqs = [NS(verify_valid_k=k, num_tokens=30+width+1, num_cached_tokens=0,
                           block_table=[2+i], temperature=temperature, draft_temperature=None)
                        for i,k in enumerate(valid)]
                out = runtime.verify(verifier, seqs, result)
                return out, [(s.num_cached_tokens, s.tree_terminal_node) for s in seqs]
            run()  # capture every new shape before fixing the RNG
            torch.cuda.manual_seed(187)
            actual = run()
            actual_wire = tuple(x.clone() for x in delivered[-1])
            actual_rng = torch.cuda.get_rng_state()
            with patch.object(FlatProxy, 'prepare', padded), patch.object(BatchedTreeProxy, 'prepare', padded):
                torch.cuda.manual_seed(187)
                expected = run()
            self.assertEqual(actual, expected)
            for x,y in zip(actual_wire, delivered[-1]): self.assertTrue(torch.equal(x,y))
            self.assertTrue(torch.equal(actual_rng, torch.cuda.get_rng_state()))
        self.assertTrue(any(width==4 for _,width,_ in observed))
        self.assertTrue(any(width==8 for _,width,_ in observed))

    def test_captured_stochastic(self): self.check(True, .7)
    def test_captured_greedy(self): self.check(True, 0.)
    def test_eager_stochastic(self): self.check(False, .7)
    def test_eager_greedy(self): self.check(False, 0.)


if __name__ == '__main__': unittest.main()
