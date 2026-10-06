import os
import unittest

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from ssd.utils.verify import verify
from ssd.utils.verify_fast import verify_stochastic


class TestStochasticFastVerify(unittest.TestCase):
    def test_same_coins_match_general_verifier(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for k in (1, 2, 4):
                    for seed in range(8):
                        torch.manual_seed(seed)
                        p = torch.randn(8, k+1, 127, dtype=dtype, device=device)
                        q = torch.randn(8, k, 127, dtype=dtype, device=device)
                        temps = torch.linspace(.2, 1.4, 8, device=device)
                        tq = temps.flip(0)
                        vk = torch.arange(8, device=device) % k + 1
                        proposals = torch.multinomial(
                            (q / tq[:, None, None]).float().softmax(-1).flatten(0, 1), 1).view(8, k)
                        spec = torch.cat((torch.zeros(8, 1, dtype=torch.long, device=device), proposals), 1)
                        torch.manual_seed(seed+100)
                        ref = verify(p, q, spec, temps, tq, jit_speculate=True, valid_k=vk)
                        torch.manual_seed(seed+100)
                        actual = verify_stochastic(p, q, spec, temps, tq, valid_k=vk)
                        self.assertEqual(ref, actual, (device, dtype, k, seed))

    def test_empirical_first_token_matches_target(self):
        # Large vectorized sample: independent proposals q and accept coins.
        # This checks distribution preservation, beyond mirrored code parity.
        n = 40000
        torch.manual_seed(2026)
        p0 = torch.tensor([.02, .18, .35, .45])
        q0 = torch.tensor([.60, .15, .20, .05])
        proposals = torch.multinomial(q0, n, replacement=True)
        spec = torch.stack((torch.zeros_like(proposals), proposals), 1)
        p = p0.log().view(1, 1, 4).expand(n, 2, 4)
        q = q0.log().view(1, 1, 4).expand(n, 1, 4)
        suffix, recovery = verify_stochastic(p, q, spec, torch.ones(n), torch.ones(n))
        first = torch.tensor([s[1] if len(s) > 1 else r for s, r in zip(suffix, recovery)])
        freq = first.bincount(minlength=4) / n
        self.assertTrue(bool(((freq-p0).abs() < .012).all()), freq)


if __name__ == "__main__":
    unittest.main()
