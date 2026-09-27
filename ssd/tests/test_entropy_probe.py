"""Mathematical and persistence checks for the diagnostic entropy collector."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "entropy_probe", Path(__file__).parents[1] / "ssd/engine/helpers/entropy_probe.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestEntropyProbe(unittest.TestCase):
    def test_known_entropies_variable_chain_and_final_flush(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "probe.json"
            p = module.EntropyProbe([56, 79], dest, 56, flush_steps=2)
            state = torch.random.get_rng_state().clone()
            # Completely accepted chain: no rejection weight, bonus mass 1.
            target = torch.tensor([[1., 0., 0., 0.]]).expand(3, -1)
            uniform = torch.full_like(target, .25)
            p.observe(torch.stack([uniform, target]), uniform[:2], target,
                      torch.zeros(2, dtype=torch.long), seq_id=7)
            # A shorter chain with rejection probability 3/4.
            target2 = uniform[:2]
            peaked = target[:2]
            p.observe(torch.stack([peaked, target2]), peaked[:1], target2,
                      torch.zeros(1, dtype=torch.long), seq_id=8)
            # Partial final chunk must not disappear.
            p.observe(torch.stack([target2, target2]), peaked[:1], target2,
                      torch.zeros(1, dtype=torch.long), seq_id=8)
            p.dump()
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            meta = json.loads(dest.read_text())
            self.assertEqual((meta["n_steps"], meta["n_rows"], len(meta["chunks"])), (3, 7, 2))
            chunks = [np.load(dest.parent / c["file"]) for c in meta["chunks"]]
            he = np.concatenate([z["entropy_proxy"] for z in chunks])
            ht = np.concatenate([z["entropy_target"] for z in chunks])
            delta = he - ht[:, None]
            np.testing.assert_allclose(delta[:3, 0], math.log(4), atol=1e-6)
            np.testing.assert_allclose(delta[3:5, 0], -math.log(4), atol=1e-6)
            np.testing.assert_array_equal(delta[:, 1], 0)
            np.testing.assert_array_equal(delta[5:], 0)
            hazard = np.concatenate([z["hazard"] for z in chunks])
            np.testing.assert_allclose(hazard, [0, 0, 1, .75, .25, .75, .25])
            for z in chunks:
                z.close()


if __name__ == "__main__":
    unittest.main()
