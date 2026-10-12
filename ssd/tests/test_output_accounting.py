import os
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from ssd import SamplingParams
from ssd.engine.sequence import Sequence, SequenceStatus
from ssd.engine.scheduler import Scheduler
from ssd.engine.step import SpecDecodeStep
from ssd.engine.block_manager import BlockManager
from ssd.engine.helpers.speculate_types import VerifyResult


class TestOutputAccounting(unittest.TestCase):
    def test_completed_block_hash_points_to_its_own_physical_page(self):
        old_size = getattr(Sequence, "block_size", None)
        Sequence.block_size = 4
        try:
            seq = Sequence(list(range(9)))
            manager = BlockManager(4, 4)
            pages = manager._allocate_n_blocks(3)
            table = [p.block_id for p in pages]
            scheduler = Scheduler.__new__(Scheduler)
            h = -1
            for index in (0, 1):
                scheduler._finalize_block(manager, seq, table, index)
                tokens = list(range(index*4, (index+1)*4))
                h = manager.compute_hash(tokens, h)
                self.assertEqual(manager.blocks[table[index]].hash, h)
                self.assertEqual(manager.hash_to_block_id[h], table[index])
            self.assertEqual(manager.blocks[table[2]].hash, -1)
        finally:
            if old_size is None:
                del Sequence.block_size
            else:
                Sequence.block_size = old_size

    def test_preemption_preserves_output_and_remaining_budget(self):
        seq = Sequence([10, 11], SamplingParams(max_new_tokens=4))
        seq.append_token(20)
        seq.append_token(21)
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.waiting = deque()
        scheduler.speculate = False
        scheduler.block_manager = SimpleNamespace(deallocate=lambda s: None)
        scheduler.eos, scheduler.max_model_len = 99, 2048
        scheduler.preempt(seq)
        self.assertEqual(seq.num_prompt_tokens, 4)  # re-prefill prefix
        self.assertEqual(seq.completion_token_ids, [20, 21])
        self.assertEqual(seq.num_completion_tokens, 2)
        suffix, finished = scheduler._handle_eos_and_max_new_tokens(seq, [22, 23, 24])
        self.assertEqual(suffix, [22, 23])
        self.assertTrue(finished)
        restored = Sequence.__new__(Sequence)
        restored.__setstate__(seq.__getstate__())
        self.assertEqual(restored.completion_token_ids, [20, 21])
        self.assertEqual(seq.clone_spec().completion_token_ids, [20, 21])

    def test_decode_counts_committed_tokens_after_cap_or_eos(self):
        for cap, eos, expected in [(2, 99, 2), (8, 20, 1)]:
            seq = Sequence([10, 11], SamplingParams(max_new_tokens=cap))
            seq.status = SequenceStatus.RUNNING
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.eos, scheduler.max_model_len = eos, 2048

            def commit(seqs, suffixes, recovery, **kwargs):
                for s, suffix in zip(seqs, suffixes):
                    clipped, _ = scheduler._handle_eos_and_max_new_tokens(s, suffix)
                    for tok in clipped:
                        s.append_token(tok)
            scheduler.postprocess_speculate = commit
            proposal = SimpleNamespace(speculations=torch.tensor([[20, 21, 22, 23]]),
                                       step_id=0, profile_cache_status=None)
            speculator = SimpleNamespace(speculate=lambda *a: proposal)
            metrics = {"phase_events": []}
            def verify(*args, **kwargs):
                metrics['phase_events'].append(dict(accepted_len=4))
                return VerifyResult([[20, 21, 22, 23]], [24], None)
            verifier = SimpleNamespace(verify=verify, metrics=metrics)
            step = SpecDecodeStep(scheduler, speculator, verifier, False, None, True)
            with patch("ssd.engine.step.decode_tokens", side_effect=lambda ids,t: str(ids)):
                count = step.decode([seq])
            self.assertEqual(count, expected)
            self.assertEqual(count, len(seq.completion_token_ids))
            event = metrics['phase_events'][0]
            self.assertEqual(event['emitted_len'], expected)
            self.assertTrue(event['clipped'])
            self.assertEqual(event['output_cap_reached'], cap==2)
            self.assertEqual(event['seq_id'], seq.seq_id)


if __name__ == "__main__":
    unittest.main()
