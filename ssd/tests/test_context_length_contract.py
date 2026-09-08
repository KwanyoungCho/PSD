"""CPU regressions for target/draft context-length handling."""

from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ssd.config import Config
from ssd.engine.scheduler import Scheduler
from ssd.engine.sequence import SequenceStatus


def _fake_hf_config(model_path: str):
    return SimpleNamespace(
        max_position_embeddings=(4096 if model_path == "/target" else 2048),
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=4,
        model_type="llama",
        vocab_size=32000,
        rope_theta=10000.0,
    )


class TestDraftRopeExtension(unittest.TestCase):
    def _config(self, **overrides):
        kwargs = dict(
            model="/target",
            draft="/draft",
            speculate=True,
            max_model_len=4096,
        )
        kwargs.update(overrides)
        with patch("ssd.config.os.path.isdir", return_value=True), \
             patch("ssd.config.AutoConfig.from_pretrained",
                   side_effect=_fake_hf_config):
            return Config(**kwargs)

    def test_default_still_clamps_to_native_draft_window(self):
        cfg = self._config()
        self.assertEqual(cfg.max_model_len, 2048)
        self.assertEqual(cfg.draft_hf_config.max_position_embeddings, 2048)

    def test_opt_in_extends_rope_but_keeps_target_as_hard_cap(self):
        cfg = self._config(extend_draft_rope=True, max_model_len=8192)
        self.assertEqual(cfg.max_model_len, 4096)
        self.assertEqual(cfg.draft_hf_config.max_position_embeddings, 4096)

    def test_draft_runner_preserves_extended_window_after_replace(self):
        cfg = self._config(extend_draft_rope=True)
        from ssd.engine.draft_runner import DraftRunner

        with patch("ssd.config.os.path.isdir", return_value=True), \
             patch("ssd.config.AutoConfig.from_pretrained",
                   side_effect=_fake_hf_config):
            draft_cfg = DraftRunner.create_draft_config(cfg)
        self.assertEqual(draft_cfg.max_model_len, 4096)
        self.assertEqual(draft_cfg.hf_config.max_position_embeddings, 4096)
        self.assertEqual(
            draft_cfg.draft_hf_config.max_position_embeddings, 4096)


class TestSchedulerLengthBoundary(unittest.TestCase):
    def test_permanent_draft_lookahead_block_is_identified(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_model_len = 2048
        scheduler.speculate = True
        seq = SimpleNamespace(num_tokens=1824)

        reason = scheduler._model_length_block_reason(seq, 13, 225)

        self.assertIn("draft needs position 2048", reason)

    def test_prompt_without_generation_room_is_rejected_at_admission(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_model_len = 2048
        scheduler.waiting = deque()
        seq = SimpleNamespace(num_tokens=2048)

        with self.assertRaisesRegex(ValueError, "leaves no room"):
            scheduler.add(seq)
        self.assertFalse(scheduler.waiting)

    def test_schedule_finishes_cleanly_instead_of_reprefill_livelock(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_model_len = 2048
        scheduler.max_num_seqs = 1
        scheduler.max_num_batched_tokens = 16384
        scheduler.speculate = True
        scheduler.draft_async = False
        scheduler.K = 8
        scheduler.response_width = 8
        scheduler.block_manager = SimpleNamespace(deallocate=lambda seq: None)
        scheduler.draft_block_manager = SimpleNamespace(
            deallocate=lambda seq: None)
        scheduler.waiting = deque()
        seq = SimpleNamespace(
            seq_id=7, num_tokens=2041, status=SequenceStatus.RUNNING)
        scheduler.running = deque([seq])

        completed, is_prefill = scheduler.schedule()

        self.assertIsNone(is_prefill)
        self.assertEqual(completed, [seq])
        self.assertEqual(seq.status, SequenceStatus.FINISHED)
        self.assertFalse(scheduler.running)


if __name__ == "__main__":
    unittest.main()
