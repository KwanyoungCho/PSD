import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from transformers import LlamaConfig
from ssd.config import Config
from ssd.engine.draft_runner import DraftRunner
from ssd.utils.misc import validate_speculative_vocab


class TestModelPair(unittest.TestCase):
    def test_non_duet_response_width_is_chain_k(self):
        with patch("ssd.config.os.path.isdir", return_value=True), \
             patch("ssd.config.AutoConfig.from_pretrained", return_value=LlamaConfig()):
            cfg = Config(model="/target", draft="/draft", speculate=True, speculate_k=4)
        self.assertFalse(cfg.duet_tree_enabled)
        self.assertEqual(cfg.duet_response_token_width, 4)

    def test_equal_vocab_size_does_not_imply_compatible_ids(self):
        target = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
        draft = SimpleNamespace(get_vocab=lambda: {"b": 0, "a": 1})
        with self.assertRaisesRegex(ValueError, "token-to-ID"):
            validate_speculative_vocab(target, draft, 2, 2)
        validate_speculative_vocab(target, target, 2, 2)

    def test_vocab_dimension_and_id_range(self):
        tok = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 3})
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_speculative_vocab(tok, tok, 4, 5)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_speculative_vocab(tok, tok, 3, 3)

    def test_float32_draft_checkpoint_uses_target_runtime_dtype(self):
        def config(path):
            return LlamaConfig(num_hidden_layers=32,
                               torch_dtype=torch.float16 if path == "/target" else torch.float32)
        with patch("ssd.config.os.path.isdir", return_value=True), \
             patch("ssd.config.AutoConfig.from_pretrained", side_effect=config):
            main = Config(model="/target", draft="/draft", speculate=True)
            draft = DraftRunner.create_draft_config(main)
        self.assertEqual(draft.hf_config.torch_dtype, torch.float16)
        self.assertEqual(draft.draft_hf_config.torch_dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
