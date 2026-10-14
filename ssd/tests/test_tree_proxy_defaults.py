"""Default selection must respect TP, overrides and non-proxy forwards."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ssd.config import Config
from ssd.engine.helpers.batch_tree_forward import proxy_stream_enabled


class ProxyDefaults(unittest.TestCase):
    def config(self, **overrides):
        values = dict(max_num_seqs=8, duet_tree_enabled=True, draft_async=True,
                      num_gpus=3, duet_exit_topm_gather=False,
                      duet_proxy_on_draft=False, target_quant_lm_head=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_replica_requires_explicit_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Config.duet_exit_replica.fget(self.config()))
            for settings in (dict(max_num_seqs=1), dict(num_gpus=2),
                             dict(draft_async=False), dict(duet_tree_enabled=False),
                             dict(duet_exit_topm_gather=True),
                             dict(duet_proxy_on_draft=True), dict(target_quant_lm_head=True)):
                with self.subTest(settings=settings):
                    self.assertFalse(Config.duet_exit_replica.fget(self.config(**settings)))

    def test_replica_explicit_override_is_preserved(self):
        for override in ('0', '1'):
            with patch.dict(os.environ, {'SSD_DUET_EXIT_REPLICA': override}, clear=True):
                self.assertEqual(Config.duet_exit_replica.fget(self.config()), override == '1')

    def test_stream_default_requires_replica_and_tp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(proxy_stream_enabled(SimpleNamespace(), 1))
            self.assertFalse(proxy_stream_enabled(SimpleNamespace(), 2))
            self.assertFalse(proxy_stream_enabled(SimpleNamespace(duet_exit_replica=True), 1))
            self.assertTrue(proxy_stream_enabled(SimpleNamespace(duet_exit_replica=True), 2))

    def test_stream_override_is_preserved(self):
        for override in ('0', '1'):
            with patch.dict(os.environ, {'SSD_BATCH_TREE_PROXY_STREAM': override}, clear=True):
                self.assertEqual(proxy_stream_enabled(SimpleNamespace(), 1), override == '1')


if __name__ == '__main__':
    unittest.main()
