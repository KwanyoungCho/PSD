"""CPU contracts for symmetric API/CLI tree defaults and explicit chain arms."""
import argparse
import ast
import contextlib
import dataclasses
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from test_blackwell_review import make_config
from ssd.config import Config


def cli(*args):
    # Execute the actual parser without importing wandb or starting an engine.
    source = Path(__file__).parents[1] / 'bench/bench.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'parse_arguments')
    namespace = {'argparse': argparse, 'sys': sys}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    with patch.object(sys, 'argv', ['bench.py', *args]), contextlib.redirect_stdout(io.StringIO()):
        return namespace['parse_arguments']()


def policies(cfg):
    return cfg.duet_p1_tree_policy, cfg.duet_p2_tree_policy, cfg.duet_tree_policy


class TreePolicyDefaults(unittest.TestCase):
    def test_api_field_defaults_enable_both_phases(self):
        fields = {f.name: f.default for f in dataclasses.fields(Config)}
        self.assertEqual(fields['duet_p1_tree_policy'], 'on')
        self.assertEqual(fields['duet_p2_tree_policy'], 'on')

    def test_api_default_matches_explicit_on_and_draft_reconstruction(self):
        implicit, implicit_draft = make_config()
        explicit, explicit_draft = make_config(duet_p1_tree_policy='on', duet_p2_tree_policy='on')
        ignored = {'hf_config', 'draft_hf_config', 'target_hf_config'}
        for left, right in ((implicit, explicit), (implicit_draft, explicit_draft)):
            self.assertEqual(policies(left), ('on', 'on', 'dynamic'))
            self.assertEqual({k: v for k, v in vars(left).items() if k not in ignored},
                             {k: v for k, v in vars(right).items() if k not in ignored})
        self.assertEqual(implicit.duet_p1_tree_allocation_policy, 'dynamic')
        self.assertEqual(implicit.duet_tree_expansion, 'last')

    def test_explicit_chain_stays_off_in_both_runners(self):
        cfg, draft = make_config(duet_p1_tree_policy='off', duet_p2_tree_policy='off')
        for value in (cfg, draft):
            self.assertEqual(policies(value), ('off', 'off', 'off'))

    def test_phase_only_overrides_stay_independent(self):
        for p1, p2 in (('off', 'on'), ('on', 'off')):
            with self.subTest(p1=p1, p2=p2):
                cfg, draft = make_config(duet_p1_tree_policy=p1, duet_p2_tree_policy=p2)
                for value in (cfg, draft):
                    self.assertEqual(policies(value), (p1, p2, 'dynamic'))

    def test_cli_defaults_match_api(self):
        args = cli('--duet', '--duet_phase1_k', '9', '--duet_phase2_k', '4')
        self.assertEqual((args.duet_p1_tree_policy, args.duet_p2_tree_policy), ('on', 'on'))
        self.assertEqual(args.k, 13)

    def test_cli_explicit_chain_and_single_phase_override(self):
        for p1, p2 in (('off', 'off'), ('off', 'on'), ('on', 'off')):
            with self.subTest(p1=p1, p2=p2):
                args = cli('--duet_p1_tree_policy', p1, '--duet_p2_tree_policy', p2)
                self.assertEqual((args.duet_p1_tree_policy, args.duet_p2_tree_policy), (p1, p2))

    def test_tree_flags_do_not_enable_duet_by_themselves(self):
        args = cli()
        self.assertFalse(args.duet)
        cfg, draft = make_config(duet_enabled=False)
        self.assertFalse(cfg.duet_enabled)
        self.assertFalse(draft.duet_enabled)


if __name__ == '__main__':
    unittest.main()
