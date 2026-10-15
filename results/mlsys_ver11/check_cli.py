"""Exercise public CLI -> canonical Config without loading model weights."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ssd/bench'), str(ROOT / 'ssd')]
from bench import parse_arguments, create_llm_kwargs
from ssd.config import Config
from summarize_ssd_run import parse_run_log

target = '/home/chokwans99/models_mlsys/layerskip-llama2-7b'
draft = '/home/chokwans99/models_mlsys/AMD-Llama-135m'
argv = ['bench.py', '--model_path', target, '--draft_path', draft,
        '--gpus', '2', '--spec', '--async', '--duet', '--k', '12',
        '--duet_phase1_k', '8', '--duet_phase2_k', '4',
        '--duet_exit_layer', '21', '--duet_draft_fan_out', '3',
        '--duet_p1_tree_policy', 'on', '--duet_p2_tree_policy', 'on',
        '--duet_p1_forward_width', '39', '--duet_p2_forward_width', '15',
        '--duet_p2_root_count', '10', '--duet_p1_roots_per_position', '3',
        '--duet_p1_tree_verify_nodes', '12', '--duet_p2_tree_verify_nodes', '8',
        '--duet_tree_c_tensor', '3', '--temp', '.7', '--max_model_len', '2048']
stdout = io.StringIO()
with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(stdout):
    args = parse_arguments()
assert '[deprecated]' not in stdout.getvalue()
kwargs = create_llm_kwargs(args, draft)
with contextlib.redirect_stdout(stdout):
    cfg = Config(target, **kwargs)
print(stdout.getvalue(), end='')
assert cfg.duet_p1_tree_max_nodes == cfg.duet_p1_tree_verify_nodes == 12
assert cfg.duet_p2_tree_max_nodes == cfg.duet_p2_tree_verify_nodes == 8
assert (cfg.duet_proxy_wire_N, cfg.duet_proxy_top_k) == (51, 51)
assert (cfg.duet_p2_seed_count, cfg.duet_proxy_total_budget) == (10, 15)
assert cfg.stochastic_only and not cfg.greedy_only
alias_argv = [x.replace('--duet_phase1_k', '--duet_k1').replace('--duet_phase2_k', '--duet_k2')
              for x in argv]
with patch.object(sys, 'argv', alias_argv), contextlib.redirect_stdout(io.StringIO()):
    alias_args = parse_arguments()
assert create_llm_kwargs(alias_args, draft) == kwargs
with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp)/'run.log'
    log.write_text(stdout.getvalue())
    row = parse_run_log(Path(tmp))
    for key, value in dict(duet_exit_layer=21, duet_phase1_k=8, duet_phase2_k=4,
            duet_p1_forward_width=39, duet_p2_forward_width=15, duet_p2_root_count=10,
            duet_proxy_top_k=51, duet_proxy_wire_N=51,
            duet_p1_tree_verify_nodes=12, duet_p2_tree_verify_nodes=8).items():
        assert row[key] == value, (key, row[key], value)
    log.write_text('[Config] DUET-SSD enabled: exit_layer=56, proxy_top_k=28, K1=8, K2=4')
    old_row = parse_run_log(Path(tmp))
    assert (old_row['duet_exit_layer'], old_row['duet_proxy_top_k'],
            old_row['duet_phase1_k'], old_row['duet_phase2_k']) == (56,28,8,4)
rejected = []
for flag in ('--duet_p1_tree_forward_scale', '--duet_proxy_top_k',
             '--duet_p2_budget', '--duet_p1_tree_max_nodes',
             '--duet_tree_proxy_threshold'):
    with patch.object(sys, 'argv', ['bench.py', flag, '1']), contextlib.redirect_stderr(io.StringIO()):
        try:
            parse_arguments()
        except SystemExit as exc:
            assert exc.code == 2
            rejected.append(flag)
        else:
            raise AssertionError(f'obsolete flag accepted: {flag}')
result = dict(passed=True, weights_loaded=False, target=target, draft=draft,
              canonical_names_not_deprecated=True, aliases_equivalent=True,
              old_and_new_log_parser=True,
              command=argv, kwargs=kwargs, G_equals_M=[12, 8],
              local_top_L=51, transmitted_candidates=51, rejected=rejected)
(Path(__file__).parent/'CLI_CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
print('CLI -> Config: PASS; obsolete flag rejection: PASS; no model weights loaded')
