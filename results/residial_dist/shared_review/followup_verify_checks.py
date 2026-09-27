"""CPU checks of production chain verification and cache response semantics.

Functions/classes are loaded verbatim through AST to avoid starting GPU model
runners. The cache test stubs only JIT model execution and uses real lookup,
token/logit copying, valid_k selection, and glue-input assembly. These tests
do not establish numerical equality of distributed transformer/KV kernels.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
HASHES = {}


def load_node(relative, name, namespace, parent=None):
    path = ROOT / relative
    HASHES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(path.read_text())
    body = tree.body
    if parent:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == parent).body
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def check_hist(observed, expected, n):
    error = np.abs(observed-expected)
    tolerance = 6*np.sqrt(expected*(1-expected)/n)+3/n
    assert np.all(error <= tolerance), (observed, expected, tolerance)
    return float(error.max())


def monte_carlo(verify, sampler, p_values, q_values, seed, n=200000):
    torch.manual_seed(seed)
    p, q = torch.tensor(p_values), torch.tensor(q_values)
    v = len(p)
    bonus = torch.arange(1, v+1, dtype=torch.float32)
    bonus /= bonus.sum()
    temperatures = torch.ones(n)
    q_logits = q.log().repeat(n, 1)
    y = sampler(q_logits, temperatures)
    # Sentinel root is outside the vocabulary: success proves that the root
    # is carried through as an already committed token, not ratio-verified.
    proposals = torch.stack([torch.full_like(y, 123456), y], -1)
    lp = torch.stack([p.log(), bonus.log()]).repeat(n, 1, 1)
    lq = q_logits[:, None, :]
    rng = torch.get_rng_state()
    outputs = []
    for hit in (False, True):
        torch.set_rng_state(rng)
        outputs.append(verify(lp, lq, proposals, temperatures, temperatures,
                              cache_hits=torch.full((n,), hit), jit_speculate=True))
    assert outputs[0] == outputs[1], 'hit/miss changes verification with jit enabled'
    suffixes, recovery = outputs[0]
    accepted = np.fromiter((len(s) == 2 for s in suffixes), dtype=bool, count=n)
    assert all(s[0] == 123456 for s in suffixes)
    rec = np.asarray(recovery)
    yy = y.numpy()
    first = np.where(accepted, yy, rec)
    hist = lambda x: np.bincount(x, minlength=v)/len(x)
    result = dict(n=n, proposal_hist=hist(yy).tolist(), first_output_hist=hist(first).tolist(),
                  reject_count=int((~accepted).sum()), hit_miss_identical=True,
                  committed_root_not_reverified=True,
                  proposal_max_abs_error=check_hist(hist(yy), q.numpy(), n),
                  output_max_abs_error=check_hist(hist(first), p.numpy(), n))
    if (~accepted).any():
        r = np.maximum(p.numpy()-q.numpy(), 0)
        r /= r.sum()
        result['correction_hist'] = hist(rec[~accepted]).tolist()
        result['correction_expected'] = r.tolist()
        result['correction_max_abs_error'] = check_hist(hist(rec[~accepted]), r, int((~accepted).sum()))
    result['bonus_max_abs_error'] = check_hist(hist(rec[accepted]), bonus.numpy(), int(accepted.sum()))
    return result


def cache_checks():
    ns = dict(torch=torch, os=os, SPLIT_K1K2_MODE=True, DUET_JIT_SHORT=True,
              DUET_JIT_SUBSET=False, TRACE_SPLIT_K1K2=False)
    ns['make_glue_decode_input_ids'] = load_node(
        'ssd/ssd/utils/async_helpers/async_spec_helpers.py', 'make_glue_decode_input_ids', ns)
    filt = load_node('ssd/ssd/engine/helpers/p2_tree.py', 'filter_unservable_tree_matches', ns)
    lookup = load_node('ssd/ssd/engine/draft_runner.py', 'hit_cache_and_respond', ns, 'DraftRunner')
    module = ModuleType('ssd.engine.helpers.p2_tree')
    module.filter_unservable_tree_matches = filt
    calls = []

    def jit(keys, num_tokens, out_logits, out_tokens, temperatures, blocks, acts=None):
        calls.append(keys.clone())
        out_tokens[:, :2] = torch.tensor([2, 1])
        out_logits[:, :2] = torch.tensor([[10., 20., 30.], [30., 20., 10.]])

    def runner(empty=False):
        return SimpleNamespace(
            device=torch.device('cpu'), hf_config=SimpleNamespace(vocab_size=3, hidden_size=2, torch_dtype=torch.float32),
            config=SimpleNamespace(jit_speculate=True, duet_enabled=True, duet_phase1_k=3,
                                   duet_phase2_k=2, use_eagle=False, verbose=False, duet_response_token_width=3),
            tree_cache_keys=torch.empty((0, 3), dtype=torch.long) if empty else torch.tensor([[7, 1, 2]]),
            tree_cache_tokens=torch.tensor([[1, 0, 0]]),
            tree_cache_logits=torch.tensor([[[1., 2., 3.], [4., 5., 6.], [0., 0., 0.]]]),
            tree_cache_valid_k=torch.tensor([2]), tree_cache_is_tree=torch.tensor([False]),
            tree_cache_activations=None, _last_n_draft_keys=0, jit_speculate=jit)

    records = []
    cases = [('hit', False, [7, 1, 2], True), ('wrong_token', False, [7, 1, 1], False),
             ('wrong_position', False, [7, 0, 2], False), ('wrong_sequence', False, [8, 1, 2], False),
             ('empty', True, [7, 1, 2], False)]
    env = {k: '0' for k in ('SSD_TREE_STAGE2', 'SSD_TREE_ROOT_SHADOW', 'SSD_PROFILE_DUET')}
    with patch.dict(sys.modules, {'ssd.engine.helpers.p2_tree': module}), patch.dict(os.environ, env):
        for name, empty, key, hit in cases:
            obj = runner(empty)
            calls.clear()
            values = lookup(obj, torch.tensor([key]), 1, 3, torch.tensor([10]),
                            torch.ones(1), torch.zeros((1, 2), dtype=torch.long))
            toks, logits, glue, hits, _, phase, valid, *_ = values
            assert bool(hits[0]) == hit
            assert int(valid[0]) == 2
            assert glue[0].item() == key[-1]
            if hit:
                assert not calls
                torch.testing.assert_close(toks, obj.tree_cache_tokens)
                torch.testing.assert_close(logits, obj.tree_cache_logits)
                assert int(phase[0]) == 2
            else:
                assert len(calls) == 1
                torch.testing.assert_close(toks[:, :2], torch.tensor([[2, 1]]))
                torch.testing.assert_close(logits[:, :2], torch.tensor([[[10., 20., 30.], [30., 20., 10.]]]))
                assert int(phase[0]) == 0
            records.append(dict(case=name, passed=True, cache_hit=hit))
    return records


def main():
    torch.set_num_threads(2)
    ns = {'torch': torch, 'nn': torch.nn}
    verify = load_node('ssd/ssd/utils/verify.py', 'verify', ns)
    sampler = load_node('ssd/ssd/layers/sampler.py', 'Sampler', ns)()
    cases = [([.5, .3, .2], [.7, .2, .1]),
             ([.3, .3, .4], [.28, .29, .43]),
             ([.5, .25, .15, .1], [.45, .26, .17, .12])]
    mc = [monte_carlo(verify, sampler, p, q, 20260911+i) for i, (p, q) in enumerate(cases)]
    cache = cache_checks()
    # Negative control: accepting q-generated proposals using p/e is biased.
    p = np.array([.5, .3, .2]); q = np.array([.7, .2, .1]); e = np.array([.2, .3, .5])
    accepted = q*np.minimum(1, p/e)
    residual = np.maximum(p-e, 0); residual /= residual.sum()
    wrong = accepted+(1-accepted.sum())*residual
    assert .5*np.abs(wrong-p).sum() > .2
    # Original campaign recorded hashes for the integration/probe paths.
    campaign = json.loads((ROOT/'ssd/experiments/proxy_source_ablation/probe_replay_20260910/CAMPAIGN.json').read_text())
    tracked = {k: hashlib.sha256((ROOT/k).read_bytes()).hexdigest() == sha
               for k, sha in campaign['measurement_sha256'].items() if k.endswith('.py')}
    assert all(tracked.values())
    result = dict(scope='CPU production sampler + chain verify (600000 proposals), and B=1 cache lookup with JIT model execution stubbed',
                  monte_carlo=mc, cache_response=cache,
                  wrong_denominator_control=dict(output=wrong.tolist(), target=p.tolist(), tv=float(.5*np.abs(wrong-p).sum())),
                  original_campaign_tracked_code_unchanged=tracked, production_code_sha256=HASHES,
                  limitation='Not an end-to-end GPU/TP/KV correctness proof; ordinary floating-point sampling tolerances apply. Original experiment target is AWQ, not the unquantized checkpoint.')
    (OUT/'followup_verify_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
