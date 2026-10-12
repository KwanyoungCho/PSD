"""Reproduce review findings and measure only the removed dispatch check."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = Path(os.environ.get("REVIEW_BASE", "/home/chokwans99/PSD-ver01-review-base"))
os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
sys.path.insert(0, str(ROOT / "ssd"))
import numpy as np
import torch
from ssd.engine.helpers import p2_tree as new_tree
from ssd.engine.draft_runner import _should_run_p2_tree, _tree_temperatures_supported
from ssd.utils.verify import verify


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old_verify = load("review_old_verify", BASE / "ssd/ssd/utils/verify.py")
old_tree = load("review_old_tree", BASE / "ssd/ssd/engine/helpers/p2_tree.py")
old_draft = load("review_old_draft", BASE / "ssd/ssd/engine/draft_runner.py")
out = dict(status="running", baseline="4b0d208", source_hashes={
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (ROOT / "ssd/ssd").rglob("*.py")})

p = torch.zeros(2, 5, 8)
for pos in range(5):
    p[:, pos, pos + 1] = 10
specs = torch.tensor([[0, 1, 2, 3, 4]] * 2)
args = (p, None, specs, torch.zeros(2), torch.zeros(2))
kwargs = dict(valid_k=torch.tensor([4, 2]))
out["ragged_reproduction"] = dict(before=old_verify.verify(*args, **kwargs),
                                   after=verify(*args, **kwargs))
assert out["ragged_reproduction"]["before"][1] == [5, 5]
assert out["ragged_reproduction"]["after"][1] == [5, 3]


def rollout(mod, arena, dev, temp, c, seed, greedy=False):
    logits = torch.tensor([[3., 3., 1., -1., -2., -3.]], device=dev)
    samples = []
    sample = mod.tree_sample_wor
    def record(*args, **kwargs):
        result = sample(*args, **kwargs)
        samples.append(tuple(t.detach().cpu().tolist() for t in result))
        return result
    kw = dict(policy="dynamic", W=1, F_total=2, c_tensor=c,
              nv=2, beta=.5, depth_cap=2, temps=torch.full((1,), temp, device=dev),
              forward_fn=lambda *args: logits.clone(),
              glue_rows_by_root=np.ones((1, 1), dtype=np.uint8),
              rope_base_by_root=[2], K_glue=0, context_len=4)
    if mod is new_tree:
        kw["greedy_only"] = greedy
    if arena:
        kw["device"] = dev
    torch.manual_seed(seed)
    rng_before = torch.cuda.get_rng_state().clone() if dev == "cuda" else torch.get_rng_state().clone()
    with patch.object(mod, "tree_sample_wor", record):
        fn = mod.run_rollout_arena if arena else mod.run_rollout
        pool, _, _ = fn([4], torch.tensor([1.]), **kw)
        if arena:
            pool = pool.to_pool(1)
    state = torch.cuda.get_rng_state().clone() if dev == "cuda" else torch.get_rng_state().clone()
    # Compare topology and raw scores, as well as all sampled children.
    data = dict(samples=samples, nodes={k: getattr(pool, k)[:pool.n].tolist()
                for k in ("tok", "parent_idx", "sib_order", "raw_q")})
    return data, state, torch.equal(rng_before, state)


out["fallback_greedy"] = []
out["fallback_positive_parity"] = 0
for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
    for arena in (False, True):
        old_wrong = new_wrong = 0
        for seed in range(16):
            before, _, _ = rollout(old_tree, arena, dev, 0., 3, seed)
            after, _, unchanged = rollout(new_tree, arena, dev, 0., 3, seed, True)
            old_wrong += sum(s[0][0][0] != 0 for s in before["samples"])
            new_wrong += sum(s[0][0][0] != 0 for s in after["samples"])
            assert unchanged
        assert old_wrong > 0 and new_wrong == 0
        out["fallback_greedy"].append(dict(device=dev, arena=arena,
             child_draws=32, before_not_argmax=old_wrong, after_not_argmax=new_wrong,
             after_rng_unchanged=True))
        for temp in (.7, 1.):
            for c in (1, 3):
                for seed in range(4):
                    before, bs, _ = rollout(old_tree, arena, dev, temp, c, seed)
                    after, ns, _ = rollout(new_tree, arena, dev, temp, c, seed)
                    assert before == after and torch.equal(bs, ns)
                    out["fallback_positive_parity"] += 1

if torch.cuda.is_available():
    temps = torch.tensor([.7], device="cuda")
    cfg = SimpleNamespace(duet_p2_tree_policy="on", greedy_only=False, stochastic_only=True)
    funcs = [lambda: old_draft._should_run_p2_tree(cfg, 1, temps),
             lambda: _should_run_p2_tree(cfg, 1, temps)]
    for fn in funcs:
        for _ in range(20):
            assert fn()
    profile = []
    for fn in funcs:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            for _ in range(10):
                assert fn()
        profile.append({e.key: e.count for e in prof.key_averages()
                        if "item" in e.key or "local_scalar" in e.key})
    assert profile[0].get("aten::_local_scalar_dense") == 10
    assert profile[1].get("aten::_local_scalar_dense", 0) == 0
    times = [[], []]
    for rep in range(9):
        for arm in ((0, 1) if rep % 2 == 0 else (1, 0)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(300):
                funcs[arm]()
            torch.cuda.synchronize()
            times[arm].append((time.perf_counter() - t) * 1e6 / 300)
    out["dispatch_microbenchmark"] = dict(before_us=statistics.median(times[0]),
         after_us=statistics.median(times[1]), samples_us=times,
         profiler_10_calls=profile,
         scope="Idle GPU host dispatch check only; excludes draft forwards and is not a TPS estimate.")

out["unit_suites"] = []
for name in ("test_ver01_review", "test_tree_accept_port", "test_p1_dynamic_tree",
             "test_p2_tree_alloc", "test_p2_executor_parity", "test_tree_verify_planless"):
    suite = unittest.defaultTestLoader.discover(str(ROOT / "ssd/tests"), pattern=name + ".py")
    with (HERE / (name + ".log")).open("w") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    row = dict(name=name, tests=result.testsRun, failures=len(result.failures),
               errors=len(result.errors), skipped=len(result.skipped))
    out["unit_suites"].append(row)
    (HERE / "CHECKS.json").write_text(json.dumps(out, indent=2) + "\n")
    print(row, flush=True)
    assert result.wasSuccessful() and not result.skipped

subprocess.run([sys.executable, str(HERE.parent / "check_paper_parity.py")],
               cwd=ROOT, check=True)
out["paper_cpu_parity"] = json.loads((HERE.parent / "PAPER_PARITY.json").read_text())
out["status"] = "complete"
(HERE / "CHECKS.json").write_text(json.dumps(out, indent=2) + "\n")
print("PASS", flush=True)
