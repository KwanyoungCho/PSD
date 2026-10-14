"""Paired review of redundant tree proposal staging; no target/TPS claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import gc
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ssd'))
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
import torch
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy, BatchedTreeAccept
from ssd.engine.helpers.flat_tree_sampling import FlatProxy, FlatAccept


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'ssd/ssd').rglob('*.py'))}


def summary(samples):
    rng = np.random.default_rng(814)
    a, b = np.asarray(samples['staged']), np.asarray(samples['direct'])
    diff = b - a
    boot = diff[rng.integers(0, len(diff), (3000, len(diff)))].mean(1)
    return dict(median_ms={k: statistics.median(v) for k, v in samples.items()},
                mean_ms={k: statistics.mean(v) for k, v in samples.items()},
                paired_mean_change_ms=float(diff.mean()),
                paired_mean_change_ci95_ms=np.quantile(boot, [.025, .975]).tolist())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=80)
    args = ap.parse_args()
    report = dict(status='running', scope='same logits/topology; preparation and combined proxy/accept; target excluded',
                  gpu=torch.cuda.get_device_name(), source_hashes=hashes(), cells=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for v in (32000, 128256):
      for cap, b, n, width, compact in ((2, 2, 4, 4, False), (8, 8, 4, 4, False),
                                      (8, 8, 4, 4, True), (8, 8, 8, 4, False),
                                      (8, 8, 8, 4, True), (4, 3, 8, 4, False)):
        torch.manual_seed(813)
        dtype = torch.bfloat16
        lengths = [width if not compact or i % 2 == 0 else 0 for i in range(b)]
        parents = [[-1, -1, 0, 0][:z] for z in lengths]
        siblings = [[0, 1, 0, 1][:z] for z in lengths]
        tokens = torch.arange(1, width + 1, device='cuda').expand(b, width)
        q = torch.randn(b, width, v, device='cuda', dtype=dtype)
        lp = torch.randn(b, n+1, v, device='cuda', dtype=dtype)
        temps = [.7] * b
        cls, ac = (FlatProxy, FlatAccept) if compact else (BatchedTreeProxy, BatchedTreeAccept)
        proxy = cls(cap, n, v, dtype, 'cuda', 15, n, 13, c_max=3)
        def prepare(arm):
            if arm == 'staged':
                t = torch.zeros(b, n, device='cuda', dtype=torch.int64)
                t[:, :width].copy_(tokens)
                z = torch.zeros(b, n, v, device='cuda', dtype=dtype)
                z[:, :width].copy_(q)
            else:
                t, z = tokens, q
            proxy.prepare(parents, siblings, t, z, temps, temps)
        prepare('staged')
        accept = ac(proxy, None, 1, False, stochastic_only=True)
        def logits():
            return lp[proxy.current['topo']['req'], proxy.current['topo']['local']] if compact else lp
        def snapshot():
            c = proxy.current if compact else dict(q=proxy.q, tokens=proxy.tokens)
            return {k: c[k].clone() for k in ('q', 'tokens')}
        expected = snapshot()
        prepare('direct')
        for k, value in snapshot().items():
            if not torch.equal(expected[k], value):
                raise AssertionError(f'Prepared {k} changed')
        for seed in range(10):
            prepare('staged')
            px = tuple(t.clone() for t in proxy.replay(logits()))
            torch.cuda.manual_seed(seed)
            ax = tuple(t.clone() for t in accept.replay(logits(), temps, temps))
            rng = torch.cuda.get_rng_state().clone()
            prepare('direct')
            py = proxy.replay(logits())
            torch.cuda.manual_seed(seed)
            ay = accept.replay(logits(), temps, temps)
            if any(not torch.equal(x, y) for x, y in zip((*px, *ax), (*py, *ay))):
                raise AssertionError('Proxy/walk changed')
            if not torch.equal(rng, torch.cuda.get_rng_state()):
                raise AssertionError('RNG changed')
        samples = {scope: {a: [] for a in ('staged', 'direct')}
                   for scope in ('prepare', 'proxy_accept')}
        def run(scope, arm):
            prepare(arm)
            if scope == 'proxy_accept':
                proxy.replay(logits())
                return accept.replay(logits(), temps, temps)
        for scope in samples:
            for _ in range(5):
                for arm in samples[scope]: run(scope, arm)
            for i in range(args.repeats):
                for arm in (('staged', 'direct') if i % 2 == 0 else ('direct', 'staged')):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    run(scope, arm)
                    torch.cuda.synchronize()
                    samples[scope][arm].append((time.perf_counter() - start) * 1000)
        row = dict(batch=b, capacity=cap, verify_width=n, payload_width=width, vocab=v,
                   compact=compact, parity_seeds=10, samples_ms=samples,
                   summary={k: summary(x) for k, x in samples.items()},
                   removed_temporary_bytes=b*n*(v*torch.tensor([], dtype=dtype).element_size()+8))
        report['cells'].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(v, b, n, width, compact, row['summary'], flush=True)
        del proxy, accept, q, lp, expected, px, ax, py, ay
        gc.collect()
        torch.cuda.empty_cache()
    report['source_unchanged'] = report['source_hashes'] == hashes()
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2))
    if not report['source_unchanged']: raise RuntimeError('Source changed during measurement')


if __name__ == '__main__': main()
