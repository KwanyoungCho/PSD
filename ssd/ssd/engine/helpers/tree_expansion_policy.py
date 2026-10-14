"""Frozen local reach/gain/frontier selection, before new child sampling.

Only future expansion changes. Original q and ordered-WOR samples are kept.
Historical calibration ships with the package and is never fitted to evaluation outputs.
"""
import hashlib
import json
import os
from pathlib import Path
import torch
from ssd.engine.helpers.tree_gain_allocation import allocate, make_constants


class TreeExpansionPolicy:
    def __init__(self, executor):
        # The serving default uses draft path scores without fitted tables.
        # Historical calibrated modes remain explicit reproduction inputs.
        self.mode = os.getenv('SSD_TREE_EXPANSION_POLICY', 'q_path')
        modes = ('q_path', 'reach', 'q_gain', 'reach_gain',
                 'reach_frontier', 'reach_gain_frontier')
        if self.mode not in modes:
            raise ValueError(f'Unknown expansion policy {self.mode}')
        self.requested_mode = self.mode
        self.fallback_reason = None
        self.calibration_hashes = {}
        def report(reason=None):
            self.fallback_reason = reason
            print('[tree-policy-resolved] '+json.dumps(dict(
                phase=executor.phase, requested=self.requested_mode,
                resolved=self.mode, fallback=reason,
                calibration_sha256=self.calibration_hashes,
                allocation=executor.policy, width=getattr(executor,'W',None),
                round_widths=getattr(executor,'round_widths',None),
                roots=getattr(executor,'R',None), depth=getattr(executor,'F',None), nodes=executor.NV,
                siblings=executor.C),sort_keys=True),flush=True)
        self.frontier = False
        self.gain = None
        self.alpha = None
        if self.mode == 'q_path':
            report()
            return
        ex = executor
        # Chain phases are intentionally unchanged in mixed chain/tree runs.
        if getattr(ex.cfg, f'duet_{ex.phase}_tree_policy', 'off') != 'on':
            self.mode = 'q_path'
            report('phase tree disabled')
            return
        if ex.policy not in ('dynamic', 'eagle'):
            self.mode = 'q_path'
            report('allocation not dynamic/eagle')
            return
        if ex.C > 3 or ex.NV > 32:
            # Historical calibration covers C<=3. Never extrapolate its
            # missing sibling gains or silently claim a different policy.
            print(f'[tree expansion] {ex.phase}: q_path fallback; '
                  f'calibration requires C<=3 and exact allocator NV<=32; '
                  f'got C={ex.C}, NV={ex.NV}', flush=True)
            self.mode = 'q_path'
            report('calibration only covers C<=3 and NV<=32')
            return
        phase = '1' if ex.phase == 'p1' else '2'
        self.frontier = 'frontier' in self.mode
        def read(key):
            filename = 'reach.json' if 'REACH' in key else 'gain.json'
            path = Path(os.environ.get(key, Path(__file__).with_name('tree_calibration') / filename))
            data = path.read_bytes()
            self.calibration_hashes[key] = hashlib.sha256(data).hexdigest()
            print(f'[tree expansion] {self.mode} {ex.phase} {key} '
                  f'SHA256={hashlib.sha256(data).hexdigest()}', flush=True)
            return json.loads(data)
        if 'reach' in self.mode:
            model = read('SSD_TREE_REACH_CALIBRATION')
            means = model['means']
            table = []
            for sibling in range(ex.C):
                key = phase + ':' + str(sibling)
                default = means.get(key, means.get(phase, means['all']))
                table.append([means.get(key+':'+str(b), default)
                              for b in range(len(model['bins'])-1)])
            self.alpha = torch.tensor(table, device=ex.dev, dtype=torch.float64)
            self.bins = torch.tensor(model['bins'][1:-1], device=ex.dev,
                                     dtype=torch.float64)
        if 'gain' in self.mode:
            model = read('SSD_TREE_GAIN_CALIBRATION')
            curve = model['curves'][phase]
            if len(curve) < ex.C+1:
                raise ValueError('Gain curve must cover C')
            self.gain = make_constants(ex.NV, curve[:ex.C+1], ex.dev)
        report()

    def priority(self, parent_logpri, parents, raw_q):
        if self.alpha is None:
            return parent_logpri.gather(0, parents) + raw_q.double().reshape(-1).clamp_min(1e-9).log()
        bins = torch.bucketize(raw_q.double().contiguous(), self.bins, right=True)
        sib = torch.arange(raw_q.shape[1], device=raw_q.device)[None, :]
        a = self.alpha[sib, bins].clamp(1e-9, 1-1e-9)
        rejection = torch.log1p(-a)
        edge = a.log() + rejection.cumsum(1) - rejection
        return parent_logpri.gather(0, parents) + edge.reshape(-1)

    def fanout(self, arena, selected, valid, remaining, future):
        root = arena.root.gather(0, selected.clamp(min=0))
        reach = arena.logpri.gather(0, selected.clamp(min=0))
        return allocate(root, reach, valid, remaining, future, self.gain)
