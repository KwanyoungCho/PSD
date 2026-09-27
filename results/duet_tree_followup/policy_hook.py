"""Opt-in research changes; no edits to production files or old experiments."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
from allocation import allocate, make_constants

HERE = Path(__file__).resolve().parent
POLICIES = {
    'q_path': dict(score='q_path', gain=False, frontier=False),
    'reach': dict(score='phase_sibling_q_bin', gain=False, frontier=False),
    'q_gain': dict(score='q_path', gain=True, frontier=False),
    'reach_gain': dict(score='phase_sibling_q_bin', gain=True, frontier=False),
    'reach_frontier': dict(score='phase_sibling_q_bin', gain=False, frontier=True),
    'reach_gain_frontier': dict(score='phase_sibling_q_bin', gain=True, frontier=True),
}


def frontier_function(module):
    original = module._arena_select_global
    source = textwrap.dedent(inspect.getsource(original))
    old = '(ar.depth == f)'
    if source.count(old) != 1:
        raise RuntimeError('Frontier source changed; review required')
    ns = dict(module.__dict__)
    exec(compile(source.replace(old, '(ar.depth <= f)'), str(__file__)+':frontier', 'exec'), ns)
    return ns['_arena_select_global']


def install():
    import ssd.engine.helpers.p2_tree as pt
    import ssd.engine.helpers.p2_tree_executor as pe
    policy = os.environ['DUET_TREE_POLICY']
    spec = POLICIES[policy]
    if os.environ['DUET_TREE_SCORE_MODE'] != spec['score']:
        raise ValueError('Policy and score mode disagree')
    cls = pe.P2TreeExecutor
    if getattr(cls, '_followup_installed', False):
        return
    if spec['frontier']:
        pt._arena_select_global = frontier_function(pt)
    if spec['gain']:
        calibration = json.loads((HERE/'gain_calibration.json').read_text())
        original_init = cls.__init__
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if self.C != 3 or self.policy not in ('dynamic', 'eagle'):
                raise ValueError('Gain hook supports the declared dynamic C=3 trial only')
            for phase in ('p1', 'p2'):
                if getattr(self.cfg, f'duet_{phase}_tree_max_nodes') != getattr(self.cfg, f'duet_{phase}_tree_verify_nodes'):
                    raise ValueError('Gain allocation requires G=M')
            phase = '1' if self.phase == 'p1' else '2'
            self.arena._gain_constants = make_constants(self.NV, calibration['curves'][phase], self.dev)
        cls.__init__ = init
        def fanout(ar, sel, sel_valid, remaining, c_tensor, R, future_rounds):
            root = ar.root.gather(0, sel.clamp(min=0))
            pri = ar.logpri.gather(0, sel.clamp(min=0))
            return allocate(root, pri, sel_valid, remaining, future_rounds, ar._gain_constants)
        pt._arena_fanout_global = fanout
    cls._followup_installed = True
    print('[tree followup hook] '+policy+'; gain SHA256='+
          hashlib.sha256((HERE/'gain_calibration.json').read_bytes()).hexdigest(), flush=True)
