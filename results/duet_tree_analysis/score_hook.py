"""Opt-in research integration: change ONLY future expansion priority.

No token is removed based on its realized score. Sampling, raw q, root scores,
fanout admission, verification, thresholds and wire tensors stay original.
Use G=M. This prototype retains current depth==round frontier restriction.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import torch


def tensor_priority(parent_logpri, par, raws, boundaries, table):
    bins=torch.bucketize(raws.double(),boundaries,right=True)
    sibling=torch.arange(raws.shape[1],device=raws.device).unsqueeze(0).expand_as(bins)
    a=table[sibling,bins].clamp(1e-9,1-1e-9)
    rejection=torch.log1p(-a)
    edge_log=a.log()+rejection.cumsum(1)-rejection
    return parent_logpri.gather(0,par)+edge_log.reshape(-1)


def install():
    if os.environ.get('DUET_TREE_SCORE_MODE', 'q_path') == 'q_path': return
    if os.environ['DUET_TREE_SCORE_MODE'] != 'phase_sibling_q_bin':
        raise ValueError('Unregistered score mode')
    import ssd.engine.helpers.p2_tree_executor as module
    cls = module.P2TreeExecutor
    if getattr(cls, '_audit_score_installed', False): return
    model_path = Path(os.environ['DUET_TREE_SCORE_CALIBRATION'])
    model = json.loads(model_path.read_text())
    original_init = cls.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for phase in ['p1','p2']:
            if getattr(self.cfg,f'duet_{phase}_tree_max_nodes') != getattr(self.cfg,f'duet_{phase}_tree_verify_nodes'):
                raise ValueError('Research score hook requires G=M in both phases')
        phase = '1' if self.phase=='p1' else '2'
        table=[]
        for sibling in range(self.C):
            key=phase+':'+str(sibling)
            mean=model['means'].get(key,model['means'].get(phase,model['means']['all']))
            table.append([model['means'].get(key+':'+str(b),mean) for b in range(len(model['bins'])-1)])
        self._audit_alpha=torch.tensor(table,dtype=torch.float64,device=self.dev)
        self._audit_bins=torch.tensor(model['bins'][1:-1],dtype=torch.float64,device=self.dev)
    source = textwrap.dedent(inspect.getsource(cls.run_once))
    old='lp = ar.logpri.gather(0, par) + rq.clamp_min(1e-9).log()'
    if source.count(old) != 1: raise RuntimeError('Production score location changed; review required')
    # Explicit arena parameter avoids relying on private attribute spelling.
    new='lp = _audit_priority(self, ar, par, raws)'
    def adapted(executor, arena, par, raws):
        return tensor_priority(arena.logpri,par,raws,executor._audit_bins,executor._audit_alpha)
    namespace=dict(module.__dict__,_audit_priority=adapted)
    exec(compile(source.replace(old,new),str(Path(__file__).resolve())+':run_once','exec'),namespace)
    cls.run_once=namespace['run_once'];cls.__init__=init;cls._audit_score_installed=True
    print('[tree score hook] phase_sibling_q_bin; calibration SHA256='+
          hashlib.sha256(model_path.read_bytes()).hexdigest(),flush=True)
