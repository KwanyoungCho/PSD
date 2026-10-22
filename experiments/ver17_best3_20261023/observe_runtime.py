"""Startup evidence only; native model, verifier and dispatch remain untouched."""
import hashlib,json,os
from pathlib import Path
import torch
from ssd.engine.model_runner import ModelRunner
_original=ModelRunner.__init__
def init(self,*args,**kwargs):
    _original(self,*args,**kwargs)
    role='draft' if self.is_draft else 'target'
    dtype='torch.'+os.environ['DUET_MEASURE_DTYPE']
    actual=dict(role=role,parameter=str(next(self.model.parameters()).dtype),kv_cache=str(self.kv_cache.dtype),
      hf_dtype=str(self.hf_config.torch_dtype),enforce_eager=self.enforce_eager,target_tp=self.num_tp_gpus,
      device_name=torch.cuda.get_device_name(self.device),torch=torch.__version__,nccl=torch.cuda.nccl.version(),
      tree_graphs=len(getattr(self,'_tree_verify_cg',{})),
      fused_tree_graphs=sum('head_graphs' in x['bufs'] for x in getattr(self,'_tree_verify_cg',{}).values()),
      fused_chain_graphs=[k for k,v in self.graph_vars.items() if 'head_graphs' in v])
    assert all(actual[k]==dtype for k in ('parameter','kv_cache','hf_dtype')),actual
    assert not self.enforce_eager
    if role=='target':
        graph=os.environ.get('SSD_VERIFY_GRAPH_LOGITS','0')=='1'
        assert bool(actual['fused_tree_graphs'])==graph,actual
        assert bool(actual['fused_chain_graphs'])==graph,actual
    actual['rng_domains']={f'p1_context{c}':ex.gen.initial_seed() for c,ex in getattr(self,'_p1_execs',{}).items()}
    if getattr(self,'_p2_exec',None) is not None:
        actual['rng_domains']['p2']=self._p2_exec.gen.initial_seed()
    p=Path(os.environ['SSD_PROFILE_DIR'])/f'ACTUAL_{role}.json'
    p.write_text(json.dumps(actual,indent=2)+'\n')
    print('ACTUAL_RUNTIME',json.dumps(actual),flush=True)
ModelRunner.__init__=init
