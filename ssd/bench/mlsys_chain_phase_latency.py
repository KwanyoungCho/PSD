"""Local B1 native-chain measurement, exactly 40 roots in each phase.

Only config arguments and CPU shape auditing are added. Native sampling,
draft forward graphs and target verification are unchanged.
"""
import json
import os
from pathlib import Path
import socket
import sys

if socket.gethostname().split('.')[0]!='orion':
    raise RuntimeError('This measurement is restricted to local orion')
os.environ.setdefault('SSD_CUDA_ARCH','8.9')
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mlsys_ps_preparation import main
from mlsys_phase_compilation import install as install_preparation_audit
install_preparation_audit()

import ssd
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers import cudagraph_helpers as profiling

ROOTS=40
K=10
FAN_OUT=[4]*7+[3]*4
assert len(FAN_OUT)==K+1 and sum(FAN_OUT)==ROOTS
_LLM=ssd.LLM


def exact_chain(*args,**kwargs):
    if (kwargs.get('max_num_seqs')!=1 or kwargs.get('duet_phase1_k')!=K
            or kwargs.get('duet_phase2_k')!=K
            or kwargs.get('duet_p1_tree_policy')!='off'
            or kwargs.get('duet_p2_tree_policy')!='off'):
        raise ValueError('Expected B1, K1=K2=10 and native chain in both phases')
    kwargs.update(duet_split_phase1_fan_out_list=FAN_OUT.copy(),
                  duet_p2_forward_width=ROOTS,duet_p2_root_count=ROOTS)
    llm=_LLM(*args,**kwargs)
    cfg=llm.config
    assert not cfg.duet_tree_enabled
    assert cfg.duet_p2_seed_count==cfg.duet_proxy_total_budget==ROOTS
    assert sum(cfg.duet_split_phase1_fan_out_list)==ROOTS
    assert cfg.duet_proxy_wire_N==82
    data=dict(kwargs=kwargs,ds_fan_out=FAN_OUT,ds_roots=ROOTS,ps_roots=ROOTS,
              k1=K,k2=K,wire_N=cfg.duet_proxy_wire_N,native_chain=True,
              note='DS near-uniform allocation over K+1 positions. PS native score selection with DS dedup. C/tree-node caps are inactive.')
    (Path(os.environ['SSD_PROFILE_DIR'])/'RESOLVED_CONFIG.json').write_text(json.dumps(data,indent=2)+'\n')
    return llm


ssd.LLM=exact_chain
_merge=DraftRunner._merge_and_populate_cache
_prefill=DraftRunner.draft_async_prefill
_exit=DraftRunner.exit


def flush(runner):
    records=getattr(runner,'_chain_shape_records',[])
    if not records:return
    epoch=getattr(runner,'_chain_shape_epoch',0)
    (Path(os.environ['SSD_PROFILE_DIR'])/f'chain_shapes_request_{epoch:04d}.json').write_text(
        json.dumps(dict(epoch=epoch,records=records),indent=2)+'\n')
    runner._chain_shape_records=[]
    runner._chain_shape_epoch=epoch+1


def merge(self,*args,**kwargs):
    result=_merge(self,*args,**kwargs)
    if self.tree_cache_keys.shape!=(80,3) or self.tree_cache_tokens.shape!=(80,K):
        raise AssertionError('Expected exactly 40 DS + 40 PS chains of depth10')
    if self._last_n_draft_keys!=ROOTS:
        raise AssertionError('Incorrect phase boundary')
    if self.split_k1_long_layout.MQ_LEN!=ROOTS or self.split_k2_layout.MQ_LEN!=ROOTS:
        raise AssertionError('Forward width is not40')
    if not hasattr(self,'_chain_shape_records'):self._chain_shape_records=[]
    self._chain_shape_records.append(dict(step_id=profiling._duet_context['step_id'],
        ds_roots=ROOTS,ps_roots=ROOTS,ds_width=ROOTS,ps_width=ROOTS,k1=K,k2=K))
    return result


def prefill(self,*args,**kwargs):
    flush(self)
    return _prefill(self,*args,**kwargs)


def exit(self,*args,**kwargs):
    flush(self)
    return _exit(self,*args,**kwargs)


DraftRunner._merge_and_populate_cache=merge
DraftRunner.draft_async_prefill=prefill
DraftRunner.exit=exit

if __name__=='__main__':main()
