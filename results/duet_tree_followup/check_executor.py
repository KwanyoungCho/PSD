"""Actual executor kernels + attention, using the repo's tiny model fixture.

This covers real P1/P2 widths, changing page buckets, ordered q references and
graph/eager equivalence. It is not a real 70B/TinyLlama inference validation.
Run each policy in a fresh process because hooks are intentionally process-local.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ssd'));sys.path.insert(0,str(ROOT/'ssd/tests'))
sys.path.insert(0,str(HERE.parent/'duet_tree_analysis'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('policy');args=ap.parse_args()
    os.environ.setdefault('SSD_HF_CACHE','/data/chokwans99/models')
    os.environ.setdefault('SSD_DATASET_DIR',str(HERE.parent/'duet_tree_al_full/dataset'))
    os.environ.setdefault('SSD_CUDA_ARCH','8.9')
    from policy_hook import POLICIES,install
    os.environ['DUET_TREE_POLICY']=args.policy
    os.environ['DUET_TREE_SCORE_MODE']=POLICIES[args.policy]['score']
    os.environ['DUET_TREE_SCORE_CALIBRATION']=str(HERE.parent/'duet_tree_analysis/calibration_frozen.json')
    os.environ['SSD_TREE_NODE_AUDIT']='fixture'
    if POLICIES[args.policy]['score']!='q_path':
        from score_hook import install as score_install
        score_install()
    install()
    from test_p2_executor_parity import _MiniCfg,_MiniDraft,TestExecutorModuleParity
    from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
    from ssd.engine.draft_runner import DraftRunner
    class Model(_MiniDraft):
        def logits_fn(self,hidden,last_only):
            logits=super().logits_fn(hidden,last_only)
            return logits+torch.linspace(-16,0,logits.shape[-1],device=logits.device)
    dev='cuda:0';page=64;v=128;h=4;hkv=2;dim=64;rows=[];audits=[]
    helper=TestExecutorModuleParity();cfg=_MiniCfg()
    cfg.duet_tree_policy='eagle';cfg.duet_p1_tree_max_nodes=cfg.duet_p1_tree_verify_nodes=8
    cfg.duet_p2_tree_max_nodes=cfg.duet_p2_tree_verify_nodes=6
    cfg.duet_p1_tree_start_threshold=cfg.duet_p1_tree_conf_threshold=0.
    writer=SimpleNamespace(_append_tree_node_audit=lambda kind,record:audits.append(record))
    for phase,r,widths,budget in [('p1',9,(9,9,9,9),8),('p1',21,(21,15,15,15),8),
                                  ('p1',27,(27,15,15,15),8),('p2',15,(15,15),6)]:
        cache=torch.zeros(12,2,page,hkv,dim,dtype=torch.float16,device=dev)
        model=Model(v,h,hkv,dim,cache,dev)
        ex=P2TreeExecutor(model,model.logits_fn,cfg,dev,page,12,v,h,hkv,dim,
            phase=phase,width=max(widths),root_count=r,depth=len(widths),max_nodes=budget,
            round_widths=widths,materialize_backbone_logits=False)
        ex.debug_buffers_enabled=True
        gen=torch.Generator(device=dev);gen.manual_seed(720+r)
        ex.parity_noise=[torch.empty(ex.W,v,device=dev).exponential_(1,generator=gen) for _ in widths]
        for ctx in (page+31,2*page+31):
            bucket=(ctx+page-1)//page;ex.prepare_bucket(bucket);helper._fill_inputs(ex,page,ctx)
            ex._local_idx=torch.full((ex.arena.capacity,),-1,dtype=torch.long,device=dev)
            cache.zero_();ex.run_once(bucket)
            fields=('view_tok','view_par','view_sib','view_pcell','out_valid','out_pq_ref','out_pq_cells','out_u_valid','dbg_sel','dbg_selv','dbg_fan')
            reference={k:getattr(ex,k).clone() for k in fields};logits=ex.cell_logits.clone();raw=ex.out_rawq.clone()
            cache.zero_();ex.capture(bucket);cache.zero_();ex.replay(bucket)
            for k,val in reference.items():
                if not torch.equal(val,getattr(ex,k)):raise ValueError(f'{phase} R{r} {k} differs')
            err=float((logits-ex.cell_logits).abs().max());qerr=float((raw-ex.out_rawq).abs().max())
            if err>1e-3 or qerr>1e-6:raise ValueError((err,qerr))
            views=dict(valid=ex.out_valid,parent_local=ex.view_par,sib_order=ex.view_sib,
                tok=ex.view_tok,raw_q=ex.view_rawq,parent_q_ref=ex.out_pq_ref,parent_q_cells=ex.out_pq_cells)
            DraftRunner._audit_tree_executor_node_coverage(writer,views,ex,r,phase=1 if phase=='p1' else 2)
            delayed=sum(int((ex.arena.depth[ex.dbg_sel[f,:w]][ex.dbg_selv[f,:w]]<f).sum()) for f,w in enumerate(widths))
            rows.append(dict(phase=phase,roots=r,widths=list(widths),bucket=bucket,logits_error=err,raw_q_error=qerr,
                nodes=int(ex.out_valid.sum()),delayed_expansions=delayed))
        # Return to the first captured bucket after a different bucket ran.
        helper._fill_inputs(ex,page,page+31);cache.zero_();ex.replay(2)
        DraftRunner._audit_tree_executor_node_coverage(writer,views,ex,r,phase=1 if phase=='p1' else 2)
        del ex,model,cache;gc.collect();torch.cuda.empty_cache()
    result=dict(passed=True,policy=args.policy,shape_buckets=rows,node_audits=len(audits),
        evaluated_contexts=sum(x['evaluated_contexts'] for x in audits),
        scope='Mini-model fixture; actual executor/attention kernels and parent q references, not the real model')
    (HERE/f'executor_{args.policy}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
