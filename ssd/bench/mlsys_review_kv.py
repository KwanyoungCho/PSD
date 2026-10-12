"""Real draft-model KV copy versus confirmed-chain recomputation diagnosis.

Synthetic equal contexts, retained legal tree KV, fragmented/contiguous pages.
This is not a new serving policy or a revival of removed KV promotion.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.config import Config
from ssd.engine.model_runner import ModelRunner
from ssd.engine.helpers.batch_tree_forward import BatchedTreeForward


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True)
    ap.add_argument('--dtype',choices=['fp16','bf16'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=15)
    ap.add_argument('--diagnose',action='store_true',help='Record numeric deviations and equal-shape controls; does not certify equivalence')
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(status='running',model=a.model,dtype=a.dtype,gpu=torch.cuda.get_device_name(),
                scope='draft full-model microbenchmark; retained valid tree KV; no network/serving/cache-hit policy',cases=[])
    def save():a.output.write_text(json.dumps(result,indent=2))
    save()
    cfg=Config(a.model,num_gpus=1,max_num_seqs=8,max_model_len=1024,
               max_num_batched_tokens=8192,gpu_memory_utilization=.18,
               enforce_eager=True,speculate=False,duet_p1_tree_policy='off',duet_p2_tree_policy='off')
    cfg.hf_config.torch_dtype=torch.float16 if a.dtype=='fp16' else torch.bfloat16
    runner=ModelRunner(cfg,rank=0,event=None,is_draft=False,num_tp_gpus=1)
    # Load one local model without spawning a target/draft process group,
    # then select the real async-draft ancestor-attention dispatch for this
    # isolated probe. No model weights or attention math are replaced.
    from ssd.layers.attention import Attention
    for module in runner.model.modules():
        if isinstance(module,Attention):
            module.draft=True;module.speculate=True;module.draft_async=True
    forward=BatchedTreeForward(runner,split=False)
    flat=runner.kv_cache.view(2,runner.kv_cache.shape[1],-1,runner.kv_cache.shape[4],runner.kv_cache.shape[5])
    bs=runner.block_size;v=runner.hf_config.vocab_size
    try:
      for b in (1,2,4,8):
       for context in (127,511):
        for c,n in ((1,8),(3,8),(3,16)):
         torch.manual_seed(900+b+context+n)
         prompts=torch.randint(20,min(v,32000),(b,context)).tolist()
         trees=torch.randint(20,min(v,32000),(b,n+1)).tolist()
         tips=torch.randint(20,min(v,32000),(b,)).tolist()
         pars=[j//c-1 for j in range(n)]
         node=n-1;path=[]
         while node>=0:path.append(node);node=pars[node]
         path.reverse()
         accepted=[0]+[j+1 for j in path]
         for layout in ('contiguous','fragmented'):
            pages=(context+n+2+bs-1)//bs
            allpages=runner.kv_cache.shape[2]
            ids=(torch.arange(b*pages) if layout=='contiguous' else torch.randperm(allpages)[:b*pages]).view(b,pages).tolist()
            def rows(tokens,parents,prefix):
                return [dict(tokens=tokens[i],parents=parents,prefix=prefix,blocks=ids[i]) for i in range(b)]
            def slots(offset,positions):
                return torch.tensor([ids[i][(offset+j)//bs]*bs+(offset+j)%bs
                                     for i in range(b) for j in positions],device=runner.device)
            prefix_rows=rows(prompts,list(range(-1,context-2)),0)
            forward.run(prefix_rows,context)
            tree_rows=rows(trees,pars,context)
            tree_slots=slots(context,range(n+1))
            forward.run(tree_rows,n+1)
            snapshot=flat[:,:,tree_slots].clone()
            source_indices=torch.tensor([i*(n+1)+j for i in range(b) for j in accepted],device=runner.device)
            destinations=slots(context,range(len(accepted)))
            chain_tokens=[[trees[i][j] for j in accepted]+[tips[i]] for i in range(b)]
            chain_rows=rows(chain_tokens,list(range(-1,len(accepted)-1)),context)
            tip_rows=rows([[t] for t in tips],[],context+len(accepted))
            def setup():flat[:,:,tree_slots]=snapshot
            def stage_restore():
                staged=flat[:,:,tree_slots].clone()
                flat[:,:,destinations]=staged[:,:,source_indices]
            def promotion():
                stage_restore()
                return forward.run(tip_rows,1)[:,-1]
            def glue():return forward.run(chain_rows,len(accepted)+1)[:,-1]
            # Capture both shapes before comparisons; writes during setup
            # are never timed as part of either treatment.
            setup();promotion();setup();glue()
            setup();p=promotion().float().clone();kv=flat[:,:,destinations].clone()
            setup();q=glue().float().clone();kv_ref=flat[:,:,destinations].clone()
            tv=(p.softmax(-1)-q.softmax(-1)).abs().sum(-1)*.5
            valid=bool(torch.isfinite(p).all() and torch.isfinite(q).all())
            check=dict(finite=valid,max_logit_abs=float((p-q).abs().max()),
                       logit_rmse=float((p-q).square().mean().sqrt()),max_tv=float(tv.max()),
                       max_kv_abs=float((kv-kv_ref).abs().max()),
                       argmax_equal=bool(torch.equal(p.argmax(-1),q.argmax(-1))))
            if a.diagnose:
                # Recompute the path, then use exactly the same one-token
                # forward shape for both KV sources. This isolates KV values
                # from shape-dependent tip projection/attention arithmetic.
                setup();glue();q_tip=forward.run(tip_rows,1)[:,-1].float().clone()
                setup();q_wide=forward.run(chain_rows,max(n+1,len(accepted)+1))[:,len(accepted)].float().clone()
                kv_wide=flat[:,:,destinations].clone()
                def compare(x,y):
                    return dict(max_tv=float(((x.softmax(-1)-y.softmax(-1)).abs().sum(-1)*.5).max()),
                                max_logit_abs=float((x-y).abs().max()),
                                argmax_equal=bool(torch.equal(x.argmax(-1),y.argmax(-1))))
                check['same_tip_shape_tree_vs_chain_kv']=compare(p,q_tip)
                check['glue_tip_shape_only']=compare(q,q_tip)
                check['tree_vs_equal_width_chain_kv_abs']=float((kv-kv_wide).abs().max())
                check['equal_width_chain_vs_glue']=compare(q_wide,q)
            check['tv_over_1pct']=check['max_tv']>.01
            if not valid or (check['tv_over_1pct'] and not a.diagnose):
                result['failed_case']=dict(batch=b,context=context,nodes=n,siblings=c,layout=layout,checks=check)
                raise AssertionError(f'KV reconstruction/context mismatch: {check}')
            times={name:[] for name in ('copy_only','promotion_plus_tip','confirmed_chain_glue')}
            fns=dict(copy_only=stage_restore,promotion_plus_tip=promotion,confirmed_chain_glue=glue)
            for rep in range(a.repeats):
                order=list(fns)
                if rep%2:order.reverse()
                for name in order:
                    setup();torch.cuda.synchronize()
                    start=time.perf_counter();fns[name]();torch.cuda.synchronize()
                    times[name].append((time.perf_counter()-start)*1000)
            med={k:statistics.median(x) for k,x in times.items()}
            result['cases'].append(dict(batch=b,context=context,nodes=n,siblings=c,layout=layout,
                accepted_nodes=len(path),glue_tokens=len(accepted)+1,checks=check,
                median_ms=med,timings_ms=times,
                staged_bytes=snapshot.numel()*snapshot.element_size()))
            save();print('KV_CASE',b,context,n,c,layout,med,check,flush=True)
      result['status']='complete_diagnostic' if a.diagnose else 'complete';save()
    except BaseException as e:
      result.update(status='failed',error=f'{type(e).__name__}: {e}');save();raise
    finally:runner.exit(hard=False)


if __name__=='__main__':main()
