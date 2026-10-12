"""Actual paper checkout, narrow opt-in ports, identical benchmark adapters."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

# Set in every spawned worker, before any ssd import.
CHECKOUT = Path(os.environ.get('PAPER_CHECKOUT', str(Path(__file__).resolve().parents[2])))
if os.getenv('PAPER_CPUSET'):
    cpus=set()
    for part in os.environ['PAPER_CPUSET'].split(','):
        ends=[int(v) for v in part.split('-')]
        cpus.update(range(ends[0],ends[-1]+1))
    os.sched_setaffinity(0,cpus)
sys.path.insert(0, str(CHECKOUT/'ssd'))
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
from mlsys_ver01_hooks import install
install(os.environ.get('PAPER_ARM', 'base'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='/home/chokwans99/models_mlsys/layerskip-llama2-7b')
    p.add_argument('--draft', default='/home/chokwans99/models_mlsys/AMD-Llama-135m')
    p.add_argument('--mode', choices=['tree', 'chain', 'ssd', 'sd', 'ar'], default='tree')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output-tokens', type=int, default=96)
    p.add_argument('--input-cap', type=int, default=512)
    p.add_argument('--seed', type=int, default=9101)
    p.add_argument('--warm-prompts', type=int, default=16)
    p.add_argument('--exit-layer', type=int, default=21)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--p1-verify', type=int, default=8)
    p.add_argument('--p2-verify', type=int, default=4)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    from ssd import LLM, SamplingParams
    from ssd.engine.llm_engine import METRICS
    from transformers import AutoTokenizer
    token = AutoTokenizer.from_pretrained(args.target)
    src = json.loads(args.prompts.read_text())
    indexes = list(range(len(src)))
    if 0 < args.limit < len(src): indexes = [i*len(src)//args.limit for i in range(args.limit)]
    prompts = [token.encode(src[i] if isinstance(src[i], str) else src[i]['turns'][0])[:args.input_cap] for i in indexes]
    tree, spec = args.mode == 'tree', args.mode != 'ar'
    kwargs = dict(num_gpus=2 if spec else 1, max_num_seqs=args.batch,
        max_model_len=2048, max_num_batched_tokens=args.batch*2048,
        gpu_memory_utilization=.45, speculate=spec, draft_async=spec and args.mode != 'sd',
        jit_speculate=True, speculate_k=6, async_fan_out=3)
    if spec: kwargs.update(draft=args.draft, duet_enabled=args.mode in ('tree','chain'), duet_exit_layer=args.exit_layer,
        duet_phase1_k=4, duet_phase2_k=2, duet_draft_fan_out=2,
        duet_p1_tree_policy='on' if tree else 'off', duet_p2_tree_policy='on' if tree else 'off',
        duet_p1_tree_max_nodes=8, duet_p1_tree_verify_nodes=args.p1_verify,
        duet_p2_tree_max_nodes=4, duet_p2_tree_verify_nodes=args.p2_verify,
        duet_tree_c_tensor=3, duet_p1_roots_per_position=2)
    if os.getenv('PAPER_BASELINE', '0') != '1':
        kwargs['greedy_only' if args.temperature == 0 else 'stochastic_only'] = True
    arm = os.environ.get('PAPER_ARM', 'base')
    report = dict(status='running', arm=arm, args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        paper_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip(),
        paper_dirty=subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT,text=True),
        source_hashes={str(f.relative_to(CHECKOUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (CHECKOUT/'ssd/ssd').rglob('*.py')},
        adapter_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('mlsys_ver01*.py')},
        kwargs=kwargs, prompt_ids=prompts, question_indexes=indexes,
        env={k:v for k,v in os.environ.items() if k.startswith(('SSD_','PAPER_')) or k=='CUDA_VISIBLE_DEVICES'},
        groups=[], warm_groups=[], steps=[], prefill=[], outputs=[])
    def save():
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report));temp.replace(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);save()
    llm=None
    try:
        llm=LLM(args.target, **kwargs)
        # Capture scheduler output without changing admission or model timing.
        current={}; original_schedule=llm.scheduler.schedule
        def schedule():
            seqs, pref=original_schedule()
            current.update(seqs=list(seqs),prefill=pref, before=[s.num_tokens for s in seqs])
            return seqs,pref
        llm.scheduler.schedule=schedule
        original_step=llm.step
        group_steps=[];group_prefill=[]
        def step(which):
            et=len(METRICS['phase_events']);tv=len(METRICS['target_verify_times'])
            dt=METRICS['decode_total_time'];pt=METRICS['prefill_total_time']
            result=original_step(which)
            seqs=current['seqs'];pref=current['prefill']
            if pref is True:
                group_prefill.append(dict(seconds=METRICS['prefill_total_time']-pt,batch=len(seqs)))
            elif pref is False:
                events=copy.deepcopy(METRICS['phase_events'][et:])
                emitted=[s.num_tokens-n for s,n in zip(seqs,current['before'])]
                cap=any(s.num_completion_tokens>=s.max_new_tokens for s in seqs)
                clipped=any(e['accepted_len']!=n for e,n in zip(events,emitted)) if spec else False
                group_steps.append(dict(seconds=METRICS['decode_total_time']-dt,batch=len(seqs),
                    cap=cap,clipped=clipped,emitted=sum(emitted),events=events,
                    verify=METRICS['target_verify_times'][tv:]))
            return result
        llm.step=step
        # Explicit fixed-size groups keep the old scheduler's resident set
        # bounded without importing the new scheduler into the baseline.
        phases=[('warm',prompts[:args.warm_prompts]),('measured',prompts)]
        for phase, data in phases:
            random.seed(args.seed-1 if phase=='warm' else args.seed)
            torch.manual_seed(args.seed-1 if phase=='warm' else args.seed)
            for start in range(0,len(data),args.batch):
                group_steps.clear();group_prefill.clear()
                params=SamplingParams(temperature=args.temperature,max_new_tokens=args.output_tokens,ignore_eos=False)
                torch.cuda.synchronize();t=time.perf_counter()
                outputs,metrics=llm.generate(data[start:start+args.batch],params,use_tqdm=False)
                torch.cuda.synchronize();wall=time.perf_counter()-t
                records=dict(index=start,n=len(outputs),wall=wall,steps=copy.deepcopy(group_steps),
                    prefill=copy.deepcopy(group_prefill),outputs=[o['token_ids'] for o in outputs])
                records['native_performance']=copy.deepcopy(metrics.get('performance'))
                report['warm_groups' if phase=='warm' else 'groups'].append(records)
                if phase=='measured':
                    report['steps'].extend(records['steps']);report['prefill'].extend(records['prefill']);report['outputs'].extend(records['outputs'])
                if start%16==0:save();print('PAPER_GROUP',arm,phase,start,flush=True)
        included=[s for s in report['steps'] if s['batch']==args.batch and not s['cap'] and not s['clipped']]
        events=[e for s in included for e in s['events']]
        def mean(v):
            v=[x for x in v if x is not None]
            return sum(v)/len(v) if v else None
        report['summary']=dict(steps=len(included),all_steps=len(report['steps']),
            tokens=sum(s['emitted'] for s in included),seconds=sum(s['seconds'] for s in included),
            tps=sum(s['emitted'] for s in included)/sum(s['seconds'] for s in included),
            verify_ms=mean([v*1000 for s in included for v in s['verify']]),
            step_ms=mean([s['seconds']*1000 for s in included]),
            al=mean([e['accepted_len'] for e in events]),hit=mean([e.get('cache_hit') for e in events]),
            actual_output_tokens=sum(map(len,report['outputs'])),
            emitted_decode_tokens=sum(s['emitted'] for s in report['steps']))
        if spec and report['summary']['actual_output_tokens']!=report['summary']['emitted_decode_tokens']:
            raise AssertionError('Actual output/decode token accounting mismatch')
        report['status']='complete';save();print('PAPER_COMPLETE',json.dumps(report['summary']),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));save();raise
    finally:
        if llm:llm.exit(hard=False)

if __name__=='__main__':main()
