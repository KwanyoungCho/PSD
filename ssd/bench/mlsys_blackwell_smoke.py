"""Small functional runs of the unadapted runtime; never a performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ssd'))
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',required=True)
    parser.add_argument('--draft',required=True)
    parser.add_argument('--mode',choices=('ar','tree'),required=True)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--temperature',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from ssd import LLM,SamplingParams
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.target,use_fast=True)
    prompts=[
        'The capital of France is',
        'Write a Python function that adds two integers.\n',
        'Translate to French: The weather is nice today.\n',
        'Explain why the sky appears blue in one sentence.\n']
    inputs=[tok.encode(x) for x in prompts]
    kwargs=dict(num_gpus=2 if args.mode=='tree' else 1,
        max_num_seqs=args.batch,max_model_len=2048,
        max_num_batched_tokens=args.batch*2048,gpu_memory_utilization=.45,
        greedy_only=args.temperature==0,stochastic_only=args.temperature>0)
    if args.mode=='tree':
        kwargs.update(draft=args.draft,speculate=True,draft_async=True,
            jit_speculate=True,duet_enabled=True,duet_exit_layer=21,
            speculate_k=6,duet_phase1_k=4,duet_phase2_k=2,
            async_fan_out=3,duet_draft_fan_out=3,duet_p2_budget=6,
            duet_p1_tree_policy='on',duet_p2_tree_policy='on',
            duet_p1_tree_max_nodes=8,duet_p2_tree_max_nodes=4,
            duet_p1_tree_verify_nodes=8,duet_p2_tree_verify_nodes=4,
            duet_tree_c_tensor=2,duet_p1_roots_per_position=1)
    report=dict(status='running',level='model_functional',started=time.time(),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        kwargs=kwargs,inputs=inputs,groups=[],source_hashes={
            str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'ssd/ssd').rglob('*.py')},
        env={k:v for k,v in os.environ.items() if k.startswith(('SSD_','CUDA_VISIBLE'))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.output)
    save();llm=None
    try:
        random.seed(734);torch.manual_seed(734)
        llm=LLM(args.target,**kwargs)
        for offset in range(0,len(inputs),args.batch):
            output,metrics=llm.generate(inputs[offset:offset+args.batch],
                SamplingParams(temperature=args.temperature,max_new_tokens=32),use_tqdm=False)
            ids=[x['token_ids'] for x in output]
            if len(ids)!=len(inputs[offset:offset+args.batch]):raise RuntimeError('Missing output')
            for row in ids:
                if not 0<len(row)<=32 or any(not 0<=t<tok.vocab_size for t in row):
                    raise RuntimeError('Output token/count contract')
            report['groups'].append(dict(offset=offset,outputs=ids,
                al=metrics.get('accepted_suffix_lens_with_recovery'),
                hit=metrics.get('cache_hits')))
            save()
        report.update(status='complete',ended=time.time());save()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),ended=time.time());save();raise
    finally:
        if llm is not None:llm.exit(hard=False)


if __name__=='__main__':main()
