"""Unadapted 70B runtime correctness smoke; short outputs are not TPS evidence."""
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


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'ssd/ssd').rglob('*.py'))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True)
    p.add_argument('--draft',required=True)
    p.add_argument('--mode',choices=('tree','ar'),default='tree')
    p.add_argument('--expansion',choices=('last','frontier'),default='last')
    p.add_argument('--normalization',choices=('full','topm'),default='topm')
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from transformers import AutoTokenizer
    from ssd import LLM,SamplingParams
    from ssd.presets import duet_70b_paper_kwargs
    tok=AutoTokenizer.from_pretrained(a.target,use_fast=True)
    prompts=['The capital of France is',
             'Write a Python function that adds two integers.\n',
             'Translate to French: The weather is nice today.\n',
             'Explain why the sky appears blue in one sentence.\n']
    inputs=[tok.encode(x) for x in prompts]
    kwargs=duet_70b_paper_kwargs(a.draft,expansion=a.expansion,normalization=a.normalization)
    kwargs.update(max_num_seqs=a.batch,greedy_only=a.temperature==0,
                  stochastic_only=a.temperature>0)
    if a.batch>1:kwargs['gpu_memory_utilization']=.55
    if a.mode=='ar':
        kwargs=dict(dtype='bfloat16',num_gpus=2,max_num_seqs=a.batch,max_model_len=2048,
                    max_num_batched_tokens=16384,gpu_memory_utilization=.7,
                    greedy_only=a.temperature==0,stochastic_only=a.temperature>0)
    report=dict(status='running',level='70b_bf16_functional',started=time.time(),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        kwargs=kwargs,inputs=inputs,groups=[],source_hashes=hashes(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        env={k:v for k,v in os.environ.items() if k.startswith(('SSD_','CUDA_VISIBLE'))})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.output)
    llm=None;save()
    try:
        random.seed(42);torch.manual_seed(42)
        llm=LLM(a.target,**kwargs)
        if llm.config.hf_config.torch_dtype!=torch.bfloat16:raise ValueError('Expected BF16 target')
        if llm.config.hf_config.num_hidden_layers!=80:raise ValueError('Expected 80-layer 70B target')
        for offset in range(0,len(inputs),a.batch):
            output,metrics=llm.generate(inputs[offset:offset+a.batch],
                SamplingParams(temperature=a.temperature,max_new_tokens=32),use_tqdm=False)
            ids=[x['token_ids'] for x in output]
            if len(ids)!=len(inputs[offset:offset+a.batch]):raise ValueError('Missing output')
            if any(not 0<len(row)<=32 or any(not 0<=v<tok.vocab_size for v in row) for row in ids):
                raise ValueError('Invalid token/count output')
            report['groups'].append(dict(offset=offset,outputs=ids,
                al=metrics.get('accepted_suffix_lens_with_recovery'),hit=metrics.get('cache_hits'),
                p1_hit=metrics.get('phase1_hits'),p2_hit=metrics.get('phase2_hits')))
            save()
        report.update(status='complete',source_unchanged=report['source_hashes']==hashes(),ended=time.time())
        if not report['source_unchanged']:raise RuntimeError('Runtime source changed during smoke')
        save()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),ended=time.time());save();raise
    finally:
        if llm is not None:llm.exit(hard=False)


if __name__=='__main__':main()
