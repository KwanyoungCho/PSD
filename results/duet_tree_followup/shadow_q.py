"""Score every observed tree node, including unexpanded leaves, with TinyLlama.

Target p for those nodes already exists in raw verifier snapshots. This adds
only a draft forward. Explicit tree mask and positions avoid serial per-leaf
prefills. Validate against saved production q at every expanded parent.
HF/BF16 numerical differences are measured, never silently treated as exact q.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE=Path(__file__).resolve().parent;POST=HERE.parent/'duet_tree_posthoc'
FULL=HERE.parent/'duet_tree_al_full'


def production_model():
    os.environ.setdefault('SSD_HF_CACHE','/data/chokwans99/models')
    os.environ.setdefault('SSD_DATASET_DIR',str(FULL/'dataset'))
    os.environ.setdefault('SSD_CUDA_ARCH','8.9')
    sys.path.insert(0,str(HERE.parents[1]/'ssd'))
    from transformers import AutoConfig
    from ssd.models.llama3 import LlamaForCausalLM
    from ssd.layers.attention import Attention
    from ssd.utils.loader import load_model
    state={}
    def attention(self,q,k,v):
        length=q.shape[0]
        qq=q.reshape(length,self.num_heads,self.head_dim).transpose(0,1)[None]
        kk=k.reshape(length,self.num_kv_heads,self.head_dim).transpose(0,1)[None]
        vv=v.reshape(length,self.num_kv_heads,self.head_dim).transpose(0,1)[None]
        out=torch.nn.functional.scaled_dot_product_attention(qq,kk,vv,attn_mask=state['mask'],
            enable_gqa=True,dropout_p=0.,scale=self.head_dim**-.5)
        return out[0].transpose(0,1).reshape(length,-1)
    Attention.forward=attention
    path='/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0';config=AutoConfig.from_pretrained(path,local_files_only=True)
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    with torch.device('cuda'):
        model=LlamaForCausalLM(config,draft=True,tp_size=1)
    torch.set_default_dtype(previous);load_model(model,path)
    model.eval();model._shadow_state=state
    return model


def reconstruct(policy, tokenizer):
    questions={q['question_id']:q for q in json.loads((FULL/'questions.json').read_text())}
    records={};histories={}
    for line in (POST/'runs'/policy/'records.jsonl').open():
        r=json.loads(line);qid=r['question_id'];question=questions[qid]
        current=tokenizer.encode(question['turns'][r['turn']],add_special_tokens=False)
        prompt=current if r['turn']==0 else histories[qid]+tokenizer.encode('\n\n',add_special_tokens=False)+current
        if hashlib.sha256(json.dumps(prompt).encode()).hexdigest()!=r['input_sha256']:
            raise ValueError('Historical prompt reconstruction failed')
        r['prompt_ids']=prompt;records[r['uid']]=r;histories[qid]=prompt+r['output_ids']
    return records


@torch.inference_mode()
def probabilities(model, prefix, par, tok, depth, temperature):
    n=len(par);length=len(prefix);total=length+n
    ids=torch.tensor([prefix+list(map(int,tok))],device='cuda')
    pos=torch.tensor([list(range(length))+[length-1+int(d) for d in depth]],device='cuda')
    mask=torch.zeros(total,total,dtype=torch.bool,device='cuda')
    mask[:length,:length]=torch.ones(length,length,dtype=torch.bool,device='cuda').tril()
    for j in range(n):
        mask[length+j,:length]=True
        ancestor=j
        while ancestor>=0:
            mask[length+j,length+ancestor]=True;ancestor=int(par[ancestor])
    additive=torch.where(mask,0.,torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[None,None]
    if hasattr(model,'_shadow_state'):
        model._shadow_state['mask']=additive
        hidden=model(ids[0],pos[0]);logits=model.compute_logits(hidden[length-1:],False).float()
    else:
        hidden=model.model(input_ids=ids,position_ids=pos,attention_mask=additive,use_cache=False).last_hidden_state
        logits=model.lm_head(hidden[:,length-1:]).float()[0]
    return torch.softmax(logits/temperature,dim=-1).cpu().numpy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--backend',choices=['hf','production'],default='production');args=ap.parse_args()
    dest=HERE/(('shadow_q_smoke' if args.smoke else 'shadow_q')+('_production' if args.backend=='production' else ''))
    if (dest/'records.jsonl').exists() and (dest/'records.jsonl').stat().st_size==0:
        dest.rename(dest.with_name(dest.name+'_failed_empty'))
    dest.mkdir(exist_ok=True)
    if (dest/'records.jsonl').exists():raise FileExistsError('Do not overwrite a shadow attempt')
    tokenizer=AutoTokenizer.from_pretrained('/home/chokwans99/awq_calibrated/layerskip_llama2_70b',local_files_only=True)
    if args.backend=='production':model=production_model()
    else:
        model=AutoModelForCausalLM.from_pretrained('/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0',
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().to('cuda')
    rows=[];start=time.time()
    with (dest/'records.jsonl').open('x',buffering=1) as sink:
        for policy in ('q_path','phase_sibling_q_bin'):
            records=reconstruct(policy,tokenizer)
            trees=[json.loads(s) for s in (POST/'runs'/policy/'trees.jsonl').open()]
            trees=[t for t in trees if 'raw_file' in t and not t['is_final_event']]
            if args.smoke:
                trees=[trees[i] for i in np.linspace(0,len(trees)-1,12).astype(int)]
            for t in trees:
                r=records[t['uid']];before=sum(e['accepted_len'] for e in r['metrics']['phase_events'][:t['event_index']])
                prefix=r['prompt_ids']+r['output_ids'][:before+1]
                if len(r['output_ids'])<before+1:raise ValueError('Missing current root token')
                with np.load(POST/'runs'/policy/t['raw_file']) as z:
                    q=probabilities(model,prefix,t['par'],t['tok'],t['depth'],t['temperature'])
                    comparisons=[]
                    for parent in sorted(set(t['par'])):
                        child=t['par'].index(parent);old=z['q'][child].astype(float);old/=old.sum()
                        new=q[parent+1].astype(float);new/=new.sum()
                        comparisons.append(dict(parent=parent,tv=float(abs(old-new).sum()/2),
                            max_abs=float(abs(old-new).max()),top1_match=bool(old.argmax()==new.argmax())))
                    if not np.isfinite(q).all():raise ValueError('Nonfinite shadow probability')
                    name=f'{policy}_{t["serial"]:06d}.npz';np.savez_compressed(dest/name,q=q)
                row={k:t[k] for k in ('policy','question_id','group','uid','phase','serial','raw_file')}
                row.update(file=name,sha256=hashlib.sha256((dest/name).read_bytes()).hexdigest(),
                    prefix_tokens=len(prefix),contexts=len(q),known_parent_checks=comparisons)
                rows.append(row);sink.write(json.dumps(row)+'\n')
                (dest/'status.json').write_text(json.dumps(dict(trees=len(rows),last_uid=t['uid'],wall_s=time.time()-start)))
                if len(rows)%25==0:print('shadow trees',len(rows),'elapsed',round(time.time()-start,1),flush=True)
    checks=[c for r in rows for c in r['known_parent_checks']];tv=np.array([c['tv'] for c in checks])
    result=dict(complete=True,smoke=args.smoke,backend=args.backend,trees=len(rows),questions=len({r['question_id'] for r in rows}),
        known_parent_checks=len(checks),known_q_tv_mean=float(tv.mean()),known_q_tv_quantiles=np.quantile(tv,[.5,.9,.95,.99,1]).tolist(),
        known_q_top1_match=float(np.mean([c['top1_match'] for c in checks])),
        numerical_gate_passed=bool(np.quantile(tv,.5)<.005 and np.quantile(tv,.95)<.02),
        wall_s=time.time()-start,scope='Same-prefix BF16 draft shadow with SDPA tree attention; target p reused from actual verifier; unexpanded observed leaves only.')
    (dest/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['numerical_gate_passed']:raise ValueError('Shadow q fails numerical agreement gate; investigate before interpretation')


if __name__=='__main__':main()
