"""Diagnostic logits at identical prefixes; timings are NOT performance results."""
import hashlib
import json
import os
from pathlib import Path

import mlsys_ver01_hooks as hooks
original_install=hooks.install

def key(tokens):
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()

def install(arm):
    original_install(arm)
    import torch
    from ssd.engine.model_runner import ModelRunner
    from ssd.engine.helpers.p2_tree import parse_tree_ints
    focus={x['key']:x for x in json.loads(Path(os.environ['GREEDY_TRACE_FOCUS']).read_text())}
    directory=Path(os.environ['GREEDY_TRACE_DIR']); directory.mkdir(parents=True,exist_ok=True)
    old_run,old_model=ModelRunner.run,ModelRunner.run_model
    def run(self,seqs,*args,**kwargs):
        self._trace_tokens=[list(s.token_ids) for s in seqs]
        return old_run(self,seqs,*args,**kwargs)
    def model(self,input_ids,positions,is_prefill,last_only=True,*args,**kwargs):
        logits=old_model(self,input_ids,positions,is_prefill,last_only,*args,**kwargs)
        if self.is_draft or self.rank!=0 or not isinstance(logits,torch.Tensor):return logits
        tokens=self._trace_tokens[0]
        if is_prefill or last_only:
            prefixes=[tokens]; rows=[logits.shape[0]-1]
        else:
            ids=input_ids.tolist();pos=positions.tolist()
            base=tokens[:pos[0]]
            meta=getattr(self,'_duet_tree_meta',None)
            if meta is not None:
                nv=(len(meta)-3)//4;ti=parse_tree_ints(meta,nv)
                prefixes=[base+[ids[0]]]
                for j in range(ti['valid']):
                    par=ti['parent_local'][j]
                    prefixes.append(prefixes[par+1]+[ti['tok'][j]])
            else:prefixes=[base+ids[:i+1] for i in range(len(ids))]
            rows=list(range(len(prefixes)))
        flat = logits.reshape(-1, logits.shape[-1])
        for prefix,row in zip(prefixes,rows):
            k=key(prefix)
            if k not in focus:continue
            file=directory/(k+'.pt')
            if file.exists():continue
            value=flat[row].detach().float().cpu().clone()
            torch.save(value,file)
            vals,inds=value.topk(16)
            record=dict(**focus[k],row=row,shape=list(logits.shape),is_prefill=is_prefill,
                last_only=last_only,actual_argmax=int(value.argmax()),dtype=str(logits.dtype),
                top_ids=inds.tolist(),top_values=vals.tolist(),
                tree_meta=getattr(self,'_duet_tree_meta',None))
            with (directory/'records.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
        return logits
    ModelRunner.run=run;ModelRunner.run_model=model
hooks.install=install
path=Path(__file__).with_name('mlsys_ver01.py')
exec(compile(path.read_text(),str(path),'exec'))
