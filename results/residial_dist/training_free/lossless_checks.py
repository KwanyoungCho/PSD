"""Production CPU sampling/verification at the tested nonunit temperatures."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit_shared',HERE.parent/'shared_review/followup_verify_checks.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def main():
    torch.set_num_threads(2)
    ns={'torch':torch,'nn':torch.nn}
    verify=old.load_node('ssd/ssd/utils/verify.py','verify',ns)
    sampler=old.load_node('ssd/ssd/layers/sampler.py','Sampler',ns)()
    records=[];n=100000
    for j,t in enumerate([1.,.7,.5]):
        torch.manual_seed(918+j)
        lp=torch.tensor([.5,.3,.2]).log();lq=torch.tensor([.7,.2,.1]).log()
        p=(lp/t).softmax(-1);q=(lq/t).softmax(-1)
        temp=torch.full((n,),t)
        y=sampler(lq.repeat(n,1),temp)
        proposals=torch.stack([torch.full_like(y,123456),y],-1)
        raw_p=torch.stack([lp,torch.tensor([.2,.3,.5]).log()]).repeat(n,1,1)
        raw_q=lq.repeat(n,1,1)
        rng=torch.get_rng_state();results=[]
        for hit in [False,True]:
            torch.set_rng_state(rng)
            results.append(verify(raw_p,raw_q,proposals,temp,temp,
                         cache_hits=torch.full((n,),hit),jit_speculate=True))
        assert results[0]==results[1]
        suffix,correction=results[0]
        assert all(x[0]==123456 for x in suffix)
        accept=np.array([len(x)==2 for x in suffix]);rec=np.asarray(correction)
        output=np.where(accept,y.numpy(),rec)
        hist=lambda x:np.bincount(x,minlength=3)/len(x)
        r=(p-q).clamp_min(0).numpy();r/=r.sum()
        records.append({'temperature':t,'proposals':n,'rejections':int((~accept).sum()),
          'proposal_error':old.check_hist(hist(y.numpy()),q.numpy(),n),
          'output_error':old.check_hist(hist(output),p.numpy(),n),
          'correction_error':old.check_hist(hist(rec[~accept]),r,int((~accept).sum())),
          'hit_miss_identical':True,'committed_root_not_reverified':True})
    out={'passed':True,'records':records,'production_code_sha256':old.HASHES,
         'scope':'300,000 CPU proposals using production sampler and verifier; same raw logits and actual T in q sampling/verification. Not a GPU/TP/KV distributional proof.'}
    (HERE/'lossless_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
