"""GPU parity and graph timing for the frozen, target-blind candidate rules."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from runtime_policy import make_policy
import ssd.engine.helpers.p2_tree as p2
spec=importlib.util.spec_from_file_location('tf',HERE/'replay.py')
tf=importlib.util.module_from_spec(spec);spec.loader.exec_module(tf)

@torch.inference_mode()
def main():
    torch.manual_seed(914);torch.set_num_threads(4)
    frozen=json.loads((HERE/'frozen.json').read_text())
    results=[]
    for t in [.5,.7,1.]:
        keys=set(frozen['selection'][str(t)][n] for n in ['overall','proxy_allocation'])
        keys.add('policy::proxy__topm__original')
        for key in sorted(keys):
            fn=make_policy(key,t,p2.pack_piv)
            maxerror=0.
            for k,dtype in [(k,d) for d in [torch.float32,torch.float16,torch.bfloat16] for k in [1,2,3,4]]:
                z=(torch.randn(k+1,32000,device='cuda')*4).to(dtype)
                zq=(torch.randn(k,32000,device='cuda')*4).to(dtype)
                y=zq.argmax(-1);wire=min(15,(k+1)*17)
                pos,tok,score=fn(z,zq,y,17,wire,False)
                e=(z.float()/t).softmax(-1)[None]
                q=torch.cat([(zq.float()/t).softmax(-1),torch.zeros_like(zq[:1],dtype=torch.float32)])[None]
                yy=torch.cat([y,torch.full_like(y[:1],-1)])[None]
                src,norm,hname=key.removeprefix('policy::').split('__')
                if src=='interval1':src='proxy'
                s=dict(tf.sources(e,q,yy))[src]
                val,ids=s.topk(17,-1)
                den=val.sum(-1,keepdim=True) if norm=='topm' else s.sum(-1,keepdim=True)
                allscore=tf.pos_weights(e,q,yy)[hname][...,None]*val/den.clamp_min(1e-10)
                expected,ix=allscore.flatten().topk(wire)
                torch.testing.assert_close(score,expected,atol=3e-7,rtol=2e-5)
                torch.testing.assert_close(pos,ix//17)
                torch.testing.assert_close(tok,ids.flatten()[ix])
                maxerror=max(maxerror,float((score-expected).abs().max()))
            # Representative K=4 graph: repeated kernels, inputs fixed.
            for _ in range(3):fn(z,zq,y,17,15,False)
            torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn(z,zq,y,17,15,False)
            g.replay();torch.cuda.synchronize()
            for a,b in zip(out,fn(z,zq,y,17,15,False)):
                torch.testing.assert_close(a,b,atol=0,rtol=0)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(1000):g.replay()
            end.record();end.synchronize()
            results.append({'temperature':t,'policy':key,'max_score_error':maxerror,
                            'graph_ms':start.elapsed_time(end)/1000})
    (HERE/'runtime_checks.json').write_text(json.dumps({'passed':True,'checks':results,
       'scope':'Synthetic full-vocabulary K=1..4 bfloat16/float16/float32 input parity; CUDA graph/eager equality; isolated bfloat16-input GPU timing, not throughput.'},indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
