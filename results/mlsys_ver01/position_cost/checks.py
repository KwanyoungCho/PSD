"""Correctness/information checks for the component experiment, not serving."""
import json
from pathlib import Path
import torch
from compute import observed,hat_scalars,mixture,overlap,hazard


@torch.inference_mode()
def main():
    e=torch.tensor([.2,.5,.3],dtype=torch.float64)
    qa=torch.tensor([.4,.5,.1],dtype=torch.float64)
    qb=torch.tensor([.4,.1,.5],dtype=torch.float64)
    hat=lambda q:min(1.,float(e[0]/q[0]))
    bar=lambda q:float(torch.minimum(e,q).sum())
    residual=lambda q:((e-q).clamp_min(0)/(e-q).clamp_min(0).sum()).tolist()
    assert qa[0]==qb[0] and hat(qa)==hat(qb)==.5
    assert abs(bar(qa)-.8)<1e-12 and abs(bar(qb)-.6)<1e-12
    assert residual(qa)==[0.,0.,1.] and residual(qb)==[0.,1.,0.]
    gpu_cases=0
    for b,k,v in [(1,4,32000),(2,3,32001),(8,12,128256)]:
        for kind in ('random','identical','disjoint'):
            torch.manual_seed(718)
            x=torch.randn(b,k,v,device='cuda').softmax(-1)
            q=torch.randn_like(x).softmax(-1)
            if kind=='identical':q=x.clone()
            if kind=='disjoint':
                x.zero_();q.zero_();x[:,:,0]=1.;q[:,:,1]=1.
            y=torch.multinomial(q.reshape(-1,v),1).reshape(b,k,1)
            torch.testing.assert_close(overlap(x,q),torch.minimum(x,q).sum(-1),rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(mixture(x,q,y,True),mixture(x,q,y),rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(observed(x,q,y),hat_scalars(x.gather(-1,y).squeeze(-1),q.gather(-1,y).squeeze(-1)),rtol=0,atol=0)
            torch.testing.assert_close(observed(x,q,y).sum(-1),torch.ones(b,device='cuda'),rtol=2e-5,atol=2e-6)
            # Identity usable to share work with residual scoring. Excluding
            # the observed token must happen AFTER this full-vocab sum.
            torch.testing.assert_close(torch.minimum(x,q).sum(-1),1-(x-q).clamp_min(0).sum(-1),rtol=2e-5,atol=2e-6)
            gpu_cases+=1
    out=dict(scalar_information_counterexample={'e':e.tolist(),'q_a':qa.tolist(),'q_b':qb.tolist(),
        'observed_alpha':[hat(qa),hat(qb)],'mean_alpha':[bar(qa),bar(qb)],
        'correction_a':residual(qa),'correction_b':residual(qb)},
        gpu_cases=gpu_cases,gpu=torch.cuda.get_device_name(),status='PASS',
        scope='Position/overlap formulas and component kernels; no full inference correctness claim.')
    Path(__file__).with_name('CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
