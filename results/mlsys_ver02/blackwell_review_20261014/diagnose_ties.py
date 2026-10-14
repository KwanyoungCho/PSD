import os,sys
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers import root_policy as rp,flat_tree_sampling as fs,batch_tree_sampling as bs
from ssd.engine.helpers.p2_tree import q_probs_from_logits,unpack_piv
B,N,V=8,8,128256;dev='cuda';dtype=torch.bfloat16
ps=[[-1,-1,0,0]]*B;ss=[[0,1,0,1]]*B
pt=bs.pack_topologies(ps,ss,N,dev,2);ft=fs.topology(ps,ss,N,2,dev,40)
torch.manual_seed(712)
tok=torch.arange(1,N+1,device=dev).expand(B,N)
logits=torch.randn(B,N+1,V,device=dev,dtype=dtype);ql=torch.randn(B,N,V,device=dev,dtype=dtype)
t=torch.full((B,),.3,device=dev)
p=rp.probabilities(logits,t);q=rp.probabilities(ql,t)
fpl=logits[ft['req'],ft['local']];fql=fs.pack_nodes(ql,ft);ftok=fs.pack_nodes(tok,ft)
fp=(fpl.float()/t[ft['req'],None]).softmax(-1);fq=q_probs_from_logits(fql,t[ft['req']],None,1)
_,term,_=bs.ladder(tok,p,q,pt,N,overlap_mix=.25,need_residual=False)
_,fterm,_=fs.ladder(ftok,fp,fq,ft,N,overlap_mix=.25,need_residual=False)
qext=torch.cat([q,q.new_zeros(B,1,V)],1)
dq=qext[torch.arange(B,device=dev)[:,None],pt['child'][:,:,0]]
dscore=torch.where(pt['child_valid'][:,:,:1],rp.token_scores(p,dq,'complement'),p)
fscore=torch.where(ft['child_valid'][:,:1],rp.token_scores(fp,fq[ft['child'][:,0]],'complement'),fp)
for label,a,b in [('p',p[ft['req'],ft['local']],fp),('q',fs.pack_nodes(q,ft),fq),('score',dscore[ft['req'],ft['local']],fscore),('term',term[ft['req'],ft['local']],fterm)]:
 print(label,'max',(a-b).abs().max().item(),'non_equal',int((a!=b).sum()),flush=True)
row=5;ctx=1;frow=int(ft['dense'][row,ctx]);ids=torch.tensor([4168,83356],device=dev)
for name,a,b in [('p',p,fp),('qparent',dq,fq[ft['child'][:,0]]),('score',dscore,fscore)]:print(name,a[row,ctx,ids].tolist(),b[frow,ids].tolist(),flush=True)
print('term example',term[row,ctx].item(),fterm[frow].item())
# Check scores after the two differently-shaped child exclusion operations.
exclude=(pt['par']+1).clamp(0,N)*V+tok;exclude=torch.where(pt['node_valid'],exclude,torch.full_like(exclude,(N+1)*V))
dflat=torch.cat([dscore.reshape(B,-1),dscore.new_zeros(B,1)],1);dflat.scatter_(1,exclude,0);dscore=dflat[:,:-1].view(B,N+1,V)
ex=torch.where(ft['child_valid'],ftok[ft['child']],V)
fscore=torch.cat([fscore,fscore.new_zeros(40,1)],1).scatter(1,ex,0)[:,:V]
print('strides',dscore.stride(),fscore.stride())
print('denom',dscore.sum(-1)[row,ctx].item(),fscore.sum(-1)[frow].item())
a,ai=dscore.topk(13,-1);b,bi=fscore.topk(13,-1)
print('tok rows',ai[row,ctx].tolist(),bi[frow].tolist())
print('value rows',a[row,ctx].tolist(),b[frow].tolist())
print('raw dtype identical?',torch.equal(dscore[row,ctx],fscore[frow]))
