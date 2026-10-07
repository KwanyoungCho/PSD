import torch,flashinfer,json
D='cuda';B=5;Q=48;P=2;S=256;H=32;HD=128;HK=8
ws=torch.empty(128*2**20,dtype=torch.uint8,device=D)
qo=torch.tensor([0,9,18,27,36,48],dtype=torch.int32,device=D)
kv=torch.arange(B+1,device=D,dtype=torch.int32)*P
ix=torch.arange(B*P,device=D,dtype=torch.int32)
last=torch.full((B,),S,dtype=torch.int32,device=D)
mask=torch.zeros(Q*P*S//8,dtype=torch.uint8,device=D)
w=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,'NHD',backend='fa2',use_cuda_graph=True,qo_indptr_buf=qo,paged_kv_indptr_buf=kv,paged_kv_indices_buf=ix,paged_kv_last_page_len_buf=last,custom_mask_buf=mask,mask_indptr_buf=qo.clone())
q=torch.randn(Q,H,HD,device=D,dtype=torch.float16)
k=torch.randn(B*P,S,HK,HD,device=D,dtype=torch.float16);v=torch.randn_like(k)
def plan(ptr):
    qp=torch.tensor(ptr,dtype=torch.int32)
    w.plan(qp,kv.cpu(),ix,last.cpu(),H,HK,HD,S,packed_custom_mask=torch.full_like(mask,255),q_data_type=torch.float16,kv_data_type=torch.float16)
    w._mask_indptr_buf.copy_(qp*(P*S//8))
plan([0,9,18,27,36,48]);a=w.run(q,(k,v));g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):out=w.run(q,(k,v))
base=tuple(int(x) for x in w._plan_info);records=[]
for ptr in [[0,3,8,17,26,48],[0,5,10,15,16,48],[0,9,9,9,9,48],[0,9,22,31,44,48]]:
    plan(ptr);g.replay();actual=out.clone();expected=w.run(q,(k,v));torch.cuda.synchronize()
    records.append(dict(qo=ptr,plan_equal=base==tuple(int(x) for x in w._plan_info),maxdiff=(actual-expected).abs().max().item(),base_plan=base,plan=tuple(int(x) for x in w._plan_info)));print(records[-1],flush=True)
open('results/mlsys_coverage/round3/packed_plan_gqa.json','w').write(json.dumps(records,indent=2))
