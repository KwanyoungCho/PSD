"""리뷰 지시: canvas 페이지 3-way — ①page0 중복 ②마지막페이지 중복
③전용 신선 페이지(sentinel 채움) — mask=0이면 셋 다 bit-동일이어야.
다르면 canvas 안전 전제(중복 조합) 폐기."""
import sys, torch
sys.path.insert(0, "/home/chokwans99/PSD/ssd")
import flashinfer

dev="cuda:0"; H,HKV,D,PAGE=32,4,64,256; W=10
def run(kvi_list, cache, q, mask_flat, p_total):
    ws=torch.empty(128*2**20,dtype=torch.uint8,device=dev)
    wr=flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws,"NHD",backend="fa2")
    kvi=torch.tensor(kvi_list,dtype=torch.int32,device=dev)
    wr.plan(torch.tensor([0,W],dtype=torch.int32,device=dev),
            torch.tensor([0,len(kvi_list)],dtype=torch.int32,device=dev),
            kvi, torch.tensor([PAGE],dtype=torch.int32,device=dev),
            H,HKV,D,PAGE,custom_mask=mask_flat,
            q_data_type=torch.float16,kv_data_type=torch.float16)
    return wr.run(q,(cache[:,0],cache[:,1])).float()

fails=0
for ctx0 in (533, 585, 640):        # 비교차: ctx0+40 <= p0*256? p0=3, 768 — 573..680 <768 ✓
    for trial in range(3):
        g=torch.Generator(device=dev).manual_seed(100*trial+ctx0)
        p0=(ctx0+PAGE-1)//PAGE
        kv_len=ctx0+W                 # round0
        # 물리 페이지 풀: p0 실페이지 + 1 신선(전용) — sentinel로 오염
        cache=(torch.randn(p0+1,2,PAGE,HKV,D,generator=g,device=dev)*0.1).half()
        cache[p0].fill_(777.0)        # 신선 페이지 = 큰 sentinel
        q=(torch.randn(W,H,D,generator=g,device=dev)*0.1).half()
        canvas=(p0+1)*PAGE
        m=torch.zeros(W,canvas,dtype=torch.bool,device=dev)
        m[:,:ctx0]=True
        for i in range(W): m[i,ctx0+i]=True   # 전부 실페이지 내 (비교차)
        mf=m.reshape(-1)
        o_p0dup = run(list(range(p0))+[0],      cache,q,mf,p0+1)  # page0 중복
        o_lastd = run(list(range(p0))+[p0-1],   cache,q,mf,p0+1)  # 마지막 중복
        o_fresh = run(list(range(p0))+[p0],     cache,q,mf,p0+1)  # 전용 sentinel
        d1=float((o_p0dup-o_fresh).abs().max())
        d2=float((o_lastd-o_fresh).abs().max())
        d3=float((o_p0dup-o_lastd).abs().max())
        tag="OK" if d1==d2==d3==0.0 else "FAIL"
        if tag=="FAIL": fails+=1
        print(f"ctx0={ctx0} t={trial}: p0dup-fresh={d1:.3e} last-fresh={d2:.3e} p0-last={d3:.3e} [{tag}]")
print(f"\n{fails} FAIL")
