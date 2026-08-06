"""fa2 preplanned vs auto JIT — 동일 입력·동일 mask에서 attention
출력 logit의 signed-mean/max 직접 측정. 실행기 hit 하락이 fa2의
systematic bias인지(mean≠0) 단순 noise인지(mean≈0) 판별.

미니모델 사용 (실 attention 계약 미러). tree 없이 single-forward의
attention만 비교 — 순수 kernel 대조.
"""
import sys, torch
sys.path.insert(0, "/home/chokwans99/PSD/ssd")
import flashinfer

dev = "cuda:0"
torch.manual_seed(0)
H, HKV, D, PAGE = 4, 2, 64, 64
V = 128
# 컨텍스트: p0+1 페이지, 마지막 페이지 부분채움
ctx0 = PAGE + 21
W = 10
for trial in range(5):
    g = torch.Generator(device=dev).manual_seed(trial)
    kv_len = ctx0 + W
    p = (kv_len + PAGE - 1) // PAGE
    lpl = kv_len - (p - 1) * PAGE
    # KV cache [pages,2,PAGE,HKV,D], q [W,H,D]
    cache = (torch.randn(p, 2, PAGE, HKV, D, generator=g,
                         device=dev) * 0.1).half()
    q = (torch.randn(W, H, D, generator=g, device=dev) * 0.1).half()
    # causal-ish mask: 각 lane은 prefix 전체 + 자기까지
    mask = torch.zeros(W, kv_len, dtype=torch.bool, device=dev)
    mask[:, :ctx0] = True
    lane = torch.arange(W, device=dev)
    for i in range(W):
        mask[i, ctx0 + i] = True
    qo = torch.tensor([0, W], dtype=torch.int32, device=dev)
    po = torch.tensor([0, p], dtype=torch.int32, device=dev)
    kvi = torch.arange(p, dtype=torch.int32, device=dev)
    lastp = torch.tensor([lpl], dtype=torch.int32, device=dev)

    def run(backend):
        ws = torch.empty(128 * 2**20, dtype=torch.uint8, device=dev)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend=backend)
        wr.plan(qo, po, kvi, lastp, H, HKV, D, PAGE,
                custom_mask=mask.reshape(-1),
                q_data_type=torch.float16, kv_data_type=torch.float16)
        o = wr.run(q, (cache[:, 0], cache[:, 1]))
        return o.float().reshape(W, H * D)

    o_auto = run("auto")
    o_fa2 = run("fa2")
    d = o_fa2 - o_auto
    print(f"trial {trial}: signed_mean={d.mean().item():+.2e} "
          f"abs_max={d.abs().max().item():.2e} "
          f"rel={d.abs().max().item()/o_auto.abs().max().item():.2e} "
          f"cos={torch.nn.functional.cosine_similarity(o_auto.flatten(), o_fa2.flatten(), dim=0).item():.6f}")
