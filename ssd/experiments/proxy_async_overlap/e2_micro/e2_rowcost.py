"""E2 — verify 행당 한계비용의 standalone 분해 측정 (설계 v6 §9 E2①⑤, G0).

엔진 무수정. 목적: engine anchor(짝런 hit_k1 vs hit_k2 주기차
11.97ms/5행 ≈ 2.4ms/행 — wire 등 혼입 포함)에서 **계산 성분**(GEMM +
attention)이 얼마인지 분해 → 트리 행 추가의 irreducible 비용 확정.

방법 (근사 명시):
- 70B TP4의 **rank당** 실제 shape로 측정 (각 rank가 병렬로 자기 shard를
  돌므로 행당 한계비용 ≈ 단일 rank 측정): hidden 8192, q_heads/rank 16,
  kv_heads/rank 2, head_dim 128, intermediate/rank 7168, 80층.
- GEMM은 fp16 dense로 측정 — AWQ Marlin(4bit)은 weight 바이트가 1/4라
  더 memory-bound → M-민감도는 여기 측정치가 **상한**.
- 층 1개 가중치만 잡고 80회 호출 (가중치 427MB ≫ L2 6MB — 호출마다
  DRAM 재스트리밍이라 80층 순회와 동일 거동).
- attention: sdpa, kv_len 768 (champion 문맥 규모), GQA 확장.

Run: cd ssd && CUDA_VISIBLE_DEVICES=0 python experiments/proxy_async_overlap/e2_micro/e2_rowcost.py
"""
import torch

H, QH, KVH, HD, INTER, LAYERS = 8192, 16, 2, 128, 7168, 80
KV_LEN = 768
MS = [1, 5, 7, 9, 11, 13]
ITERS = 30


def bench(fn, iters=ITERS, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    torch.cuda.init()
    dev = "cuda"
    dt = torch.float16
    # 층 1개 가중치 (rank-shard shape)
    w_qkv = torch.randn((QH + 2 * KVH) * HD, H, device=dev, dtype=dt)
    w_o = torch.randn(H, QH * HD, device=dev, dtype=dt)
    w_gu = torch.randn(2 * INTER, H, device=dev, dtype=dt)
    w_dn = torch.randn(H, INTER, device=dev, dtype=dt)

    print(f"[GEMM] rank-shard 80층 등가 (fp16 — AWQ는 더 평평할 상한), "
          f"kv_len={KV_LEN}")
    print(f"{'M(행수)':>7} {'GEMM 80층(ms)':>13} {'attn 80층(ms)':>13} "
          f"{'합계(ms)':>9}")
    tot = {}
    for M in MS:
        x = torch.randn(M, H, device=dev, dtype=dt)
        xo = torch.randn(M, QH * HD, device=dev, dtype=dt)
        xi = torch.randn(M, INTER, device=dev, dtype=dt)

        def gemms():
            torch.mm(x, w_qkv.t())
            torch.mm(xo, w_o.t())
            torch.mm(x, w_gu.t())
            torch.mm(xi, w_dn.t())
        g = bench(gemms) * LAYERS

        q = torch.randn(1, QH, M, HD, device=dev, dtype=dt)
        k = torch.randn(1, KVH, KV_LEN + M, HD, device=dev, dtype=dt)
        v = torch.randn(1, KVH, KV_LEN + M, HD, device=dev, dtype=dt)
        ke = k.repeat_interleave(QH // KVH, dim=1)
        ve = v.repeat_interleave(QH // KVH, dim=1)
        mask = torch.ones(M, KV_LEN + M, device=dev, dtype=torch.bool)

        def attn():
            torch.nn.functional.scaled_dot_product_attention(
                q, ke, ve, attn_mask=mask)
        a = bench(attn) * LAYERS
        tot[M] = g + a
        print(f"{M:>7} {g:>13.2f} {a:>13.2f} {g + a:>9.2f}")

    print("\n[행당 한계비용 (계산 성분)]")
    for m0, m1 in zip(MS, MS[1:]):
        d = (tot[m1] - tot[m0]) / (m1 - m0)
        print(f"  M {m0}→{m1}: {d:+.3f} ms/행")
    d_5_9 = (tot[9] - tot[5]) / 4
    print(f"\n  체인5행→트리9행 구간 (N_v=8 해당): {d_5_9:+.3f} ms/행 "
          f"→ +4행 = {4 * d_5_9:+.2f} ms")
    print(f"  engine anchor 2.4ms/행 대비: 계산 성분 {d_5_9:.3f} — "
          f"차이는 wire/NCCL/launch 등 (분해 근거)")

    # ---- draft 증분 프로토타입 (eager 구간 ops) ----
    pool_pri = torch.rand(130, device=dev)
    dist32 = torch.rand(10, 32, device=dev)

    def draft_step_ops():
        torch.topk(pool_pri, 10)                       # 선택
        g_ = torch.distributions.Exponential(1.0)      # 비복원 샘플 (race)
        r = dist32 / torch.rand_like(dist32).clamp_min(1e-9)
        torch.topk(r, 3, dim=-1)
        pool_pri.scatter_(0, torch.arange(10, device=dev), torch.rand(10, device=dev))
    t_draft = bench(draft_step_ops) * 4                # F_total=4 forward
    print(f"\n[draft 증분 프로토타입] 선택+비복원샘플+장부 ×4 forward = "
          f"{t_draft:.3f} ms/step (트리 rollout eager 구간; forward 자체는 현행과 동수)")


if __name__ == "__main__":
    main()
