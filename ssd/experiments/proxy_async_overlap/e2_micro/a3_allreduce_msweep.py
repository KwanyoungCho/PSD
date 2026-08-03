"""Phase A3 — 4-GPU(0-3) NCCL allreduce [M, 8192] fp16의 M 스케일링 실측.

19번 트랙 범인 후보 ③: 70B TP4 verify는 층당 allreduce 2회 × 80층 =
스텝당 160회. PCIe(3090, NVLink 없음) 대역이 병목이면 행(M) 증가가
여기서 비쌀 수 있다.

Run: cd ssd && python experiments/proxy_async_overlap/e2_micro/a3_allreduce_msweep.py
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MS = [1, 5, 9, 13]
H = 8192
ITERS = 200
CALLS_PER_LAYER = 2
LAYERS = 80


def worker(rank, world, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29611"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    out = {}
    for M in MS:
        x = torch.randn(M, H, device=f"cuda:{rank}", dtype=torch.float16)
        for _ in range(30):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        dist.barrier()
        s.record()
        for _ in range(ITERS):
            dist.all_reduce(x)
        e.record()
        torch.cuda.synchronize()
        out[M] = s.elapsed_time(e) / ITERS
    if rank == 0:
        results.update(out)
    dist.destroy_process_group()


def main():
    mgr = mp.Manager()
    results = mgr.dict()
    mp.spawn(worker, args=(4, results), nprocs=4, join=True)
    res = dict(results)
    print(f"{'M':>4} {'1회(µs)':>9} {'스텝(160회, ms)':>15}")
    for M in MS:
        us = res[M] * 1000
        print(f"{M:>4} {us:>9.1f} {res[M] * CALLS_PER_LAYER * LAYERS:>15.2f}")
    print("\n[행당 한계비용 — allreduce 성분]")
    for a, b in ((1, 5), (5, 9), (9, 13), (5, 13)):
        d = (res[b] - res[a]) * CALLS_PER_LAYER * LAYERS / (b - a)
        print(f"  M {a}→{b}: {d:+.3f} ms/행 ({d / LAYERS * 1000:+.1f} µs/layer/행)")


if __name__ == "__main__":
    main()
