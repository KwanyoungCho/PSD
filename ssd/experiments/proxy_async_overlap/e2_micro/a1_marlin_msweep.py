"""Phase A1 — Marlin(AWQ W4A16) 커널의 M(행수) 스케일링 실측.

19번 트랙: verify 행당 22µs/layer/행의 범인 후보 ①. **실제 70B
rank-shard artifact**의 Marlin state로 측정 (합성 shape 아님).

Run: cd ssd && CUDA_VISIBLE_DEVICES=0 python \\
     experiments/proxy_async_overlap/e2_micro/a1_marlin_msweep.py
"""
import torch

from ssd.quant.io import load_awq_artifact
from ssd.quant.marlin import awq_matmul
from ssd.quant.build import RawAwqTensors, build_awq_state

PREFIX = "/data2/chokwans99/awq_artifacts/layerskip_llama2_70b/autoawq_tp4"
MS = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 16]
ITERS = 50


def bench(fn, iters=ITERS, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    art = load_awq_artifact(PREFIX, tp_rank=0, tp_size=4)
    raw_states = next(art[k] for k in ("states", "modules", "layers")
                      if isinstance(art.get(k), dict))
    dev = torch.device("cuda")
    picks = {}
    for tag in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        name = f"model.layers.0.self_attn.{tag}" \
            if "proj" in tag and tag in ("qkv_proj", "o_proj") \
            else f"model.layers.0.mlp.{tag}"
        d = raw_states[name]
        raw = RawAwqTensors(
            qweight=d["qweight"], qzeros=d["qzeros"], scales=d["scales"],
            in_features=d["in_features"], out_features=d["out_features"],
            group_size=d["group_size"], bias=d.get("bias"))
        picks[tag] = (name, build_awq_state(raw, dev))
    print("[picked]", {k: (v[1].in_features, v[1].out_features)
                       for k, v in picks.items()})
    print(f"\n{'M':>4} " + " ".join(f"{t:>12}" for t in picks) +
          f" {'층합(ms)':>10} {'80층(ms)':>10}")
    tot = {}
    for M in MS:
        row, layer_ms = [], 0.0
        for tag, (name, stt) in picks.items():
            x = torch.randn(M, stt.in_features, device=dev,
                            dtype=torch.float16)
            t = bench(lambda: awq_matmul(x, stt))
            row.append(f"{t * 1000:>12.1f}")      # µs
            layer_ms += t
        tot[M] = layer_ms * 80
        print(f"{M:>4} " + " ".join(row) +
              f" {layer_ms:>10.3f} {tot[M]:>10.2f}")

    print("\n[행당 한계비용 — Marlin GEMM 성분, 80층 기준]")
    for a, b in ((1, 5), (5, 9), (9, 13), (5, 13)):
        d = (tot[b] - tot[a]) / (b - a)
        print(f"  M {a}→{b}: {d:+.3f} ms/행  ({d / 80 * 1000:+.1f} µs/layer/행)")
    print("\n비교 기준: 엔진 실측 graph 성분 21.5~22.3 µs/layer/행 (19번 §분해)")


if __name__ == "__main__":
    main()
