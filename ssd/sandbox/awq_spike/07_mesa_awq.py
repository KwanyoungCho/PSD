"""Phase 6: MESA-SSD with AWQ-quantized target.

Topology:
  - target on TP=2 (GPUs 0, 1) — AWQ W4A16
  - draft on GPU 2 (dense Llama-3.2-1B-Instruct)
  - async spec + mesa enabled

Run with 3 GPUs.
"""
import os
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13240")

MODEL = "/data2/chokwans99/models/layerskip-llama3-8B"
DRAFT = "/data2/chokwans99/models/Llama-3.2-1B-Instruct"
ARTIFACT = "/tmp/awq_artifacts/layerskip8b_tp2"

from ssd import LLM, SamplingParams


def main():
    llm = LLM(
        model=MODEL,
        draft=DRAFT,
        num_gpus=3,                  # target tp=2 + draft=1
        speculate=True,
        speculate_k=4,
        draft_async=True,
        async_fan_out=3,
        jit_speculate=True,          # MESA requirement
        mesa_enabled=True,
        mesa_exit_layer=21,
        mesa_draft_fan_out=1,
        max_model_len=512,
        max_num_seqs=1,              # MESA Rev1 constraint
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.4,
        enforce_eager=False,         # MESA requires CUDA graphs
        target_quant_enabled=True,
        target_quant_backend="awq_marlin",
        target_quant_awq_artifact=ARTIFACT,
    )
    sp = SamplingParams(temperature=0.6, max_new_tokens=48)
    out = llm.generate(["The capital of France is"], sp)
    print(f"[mesa-smoke] {out!r}")


if __name__ == "__main__":
    main()
