"""Phase 5: sync-spec decode with AWQ target + dense 1B draft.

Topology: target TP=2 (2 GPUs) + sync draft same 2 GPUs (sync spec doesn't
need a third GPU; draft runs on rank 0's worker between verifies).

Run with 2 GPUs visible.
"""
import os
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13230")

MODEL = "/data2/chokwans99/models/layerskip-llama3-8B"
DRAFT = "/data2/chokwans99/models/Llama-3.2-1B-Instruct"
ARTIFACT = "/tmp/awq_artifacts/layerskip8b_tp2"

from ssd import LLM, SamplingParams


def main():
    llm = LLM(
        model=MODEL,
        draft=DRAFT,
        num_gpus=2,
        speculate=True,
        speculate_k=4,
        draft_async=False,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.4,
        enforce_eager=False,
        target_quant_enabled=True,
        target_quant_backend="awq_marlin",
        target_quant_awq_artifact=ARTIFACT,
    )
    sp = SamplingParams(temperature=0.0, max_new_tokens=48)
    out = llm.generate(["The capital of France is"], sp)
    print(f"[spec-smoke] {out[0]['text']!r}")


if __name__ == "__main__":
    main()
