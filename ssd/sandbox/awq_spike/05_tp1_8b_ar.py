"""TP=1 AR on layerskip-llama3-8B to isolate TP bug vs RTN quality."""
import os
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13220")

MODEL = "/data2/chokwans99/models/layerskip-llama3-8B"
ARTIFACT = "/tmp/awq_artifacts/layerskip8b_tp1"

from ssd import LLM, SamplingParams


def main():
    llm = LLM(
        model=MODEL,
        num_gpus=1,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.45,
        enforce_eager=False,
        target_quant_enabled=True,
        target_quant_backend="awq_marlin",
        target_quant_awq_artifact=ARTIFACT,
    )
    sp = SamplingParams(temperature=0.0, max_new_tokens=48)
    out = llm.generate(["The capital of France is"], sp)
    print(f"[tp1-quant] {out[0]!r}")


if __name__ == "__main__":
    main()
