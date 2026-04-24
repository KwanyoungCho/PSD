"""Phase 5 TP=2 AR smoke with LayerSkip-Llama3-8B + AWQ artifact."""
import os

os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13210")

MODEL = "/data2/chokwans99/models/layerskip-llama3-8B"
ARTIFACT = "/tmp/awq_artifacts/layerskip8b_tp2"

from ssd import LLM, SamplingParams


def main():
    print(f"[smoke] tp=2 model={MODEL}  artifact={ARTIFACT}")
    llm = LLM(
        model=MODEL,
        num_gpus=2,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.45,
        enforce_eager=False,
        target_quant_enabled=True,
        target_quant_backend="awq_marlin",
        target_quant_awq_artifact=ARTIFACT,
    )
    print("[smoke] engine ready")
    sp = SamplingParams(temperature=0.0, max_new_tokens=48)
    out = llm.generate(["The capital of France is"], sp)
    print(f"[smoke] output = {out[0]!r}")


if __name__ == "__main__":
    main()
