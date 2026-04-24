"""Verify AutoAWQ-calibrated 1B decodes correctly via our Marlin runtime."""
import os
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13290")

from ssd import LLM, SamplingParams

MODEL = "/data2/chokwans99/awq_calibrated/llama3p2_1b_autoawq"
ART = "/data2/chokwans99/awq_artifacts/llama3p2_1b/autoawq_tp1"

def main():
    llm = LLM(
        model=MODEL,
        num_gpus=1,
        max_model_len=512, max_num_seqs=1, max_num_batched_tokens=512,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        target_quant_enabled=True,
        target_quant_backend="awq_marlin",
        target_quant_awq_artifact=ART,
    )
    sp = SamplingParams(temperature=0.0, max_new_tokens=32)
    out = llm.generate(["The capital of France is"], sp, use_tqdm=False)
    outputs = out[0] if isinstance(out, tuple) else out
    print(f"[autoawq] text: {outputs[0]['text']!r}")

if __name__ == "__main__":
    main()
