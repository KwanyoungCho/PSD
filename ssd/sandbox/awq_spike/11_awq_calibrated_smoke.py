"""Smoke: run SSD with a properly AWQ-calibrated 1B model.

Compares decoded output between:
  - RTN-direct    (no calibration; existing /tmp/awq_artifacts/llama1b)
  - AWQ-calibrated (new: /data2/awq_calibrated/llama3p2_1b + its artifact)

Both use the SAME Marlin runtime kernel; the only difference is the
calibration step that picks scales. So decode SPEED is expected equal;
quality (perplexity, downstream coherence) should be slightly better
for AWQ-calibrated.
"""
import os
os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("SSD_DIST_PORT", "13280")

from ssd import LLM, SamplingParams


def run(model_path, artifact, label):
    llm = LLM(
        model=model_path,
        num_gpus=1,
        max_model_len=512, max_num_seqs=1, max_num_batched_tokens=512,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        target_quant_enabled=True,
        target_quant_backend="awq_marlin",
        target_quant_awq_artifact=artifact,
    )
    sp = SamplingParams(temperature=0.0, max_new_tokens=32)
    out = llm.generate(["The capital of France is"], sp, use_tqdm=False)
    outputs = out[0] if isinstance(out, tuple) else out
    print(f"[{label}] text: {outputs[0]['text']!r}")
    print(f"[{label}] ids : {outputs[0]['token_ids'][:16]}")
    import torch
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "awq"
    if which == "rtn":
        run("/data2/chokwans99/models/Llama-3.2-1B-Instruct",
            "/tmp/awq_artifacts/llama1b", "rtn")
    else:
        run("/data2/chokwans99/awq_calibrated/llama3p2_1b",
            "/data2/chokwans99/awq_artifacts/llama3p2_1b/custom_awq_tp1", "awq-calibrated")
