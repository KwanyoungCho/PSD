"""Phase 5 smoke: instantiate SSD LLM with AWQ artifact, run AR decode.

Uses Llama-3.2-1B-Instruct + the RTN artifact written by
scripts/awq_import.py. Target-only quantization, dense lm_head, tp_size=1.

Run (from ssd/):
    python -O sandbox/awq_spike/03_ar_smoke.py
"""
import os

os.environ.setdefault("SSD_HF_CACHE", "/data2/chokwans99/models")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
os.environ.setdefault("SSD_CUDA_ARCH", "8.6")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

MODEL = os.environ.get(
    "SSD_AWQ_TEST_MODEL",
    "/data2/chokwans99/models/Llama-3.2-1B-Instruct",
)
ARTIFACT = os.environ.get("SSD_AWQ_TEST_ARTIFACT", "/tmp/awq_artifacts/llama1b")

import torch

from ssd import LLM, SamplingParams

def main():
    print(f"[smoke] model={MODEL}  artifact={ARTIFACT}")
    llm = LLM(
        model=MODEL,
        num_gpus=1,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.45,
        enforce_eager=False,     # also exercise the CUDA graph path
        target_quant_enabled=True,
        target_quant_backend="awq_marlin",
        target_quant_awq_artifact=ARTIFACT,
    )
    print("[smoke] engine ready")

    sp = SamplingParams(temperature=0.0, max_new_tokens=32)
    prompt = "The capital of France is"
    output = llm.generate([prompt], sp)
    print(f"[smoke] output[0] = {output[0]!r}")
    assert output and output[0], "no tokens generated"
    print("[smoke] OK")


if __name__ == "__main__":
    main()
