"""Test DUET-SSD with Llama2 models: LayerSkip-Llama2-13B + TinyLlama-1.1B"""
import os, sys
os.environ["SSD_CUDA_ARCH"] = "8.6"
os.environ["SSD_HF_CACHE"] = "/data2/chokwans99/models"
os.environ["SSD_DATASET_DIR"] = "/tmp"

import ssd.paths
from ssd import LLM, SamplingParams

TARGET = "/data2/chokwans99/models/layerskip-llama2-13B"
DRAFT = "/data2/chokwans99/models/TinyLlama-1.1B-Chat-v1.0"

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    prompt = "The capital of France is"
    sp = SamplingParams(temperature=0.6, max_new_tokens=64)

    kwargs = dict(
        model=TARGET, draft=DRAFT, num_gpus=3,
        speculate=True, speculate_k=4, draft_async=True,
        async_fan_out=3, max_num_seqs=1, max_model_len=512,
    )

    if mode == "duet":
        kwargs["duet_enabled"] = True
        kwargs["duet_exit_layer"] = 26  # 40 * 2/3 ≈ 26
        print("MODE: DUET-SSD")
    else:
        print("MODE: Baseline SSD")

    llm = LLM(**kwargs)
    outputs = llm.generate([prompt], sp)
    for o in outputs:
        if isinstance(o, dict):
            print(f"Text: {o.get('text', '')[:200]}")
        else:
            print(f"Output: {str(o)[:200]}")
    print(f"\n{mode.upper()} test PASSED!")

if __name__ == "__main__":
    main()
