"""Quick smoke test for SSD engine with LayerSkip 8B + Llama 1B."""
import os, sys
os.environ["SSD_CUDA_ARCH"] = "8.6"
os.environ["SSD_HF_CACHE"] = "/data2/chokwans99/models"
os.environ["SSD_DATASET_DIR"] = "/tmp"

import ssd.paths  # noqa
from ssd import LLM, SamplingParams

TARGET = "/data2/chokwans99/models/layerskip-llama3-8B"
DRAFT = "/data2/chokwans99/models/Llama-3.2-1B-Instruct"


def main():
    prompt = "The capital of France is"
    sp = SamplingParams(temperature=0, max_new_tokens=64)

    mode = sys.argv[1] if len(sys.argv) > 1 else "ar"

    if mode == "ar":
        print("=" * 60)
        print("TEST: Autoregressive (1 GPU, no speculation)")
        print("=" * 60)
        llm = LLM(model=TARGET, num_gpus=1, max_num_seqs=1,
                  max_model_len=512, enforce_eager=True)
        outputs = llm.generate([prompt], sp)

    elif mode == "async":
        print("=" * 60)
        print("TEST: Async SSD (2 GPUs, k=4, f=2)")
        print("=" * 60)
        llm = LLM(model=TARGET, draft=DRAFT, num_gpus=2,
                  speculate=True, speculate_k=4, draft_async=True,
                  async_fan_out=2, max_num_seqs=1,
                  max_model_len=512, enforce_eager=True)
        outputs = llm.generate([prompt], sp)

    for o in outputs:
        print(f"\nPrompt: {prompt}")
        if isinstance(o, dict):
            print(f"Text: {o.get('text', o)}")
        else:
            print(f"Output: {o}")

    print(f"\n{mode.upper()} test passed!")


if __name__ == "__main__":
    main()
