"""Inspect the first greedy divergence using independently evaluated HF logits.

This diagnoses near ties; it does not certify that a kernel mismatch is harmless.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar", type=Path, required=True)
    parser.add_argument("--duet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ar, duet = json.loads(args.ar.read_text()), json.loads(args.duet.read_text())
    if ar["prompt_sha256"] != duet["prompt_sha256"]:
        raise ValueError("Prompt mismatch")
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        ar["args"]["target"], dtype="auto", device_map="cuda",
        attn_implementation="sdpa").eval()
    rows = []
    for cell in duet["cells"]:
        if cell["temperature"] != 0:
            continue
        ref = next(c for c in ar["cells"] if c["batch"] == cell["batch"] and c["temperature"] == 0)
        for i, (a, d) in enumerate(zip(ref["outputs"], cell["outputs"])):
            if a["token_ids"] == d["token_ids"]:
                continue
            pos = next((j for j, (u, v) in enumerate(zip(a["token_ids"], d["token_ids"])) if u != v), None)
            if pos is None:
                rows.append(dict(batch=cell["batch"], question=i, reason="length_only"))
                continue
            prefix = ar["prompt_ids"][i] + a["token_ids"][:pos]
            with torch.inference_mode():
                logits = model(torch.tensor([prefix], device="cuda"), use_cache=False).logits[0, -1].float()
            scores, ids = logits.topk(5)
            row = dict(batch=cell["batch"], question=i, position=pos,
                       ar_token=a["token_ids"][pos], duet_token=d["token_ids"][pos],
                       hf_top_ids=ids.tolist(), hf_top_logits=scores.tolist(),
                       ar_logit=float(logits[a["token_ids"][pos]]),
                       duet_logit=float(logits[d["token_ids"][pos]]))
            rows.append(row)
            print(row, flush=True)
    args.output.write_text(json.dumps(dict(dtype=str(model.dtype), divergences=rows), indent=2))


if __name__ == "__main__":
    main()
