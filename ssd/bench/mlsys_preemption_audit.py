"""Independently check the first output token after re-prefill with HF logits."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run = json.loads(a.run.read_text())
    if run["status"] != "complete":
        raise ValueError("Run incomplete")
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(run["args"]["target"],
        dtype="auto", device_map="cuda", attn_implementation="sdpa").eval()
    records = []
    for cell in run["cells"]:
        if cell["temperature"] != 0:
            raise ValueError("Greedy audit only")
        seen = set()
        for ev in cell["preemption_events"]:
            index, pos = ev["question"], ev["output_position"]
            if (index,pos) in seen or pos == 0:
                continue
            seen.add((index,pos))
            output = cell["outputs"][index]["token_ids"]
            prefix = run["prompt_ids"][index] + output[:pos]
            if prefix != ev["prefix_ids"]:
                raise AssertionError("Previously emitted output was lost or changed on re-prefill")
            with torch.inference_mode():
                logits = model(torch.tensor([prefix], device="cuda"), use_cache=False).logits[0,-1].float()
            values, ids = logits.topk(5)
            chosen = output[pos]
            row = dict(question=index, output_position=pos, token=chosen,
                       hf_top_ids=ids.tolist(), hf_top_logits=values.tolist(),
                       selected_logit=float(logits[chosen]), gap=float(values[0]-logits[chosen]))
            records.append(row)
    result = dict(run=str(a.run), dtype=str(model.dtype), checked=len(records),
                  max_logit_gap=max((r["gap"] for r in records), default=None),
                  all_max_logit=bool(records) and all(r["gap"] == 0 for r in records),
                  all_in_top5=all(r["token"] in r["hf_top_ids"] for r in records), records=records)
    a.output.write_text(json.dumps(result, indent=2))
    print({k:v for k,v in result.items() if k != "records"})


if __name__ == "__main__":
    main()
