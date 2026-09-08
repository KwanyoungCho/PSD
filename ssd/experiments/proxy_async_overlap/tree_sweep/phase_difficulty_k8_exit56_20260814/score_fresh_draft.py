#!/usr/bin/env python3
"""Teacher-force TinyLlama on realized prefixes and score phase difficulty.

This is deliberately independent of cached P1/P2 proposals.  At each fully
observed verification event, it reports the next realized token's draft NLL
and rank plus the consecutive greedy agreement length (capped at K=8).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

MODEL = Path("/home/eslab/models/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6")
SOURCES = {0: "Miss (fresh JIT)", 1: "P1", 2: "P2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=/path/to/raw.jsonl")
    ap.add_argument("--output", default="analysis/fresh_draft_events.csv")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True).to(args.device)
    model.eval()
    cache = {}
    records = []
    for spec in args.inputs:
        arm, path = spec.split("=", 1)
        for row in map(json.loads, open(path)):
            prompt = [int(x) for x in row["prompt_token_ids"]]
            output = [int(x) for x in row["output_token_ids"]]
            key = (tuple(prompt), tuple(output))
            if key not in cache:
                ids = torch.tensor([prompt + output], dtype=torch.long,
                                   device=args.device)
                with torch.inference_mode():
                    logits = model(ids).logits[0].float().cpu()
                cache[key] = logits
            logits = cache[key]
            completion_offset = 1
            for index, event in enumerate(row["phase_events"]):
                al = int(event["accepted_len"])
                fully_observed = completion_offset + al <= len(output)
                if fully_observed and completion_offset < len(output):
                    pred_pos = len(prompt) + completion_offset - 1
                    target = output[completion_offset]
                    first = logits[pred_pos]
                    logp = torch.log_softmax(first, dim=-1)[target].item()
                    rank = int((first > first[target]).sum().item()) + 1
                    agree = 0
                    max_k = min(8, len(output) - completion_offset)
                    for j in range(max_k):
                        lp = logits[pred_pos + j]
                        if int(lp.argmax().item()) != output[completion_offset + j]:
                            break
                        agree += 1
                    records.append({
                        "arm": arm, "uid": row["uid"],
                        "question_key": f'{row["group"]}:{row["question_id"]}',
                        "group": row["group"], "seed": int(row.get("seed", -1)),
                        "event_index": index,
                        "source_id": int(event["source"]),
                        "source": SOURCES[int(event["source"])],
                        "cached_accepted_len": al,
                        "valid_k": event.get("valid_k"),
                        "fresh_greedy_al": agree + 1,
                        "fresh_first_token_nll": -logp,
                        "fresh_first_token_rank": rank,
                        "completion_offset": completion_offset,
                    })
                completion_offset += al

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader(); w.writerows(records)
    print(f"wrote {out}: {len(records)} events; {len(cache)} unique trajectories")


if __name__ == "__main__":
    main()
