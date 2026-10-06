"""Conservative post-hoc exposure audit for the fixed block-hash bug.

A zero result is useful for these archived runs, not a general cache proof.
Token-chunk matches deliberately ignore chained prefix hashes and execution order,
so possible_prompt_reuse overestimates reuse opportunities. Full model runs here
use 256-token pages. Warmup generates at most 16 tokens before metrics reset.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    for folder in ("full_llama2", "full_llama2_references", "full_llama3", "full_llama3_fast", "full_greedy"):
        for path in sorted((ROOT/folder).glob("llama*.json")):
            run = json.loads(path.read_text())
            assert run["status"] == "complete"
            prompts = run["prompt_ids"]
            chunks = {tuple(ids[j:j+256]) for ids in prompts for j in range(0, len(ids)-255, 256)}
            cells = []
            for cell in run["cells"]:
                generated = []
                for ids, output in zip(prompts, cell["outputs"]):
                    joined = ids + output["token_ids"]
                    generated.extend(tuple(joined[j:j+256])
                        for j in range((len(ids)//256)*256, (len(joined)//256)*256, 256))
                excess = cell["metrics"]["prefill_total_tokens"] - sum(map(len,prompts))
                if run["args"]["mode"] == "ar":
                    excess -= len(prompts)  # AR prefill emits the first token.
                cells.append(dict(generated_completed_blocks=len(generated),
                    possible_prompt_reuse=sum(chunk in chunks for chunk in generated),
                    warm_possible_crossings=sum((len(p)+16)//256-len(p)//256 for p in prompts[:cell["batch"]]),
                    reprefill_token_excess=excess))
            rows.append(dict(file=str(path.relative_to(ROOT.parent.parent)), cells=cells))
    (ROOT/"block_hash_exposure_audit.json").write_text(json.dumps(rows, indent=2))
    print("Audited", len(rows), "runs")
    assert all(c["possible_prompt_reuse"] == c["warm_possible_crossings"] == c["reprefill_token_excess"] == 0
               for r in rows for c in r["cells"]), "Exposure found; inspect before using results"


if __name__ == "__main__":
    main()
