"""Read-only source/history and saved-profile audit for MERGE_REVIEW.md.

This does not benchmark or execute a model. Upstream checks inspect a pinned
public SSD checkout; finding the same source pattern is not a runtime proof.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = "a82f7d24fb36827a9a81a3567f344dccb71f193e"
SYSTEMS = "68d0a26d17303a416ad3f27009454ba3ced814be"
RESEARCH = "cc4a3ba11475aeaf6c20069c9a65315f78b0cf1e"
UPSTREAM = "d7eb8fa0edb77a6d0876af1903367b9bb82f54e7"


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def snapshot(ref, path):
    return git("show", f"{ref}:{path}")


def changed(ref):
    return set(git("diff", "--name-only", f"{BASE}..{ref}").splitlines())


def profile_summary(model):
    directory = ROOT / "results/mlsys_coverage/profiles" / f"{model}_profile_fast"
    result = {}
    for path in sorted(directory.glob("duet_profile_*.json")):
        rows = json.loads(path.read_text())
        labels = defaultdict(list)
        steps = defaultdict(dict)
        for row in rows:
            step = row.get("step_id")
            if step is None or step < 10 or "cuda_ms" not in row:
                continue
            labels[row["label"]].append(row["cuda_ms"])
            steps[step][row["label"]] = row["cuda_ms"]
        result[str(path.relative_to(ROOT))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "labels": {k: {"n": len(v), "median_ms": statistics.median(v)}
                       for k, v in labels.items()},
        }
        sums = [s["graph_pre"] + s["graph_post"] for s in steps.values()
                if "graph_pre" in s and "graph_post" in s]
        if sums:
            result[str(path.relative_to(ROOT))]["paired_pre_post_sum"] = {
                "n": len(sums), "median_ms": statistics.median(sums)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("review_audit.json"))
    args = parser.parse_args()
    systems, research = changed(SYSTEMS), changed(RESEARCH)
    data = {
        "scope": "Static code/history inspection and reaggregation of existing profiles; no new GPU run.",
        "base": BASE, "systems": SYSTEMS, "research": RESEARCH,
        "upstream_inspected_sha": UPSTREAM,
        "runtime_overlap": sorted(p for p in systems & research if p.startswith("ssd/ssd/")),
        "commits": git("log", "--reverse", "--format=%h %s", f"{BASE}..{SYSTEMS}").splitlines(),
        "code_file_numstat": git("diff", "--numstat", f"{BASE}..{SYSTEMS}", "--", "ssd").splitlines(),
        "source_patterns": {},
        "profiles": {m: profile_summary(m) for m in ("llama2", "llama3")},
        "profile_scope": "Historical round1 instrumented chain profiles, nominal B8/K1=4/K2=2/exit21; step_id>=10; includes batch shrink. Component medians are not same-step critical paths or final round3 timings.",
        "draft_forest_capture_counts": {},
    }
    patterns = {
        "prefill_cache_flatten": ("ssd/layers/attention.py", ["k, v = k_cache, v_cache"]),
        "unbounded_prefill_admission": ("ssd/engine/scheduler.py", ["while self.waiting:"]),
        "mutable_prompt_boundary": ("ssd/engine/sequence.py", ["return self.num_tokens - self.num_prompt_tokens"]),
        "preemption_moves_boundary": ("ssd/engine/scheduler.py", ["seq.num_prompt_tokens = seq.num_tokens"]),
        "block_hash_last_page": ("ssd/engine/scheduler.py", ["last_block = block_manager.blocks[block_table[-1]]"]),
        "raw_suffix_tps_counter": ("ssd/engine/step.py", ["return sum(len(s) for s in out_verify_result.new_suffixes)"]),
        "ragged_length_clamp": ("ssd/utils/verify.py", ["accept_until = torch.minimum(accept_until, valid_k)"]),
    }
    if args.upstream and git("rev-parse", "HEAD", root=args.upstream) != UPSTREAM:
        raise ValueError("Upstream revision differs; review instead of silently changing evidence")
    for name, (path, strings) in patterns.items():
        entry = {"path": path, "needles": strings}
        for label, ref in (("earliest_local", "f46aecd"), ("paper_base", BASE), ("systems", SYSTEMS)):
            text = snapshot(ref, "ssd/" + path)
            entry[label] = all(s in text for s in strings)
        if args.upstream:
            text = (args.upstream / path).read_text()
            entry["public_upstream"] = all(s in text for s in strings)
            entry["upstream_file_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        data["source_patterns"][name] = entry
    for model in ("llama2", "llama3"):
        path = ROOT / f"results/mlsys_coverage/round3/comparison/{model}_tree_optimized.log"
        match = re.search(r"\[metrics\] batched tree: (\{[^\n]+\})", path.read_text())
        data["draft_forest_capture_counts"][model] = ast.literal_eval(match.group(1))
    data["graph_counter_scope"] = "captures counts draft forest executors only; excludes target/glue/root/input/accept captures and is cumulative over warmup plus two passes."
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "overlap": data["runtime_overlap"],
                      "graph_counts": data["draft_forest_capture_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
