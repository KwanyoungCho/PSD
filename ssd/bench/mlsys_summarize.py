"""Regenerate comparable throughput, batch-padding and profiler summaries."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


def summarize_file(path):
    run = json.loads(path.read_text())
    if run.get("status") != "complete":
        return []
    result = []
    for cell in run["cells"]:
        events = cell["metrics"].get("phase_events", [])
        steps = defaultdict(list)
        for ev in events:
            steps[ev["step_id"]].append(ev)
        needed = sum((ev.get("valid_k") or 0)+1 for ev in events)
        dense = sum((ev.get("verify_width") or 0)+1 for ev in events)
        row = dict(file=str(path), mode=run["args"]["mode"],
                   batch=cell["batch"], temperature=cell["temperature"], seed=cell["seed"],
                   n_prompts=len(run["prompt_ids"]), truncated=run["truncated_prompts"],
                   k1=run["args"]["k1"], k2=run["args"]["k2"],
                   dfo=run["args"]["draft_fan_out"],
                   **cell["summary"],
                   target_peak_allocated_gib=cell["target_peak_allocated_bytes"]/2**30,
                   target_peak_reserved_gib=cell["target_peak_reserved_bytes"]/2**30)
        if events:
            row.update(mean_live_batch=len(events)/len(steps),
                       sequence_padding_fraction=1-needed/dense,
                       any_miss_step_fraction=mean(any(not e["cache_hit"] for e in es) for es in steps.values()),
                       mixed_step_fraction=mean(any(e["cache_hit"] for e in es) and
                                               any(not e["cache_hit"] for e in es) for es in steps.values()))
        result.append(row)
    return result


def summarize_profile(folder):
    summary = {}
    for path in sorted(folder.glob("**/duet_profile_*.json")):
        groups = defaultdict(list)
        for row in json.loads(path.read_text()):
            if row["label"] == "_anchor":
                continue
            groups[row["label"]].append(row)
        summary[str(path)] = {
            label: dict(count=len(rows), gpu_ms_mean=mean(r["cuda_ms"] for r in rows),
                        cpu_dispatch_ms_mean=mean((r["cpu_dispatch_end_ns"]-r["cpu_dispatch_start_ns"])/1e6 for r in rows))
            for label, rows in groups.items()}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    rows = []
    for path in a.runs:
        rows.extend(summarize_file(path))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(rows=rows, profiles=(summarize_profile(a.profiles)
                                            if a.profiles else {})), indent=2))
    with a.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
