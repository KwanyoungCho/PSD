"""Read-only reanalysis of the common-fix review; no new model executions."""
import hashlib
import json
import statistics
import subprocess
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUTS = {}


def read(relative):
    path = ROOT / relative
    if path.exists():
        data = path.read_bytes()
    else:
        with tarfile.open(ROOT / "RAW_RESULTS.tar.gz") as archive:
            data = archive.extractfile(relative).read()
    INPUTS[relative] = hashlib.sha256(data).hexdigest()
    return data.decode()


def included(run):
    return [s for s in run["steps"]
            if s["batch"] == run["args"]["batch"]
            and not (s["cap"] or s["clipped"])]


def main():
    runs = {arm: json.loads(read(f"tree_r1/{arm}.json"))
            for arm in ("before", "after")}
    rows = {arm: included(run) for arm, run in runs.items()}
    key = lambda s: (s["emitted"], [(e["source"], e["cache_hit"], e["accepted_len"])
                                   for e in s["events"]])
    assert runs["before"]["prompt_ids"] == runs["after"]["prompt_ids"]
    assert runs["before"]["outputs"] == runs["after"]["outputs"]
    assert [key(s) for s in rows["before"]] == [key(s) for s in rows["after"]]
    source_stats = {}
    for source in (None, 0, 1, 2):
        metrics = {}
        for metric in ("step", "verify", "nonverify"):
            values = {}
            for arm, steps in rows.items():
                chosen = [s for s in steps if source is None or s["events"][0]["source"] == source]
                values[arm] = np.array([
                    s["seconds"] if metric == "step" else
                    sum(s["verify"]) if metric == "verify" else
                    s["seconds"] - sum(s["verify"]) for s in chosen]) * 1000
            a, b = values["before"], values["after"]
            metrics[metric] = dict(
                steps=len(a), before_mean_ms=float(a.mean()), after_mean_ms=float(b.mean()),
                mean_delta_ms=float((b-a).mean()),
                quantiles=[0.1, 0.5, 0.9, 0.99],
                marginal_quantile_delta_ms=(np.quantile(b, [.1, .5, .9, .99])
                                           - np.quantile(a, [.1, .5, .9, .99])).tolist())
        source_stats["all" if source is None else str(source)] = metrics
    clocks = {}
    populations = {}
    for arm, run in runs.items():
        samples = [json.loads(line) for line in read(f"tree_r1/{arm}.telemetry.jsonl").splitlines()]
        gpu = []
        for sample in samples:
            for line in sample["gpu"].splitlines():
                index, temp, power, clock, util, mem = [float(s.strip().split()[0]) for s in line.split(",")]
                if index == 4:
                    gpu.append((clock, util))
        clocks[arm] = dict(
            target_gpu=4, all_sample_mean_mhz=statistics.mean(c for c, u in gpu),
            util_at_least_50pct_mean_mhz=statistics.mean(c for c, u in gpu if u >= 50),
            note="Utilization filter is a coarse activity proxy, not a decode-kernel-aligned clock trace.")
        steps = run["steps"]
        populations[arm] = dict(
            prompts=len(run["outputs"]), all_steps=len(steps), included_steps=len(rows[arm]),
            cap_steps=sum(s["cap"] for s in steps), clipped_steps=sum(s["clipped"] for s in steps),
            all_decode_emitted_tps=sum(s["emitted"] for s in steps)/sum(s["seconds"] for s in steps),
            filtered_tps=run["summary"]["tps"])
    doc = "ssd/docs/duet/TREE_IMPLEMENTATION.md"
    paper = subprocess.check_output(["git", "show", f"a82f7d2:{doc}"], cwd=ROOT, text=True)
    evidence = [dict(line=i, text=line) for i, line in enumerate(paper.splitlines(), 1)
                if "N1=14, M1=12" in line]
    assert evidence
    out = dict(
        scope="Existing raw data reanalysis only; no new GPU runs or regression-suite reruns.",
        paired_outputs_and_source_acceptance_equal=True,
        source_timing=source_stats, target_clocks=clocks, populations=populations,
        paper_g_gt_m=dict(commit="a82f7d2", path=doc, evidence=evidence),
        input_sha256=INPUTS,
        conclusion="Uniform shift does not identify cause; active clocks are essentially equal. G>M AL/TPS comparison remains pending.")
    (ROOT / "REVIEW_AUDIT.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"source_step_delta_ms": {s: v["step"]["mean_delta_ms"] for s, v in source_stats.items()},
                      "target_clocks": clocks, "populations": populations}, indent=2))


if __name__ == "__main__":
    main()
