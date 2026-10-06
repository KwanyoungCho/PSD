"""Regenerate final evidence from completed, uncontaminated full-corpus runs."""
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent


def stats(values):
    return dict(n=len(values), mean=mean(values),
                sd=stdev(values) if len(values) > 1 else None,
                values=values)


def decode_tps(run, cell):
    n = (sum(len(o["token_ids"]) for o in cell["outputs"])
         if run["args"]["mode"] != "ar" else cell["metrics"]["decode_total_tokens"])
    return n / cell["metrics"]["decode_total_time"]


def main():
    groups = {}
    files = {}
    for folder in ("full_llama2", "full_llama2_references", "full_llama3", "full_llama3_fast"):
        for path in sorted((ROOT/folder).glob("llama*.json")):
            run = json.loads(path.read_text())
            if run.get("status") != "complete":
                continue
            manifest = json.loads((path.parent/"campaign.json").read_text())
            job = next((j for j in manifest if j["name"] == path.stem), None)
            if job is None or job.get("external_pids"):
                continue  # Not yet exited, or contaminated GPU measurements.
            model, arm, seed = path.stem.split("_")
            arm = {"optimized": "width1", "wide": "width4"}.get(arm, arm)
            key = model + "/" + arm
            cell = run["cells"][0]
            groups.setdefault(key, []).append((int(seed[1:]), run, cell, path))
            files[(model, arm, int(seed[1:]))] = run
    summary = {}
    for key, items in groups.items():
        items.sort(key=lambda x:x[0])
        entry = dict(seeds=[x[0] for x in items], files=[str(x[3].relative_to(ROOT)) for x in items],
                     n_prompts=len(items[0][1]["prompt_ids"]),
                     truncated_prompts=items[0][1]["truncated_prompts"],
                     k1=items[0][1]["args"]["k1"], k2=items[0][1]["args"]["k2"],
                     draft_fan_out=items[0][1]["args"]["draft_fan_out"])
        for field in ("decode_tps", "end_to_end_tps", "al_including_recovery", "cache_hit"):
            vals = [decode_tps(x[1], x[2]) if field == "decode_tps" else x[2]["summary"][field] for x in items]
            if all(v is not None for v in vals):
                entry[field] = stats(vals)
        if items[0][2]["metrics"].get("phase_events"):
            pads, misses, live = [], [], []
            for _, _, cell, _ in items:
                evs = cell["metrics"]["phase_events"]
                steps = {}
                for e in evs:
                    steps.setdefault(e["step_id"], []).append(e)
                pads.append(1-sum(e["valid_k"]+1 for e in evs)/sum(e["verify_width"]+1 for e in evs))
                misses.append(mean(any(not e["cache_hit"] for e in es) for es in steps.values()))
                live.append(len(evs)/len(steps))
            entry.update(sequence_padding_fraction=stats(pads), any_miss_step_fraction=stats(misses),
                         mean_live_batch=stats(live))
        summary[key] = entry
    comparisons = {}
    for model in ("llama2", "llama3"):
        for candidate in ("fast", "width1", "width4", "ssd"):
            rows = []
            for seed in (2026, 2027, 2028):
                base = files.get((model, "baseline", seed))
                new = files.get((model, candidate, seed))
                if base is None or new is None:
                    continue
                a, b = base["cells"][0], new["cells"][0]
                assert base["prompt_sha256"] == new["prompt_sha256"]
                assert base["output_caps"] == new["output_caps"]
                rows.append(dict(seed=seed,
                    decode_gain_percent=100*(decode_tps(new,b)/decode_tps(base,a)-1),
                    e2e_gain_percent=100*(b["summary"]["end_to_end_tps"]/a["summary"]["end_to_end_tps"]-1),
                    exact_outputs=sum(x["token_ids"] == y["token_ids"] for x,y in zip(a["outputs"], b["outputs"])),
                    output_count=len(a["outputs"])))
            if rows:
                comparisons[model+"/"+candidate] = dict(rows=rows,
                    decode_gain_percent=stats([r["decode_gain_percent"] for r in rows]))
    (ROOT/"FINAL_NUMBERS.json").write_text(json.dumps(dict(groups=summary, comparisons=comparisons), indent=2))
    greedy_files = [f"full_greedy/{m}_{a}.json" for m in ("llama2", "llama3")
                    for a in ("reference", "specialized")]
    greedy_files += ["closing/llama2_k62.json", "greedy_repeat/llama2_k42_repeat.json",
                     "greedy_repeat/llama2_k62_repeat.json"]
    greedy = {}
    for name in greedy_files:
        path = ROOT/name
        run = json.loads(path.read_text())
        job = next(j for j in json.loads((path.parent/"campaign.json").read_text())
                   if j["name"] == path.stem)
        assert run["status"] == "complete" and not job["external_pids"]
        cell = run["cells"][0]
        greedy[name] = dict(decode_tps=decode_tps(run, cell),
            end_to_end_tps=cell["summary"]["end_to_end_tps"],
            al_including_recovery=cell["summary"]["al_including_recovery"],
            n_prompts=len(run["prompt_ids"]), k1=run["args"]["k1"],
            gpu_uuids=job["gpu_uuids"])
    pairs = [(f"full_greedy/{m}_reference.json", f"full_greedy/{m}_specialized.json")
             for m in ("llama2", "llama3")]
    pairs += [("full_greedy/llama2_specialized.json", "closing/llama2_k62.json"),
              ("greedy_repeat/llama2_k42_repeat.json", "greedy_repeat/llama2_k62_repeat.json")]
    greedy_comparisons = []
    for a, b in pairs:
        ar, br = (json.loads((ROOT/p).read_text()) for p in (a, b))
        assert ar["prompt_sha256"] == br["prompt_sha256"]
        assert greedy[a]["gpu_uuids"] == greedy[b]["gpu_uuids"]
        ao, bo = ar["cells"][0]["outputs"], br["cells"][0]["outputs"]
        greedy_comparisons.append(dict(reference=a, candidate=b,
            decode_gain_percent=100*(greedy[b]["decode_tps"]/greedy[a]["decode_tps"]-1),
            exact_outputs=sum(x["token_ids"] == y["token_ids"] for x,y in zip(ao,bo)),
            output_count=len(ao)))
    (ROOT/"GREEDY_NUMBERS.json").write_text(json.dumps(
        dict(runs=greedy, comparisons=greedy_comparisons), indent=2))
    for key, entry in sorted(summary.items()):
        t = entry["decode_tps"]
        print(f"{key:18s} n={t['n']} TPS={t['mean']:.2f} sd={t['sd']} E2E={entry['end_to_end_tps']['mean']:.2f}")


if __name__ == "__main__":
    main()
