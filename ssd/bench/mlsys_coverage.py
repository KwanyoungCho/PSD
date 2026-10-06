"""Reproducible dense-model/batch/temperature validation (one engine per run).

Run with python -O. Save complete token IDs for greedy AR comparisons. This
is an infrastructure validation harness, not a tuned paper performance claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

# A shared venv may have an editable installation pointing at another
# worktree. Resolve this checkout first, also in spawned draft/TP workers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--draft")
    p.add_argument("--mode", choices=("ar", "sd", "ssd", "duet-chain", "duet-tree"), required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    p.add_argument("--temperatures", type=float, nargs="+", default=[0., .7])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--target-tp", type=int, default=1)
    p.add_argument("--prompts", type=Path, required=True,
                   help="JSON list of strings or question objects with turns; first turns only")
    p.add_argument("--limit", type=int, default=16, help="Evenly spaced subset; 0 = all questions")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--input-cap", type=int, default=512)
    p.add_argument("--k1", type=int, default=4)
    p.add_argument("--k2", type=int, default=2)
    p.add_argument("--draft-fan-out", type=int, default=2)
    p.add_argument("--proxy-fan-out", type=int, default=1)
    p.add_argument("--p1-tree", action="store_true")
    p.add_argument("--exit-layer", type=int, default=21)
    p.add_argument("--memory-fraction", type=float, default=.45)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--ragged-limits", action="store_true",
                   help="Vary output caps to exercise batch shrink/refill")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(metrics, outputs, wall):
    events = metrics.get("phase_events", [])
    total = sum(len(o["token_ids"]) for o in outputs)
    return dict(
        output_tokens=total, wall_s=wall, end_to_end_tps=total / wall,
        decode_tps=(metrics["decode_total_tokens"] / metrics["decode_total_time"]
                    if metrics["decode_total_time"] else None),
        al_including_recovery=mean(metrics["accepted_suffix_lens_with_recovery"]),
        accepted_draft_tokens=mean([e["accepted_spec_len"] for e in events]),
        # Sequence-event weighting, not an average of variable-size batches.
        cache_hit=mean([e["cache_hit"] for e in events]),
        phase1_hit=mean([int(e["source"] == 1) for e in events]),
        phase2_hit=mean([int(e["source"] == 2) for e in events]),
        tree_verify_events=sum(bool(e.get("tree")) for e in events),
        verify_events=len(events))


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if min(args.batches) < 1 or args.max_new_tokens < 1:
        raise ValueError("Batch sizes and output length must be positive")
    if args.mode != "ar" and not args.draft:
        raise ValueError("--draft is required for speculative modes")
    os.environ.setdefault("SSD_HF_CACHE", str(Path(args.target).parent))
    os.environ.setdefault("SSD_DATASET_DIR", str(args.prompts.parent))
    # Set SSD_CUDA_ARCH externally for the machine; no fixed GPU IDs here.
    import torch
    from transformers import AutoTokenizer
    from ssd import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.target)
    source = json.loads(args.prompts.read_text())
    if not isinstance(source, list) or not source:
        raise ValueError("--prompts must contain a nonempty JSON list")
    indexes = list(range(len(source)))
    if 0 < args.limit < len(source):
        indexes = [i * len(source) // args.limit for i in range(args.limit)]
    texts = [source[i] if isinstance(source[i], str) else source[i]["turns"][0]
             for i in indexes]
    encoded = [tok.encode(s) for s in texts]
    prompts = [ids[:args.input_cap] for ids in encoded]
    caps = [max(1, args.max_new_tokens - (i % 4) * (args.max_new_tokens // 5))
            if args.ragged_limits else args.max_new_tokens for i in indexes]
    async_mode = args.mode in ("ssd", "duet-chain", "duet-tree")
    duet = args.mode.startswith("duet")
    kwargs = dict(
        num_gpus=args.target_tp + int(async_mode),
        max_num_seqs=max(args.batches), max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.batches) * args.max_model_len,
        gpu_memory_utilization=args.memory_fraction,
        speculate=args.mode != "ar", draft_async=async_mode,
        jit_speculate=True, speculate_k=args.k1 + args.k2 if duet else args.k1,
        async_fan_out=args.draft_fan_out + args.proxy_fan_out,
    )
    if args.draft:
        kwargs["draft"] = args.draft
    if duet:
        kwargs.update(
            duet_enabled=True, duet_exit_layer=args.exit_layer,
            duet_phase1_k=args.k1, duet_phase2_k=args.k2,
            duet_draft_fan_out=args.draft_fan_out,
            duet_p1_tree_policy="on" if args.p1_tree else "off",
            duet_p2_tree_policy="on" if args.mode == "duet-tree" else "off",
            duet_p2_tree_max_nodes=args.k2 * 2,
            duet_p2_tree_verify_nodes=args.k2 * 2,  # G=M, no pruning
            duet_p1_tree_max_nodes=args.k1 * 2,
            duet_p1_tree_verify_nodes=args.k1 * 2,
        )
    if args.mode == "duet-tree" and (args.batches != [1] or 0 in args.temperatures):
        raise ValueError("Dynamic tree validation currently requires B=1 and T>0; use duet-chain for B>1/T=0")
    env = {k: v for k, v in os.environ.items()
           if k.startswith("SSD_") or k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
    report = dict(
        status="running", args={k: str(v) if isinstance(v, Path) else v
                                for k, v in vars(args).items()},
        engine_kwargs=kwargs, env=env, torch=torch.__version__,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
        ssd_module_path=str(Path(sys.modules["ssd"].__file__).resolve()),
        source_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        prompt_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        question_indexes=indexes, prompt_ids=prompts, output_caps=caps,
        truncated_prompts=sum(len(a) != len(b) for a, b in zip(encoded, prompts)),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        cells=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.output)

    save()
    llm = LLM(args.target, **kwargs)
    report["target_dtype"] = str(llm.config.hf_config.torch_dtype)
    report["draft_dtype"] = str(llm.config.draft_hf_config.torch_dtype) if args.mode != "ar" else None
    try:
        for temperature in args.temperatures:
            for batch in args.batches:
                # Captured capacity stays fixed; only the scheduler limit changes.
                llm.scheduler.max_num_seqs = batch
                warm = SamplingParams(temperature=temperature, max_new_tokens=16,
                                      ignore_eos=args.ignore_eos)
                llm.generate(prompts[:batch], warm, use_tqdm=False)
                for seed in args.seeds:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    # SSD_SEED seeds draft process at startup. Each cell records
                    # that separately: resetting rank0 is NOT a draft RNG reset.
                    params = [SamplingParams(temperature=temperature,
                              max_new_tokens=n, ignore_eos=args.ignore_eos) for n in caps]
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    outputs, metrics = llm.generate(prompts, params, use_tqdm=False)
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                    if len(outputs) != len(prompts):
                        raise RuntimeError("Incomplete generation")
                    for out, cap in zip(outputs, caps):
                        ids = out["token_ids"]
                        if not ids or len(ids) > cap:
                            raise RuntimeError(f"Invalid output length {len(ids)} > {cap}")
                        if len(ids) < cap and ids[-1] != tok.eos_token_id:
                            raise RuntimeError("Early stop without EOS: increase the context window")
                    cell = dict(batch=batch, temperature=temperature, seed=seed,
                                outputs=outputs, metrics=json.loads(json.dumps(metrics)),
                                target_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                target_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                summary=summarize(metrics, outputs, wall))
                    report["cells"].append(cell)
                    save()
                    print("COVERAGE_CELL", json.dumps({k: cell[k] for k in
                          ("batch", "temperature", "seed", "summary")}), flush=True)
        report["status"] = "complete"
        save()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        llm.exit(hard=False)


if __name__ == "__main__":
    main()
