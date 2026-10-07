"""Reproducible dense-model/batch/temperature validation (one engine per run).

Run with python -O. Save complete token IDs for greedy AR comparisons. This
is an infrastructure validation harness, not a tuned paper performance claim.
"""
import argparse
from collections import Counter
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
    p.add_argument("--p1-verify-nodes", type=int)
    p.add_argument("--p2-verify-nodes", type=int)
    p.add_argument("--exit-layer", type=int, default=21)
    p.add_argument("--p1-nodes", type=int)
    p.add_argument("--p2-nodes", type=int)
    p.add_argument("--tree-width", type=int, default=3)
    p.add_argument("--p1-roots", type=int, default=2)
    p.add_argument("--p2-budget", type=int)
    p.add_argument("--tree-root-count", type=int)
    p.add_argument("--tree-beta", type=float, default=.5)
    p.add_argument("--tree-proxy-threshold", type=float, default=.01)
    p.add_argument("--tree-conf-threshold", type=float, default=.03)
    p.add_argument("--root-source", choices=("residual","proxy","complement"), default="residual")
    p.add_argument("--root-normalization", choices=("topm","full"), default="topm")
    p.add_argument("--root-overlap-mix", type=float, default=0.)
    p.add_argument("--memory-fraction", type=float, default=.45)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--greedy-only", action="store_true", help="Specialize all captured samplers for T=0")
    p.add_argument("--audit-preemption", action="store_true", help="Save CPU prefix snapshots for independent HF audit (diagnostic runs only)")
    p.add_argument("--ragged-limits", action="store_true",
                   help="Vary output caps to exercise batch shrink/refill")
    p.add_argument("--tree-calibration-dir", type=Path, help="Diagnostic B1 legacy tree snapshots")
    p.add_argument("--preflight-test", help="Optional unittest module before model allocation")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(metrics, outputs, wall, speculative=False):
    events = metrics.get("phase_events", [])
    total = sum(len(o["token_ids"]) for o in outputs)
    # Speculative prefill defers its recovery token until the first decode.
    # Thus all returned output tokens belong to decode. AR emits its first
    # token during prefill and has a correct per-step decode counter.
    decode_tokens = total if speculative else metrics["decode_total_tokens"]
    clean = [e for e in events if not e.get("output_cap_reached", False) and not e.get("clipped", False)]
    steps = metrics.get("decode_steps", [])
    clean_steps = [s for s in steps if not s['output_cap_reached'] and not s['clipped']]
    clean_time = sum(s['seconds'] for s in clean_steps)
    return dict(
        boundary_excluded_al=mean([e['accepted_len'] for e in clean]),
        emitted_al=mean([e.get('emitted_len', e['accepted_len']) for e in events]),
        boundary_excluded_step_tps=(sum(s['emitted_tokens'] for s in clean_steps)/clean_time if clean_time else None),
        boundary_excluded_events=len(events)-len(clean),
        boundary_excluded_steps=len(steps)-len(clean_steps),
        output_tokens=total, wall_s=wall, end_to_end_tps=total / wall,
        decode_output_tokens=decode_tokens,
        decode_tps=(decode_tokens / metrics["decode_total_time"]
                    if metrics["decode_total_time"] else None),
        al_including_recovery=mean(metrics["accepted_suffix_lens_with_recovery"]),
        accepted_draft_tokens=mean([e["accepted_spec_len"] for e in events]),
        # Sequence-event weighting, not an average of variable-size batches.
        cache_hit=mean([e["cache_hit"] for e in events]),
        phase1_hit=mean([int(e["source"] == 1) for e in events]),
        phase2_hit=mean([int(e["source"] == 2) for e in events]),
        observed_miss_valid_k=dict(Counter(e["valid_k"] for e in events if not e["cache_hit"])),
        tree_verify_events=sum(bool(e.get("tree")) for e in events),
        verify_events=len(events))


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if min(args.batches) < 1 or args.max_new_tokens < 1:
        raise ValueError("Batch sizes and output length must be positive")
    if min(args.temperatures) < 0:
        raise ValueError("Temperatures must be nonnegative")
    if args.greedy_only and any(t != 0 for t in args.temperatures):
        raise ValueError("--greedy-only requires all --temperatures to be 0")
    if args.p1_tree and args.mode != "duet-tree":
        raise ValueError("--p1-tree requires --mode duet-tree to label execution correctly")
    if ('SSD_DUET_MISS_K' in os.environ or int(os.getenv('SSD_DUET_MISS_WIDTH','1'))!=1):
        if args.mode!='duet-tree' or (max(args.batches)==1 and os.getenv('SSD_BATCHED_TREE','0')!='1'):
            raise ValueError('Independent miss depth/width requires the unified duet-tree service; use SSD_BATCHED_TREE=1 for B1')
    if args.mode != "ar" and not args.draft:
        raise ValueError("--draft is required for speculative modes")
    os.environ.setdefault("SSD_HF_CACHE", str(Path(args.target).parent))
    os.environ.setdefault("SSD_DATASET_DIR", str(args.prompts.parent))
    # Set SSD_CUDA_ARCH externally for the machine; no fixed GPU IDs here.
    import torch
    from transformers import AutoTokenizer
    from ssd import LLM, SamplingParams

    if args.preflight_test:
        import unittest
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        suite=unittest.defaultTestLoader.loadTestsFromNames(args.preflight_test.split(","))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise RuntimeError('Preflight tests failed')
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        # Do not retain test graphs/tensors when measuring the engine's KV
        # capacity. First-pass compile caches may still be warm; later passes
        # are reported separately by the experiment analysis.
        import gc
        del suite
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    observer=None
    if args.tree_calibration_dir:
        if args.batches != [1] or args.mode!='duet-tree' or os.getenv('SSD_BATCHED_TREE','0')!='0':
            raise ValueError('Tree calibration observer requires legacy B1 tree')
        from mlsys_tree_observer import TreeObserver
        observer=TreeObserver(args.tree_calibration_dir)

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
        greedy_only=args.greedy_only,
        jit_speculate=True, speculate_k=args.k1 + args.k2 if duet else args.k1,
        async_fan_out=args.draft_fan_out + args.proxy_fan_out,
        duet_p1_tree_policy="off", duet_p2_tree_policy="off",
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
            duet_p2_tree_max_nodes=args.p2_nodes or args.k2 * 2,
            duet_p2_tree_verify_nodes=args.p2_verify_nodes or args.p2_nodes or args.k2 * 2,
            duet_p1_tree_max_nodes=args.p1_nodes or args.k1 * 2,
            duet_p1_tree_verify_nodes=args.p1_verify_nodes or args.p1_nodes or args.k1 * 2,
            duet_tree_c_tensor=args.tree_width,
            duet_p1_roots_per_position=args.p1_roots,
            duet_p2_budget=args.p2_budget,
            duet_tree_root_count=args.tree_root_count,
            duet_tree_beta=args.tree_beta,
            duet_tree_proxy_threshold=args.tree_proxy_threshold,
            duet_tree_conf_threshold=args.tree_conf_threshold,
            duet_root_source=args.root_source,
            duet_root_normalization=args.root_normalization,
            duet_root_overlap_mix=args.root_overlap_mix,
        )
    env = {k: v for k, v in os.environ.items()
           if k.startswith("SSD_") or k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
    checkout = Path(__file__).resolve().parents[2]
    runtime_sources = sorted((checkout / "ssd/ssd").rglob("*.py")) + sorted(
        (checkout / "ssd/bench").glob("mlsys*.py"))
    report = dict(
        status="running", args={k: str(v) if isinstance(v, Path) else v
                                for k, v in vars(args).items()},
        engine_kwargs=kwargs, env=env, torch=torch.__version__,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
        runtime_source_sha256={str(p.relative_to(checkout)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in runtime_sources},
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
    llm = None
    try:
        llm = LLM(args.target, **kwargs)
        report["target_dtype"] = str(llm.config.hf_config.torch_dtype)
        report["draft_dtype"] = str(llm.config.draft_hf_config.torch_dtype) if args.mode != "ar" else None
        report["effective_features"] = dict(
            fast_verify=os.environ.get("SSD_FAST_VERIFY", "1") == "1",
            batched_proxy_graph=os.environ.get("SSD_BATCHED_PROXY_GRAPH", "1") == "1",
            exit_replica=llm.config.duet_exit_replica,
            jit_speculate=llm.config.jit_speculate)
        report['resolved_env']={k:v for k,v in os.environ.items() if k.startswith('SSD_')}
        if duet:
            report['effective_features'].update(
                duet_jit_short=llm.config.duet_jit_short,
                miss_depth=int(os.getenv('SSD_DUET_MISS_K',str(args.k2 if llm.config.duet_jit_short else max(args.k1,args.k2)))),
                miss_width=int(os.getenv('SSD_DUET_MISS_WIDTH','1')),
                tree_fused_math=os.getenv('SSD_TREE_FUSED_MATH','0')=='1',
                tree_parallel_insert=os.getenv('SSD_TREE_PARALLEL_INSERT','0')=='1',
                bulk_export=os.getenv('SSD_BATCH_TREE_BULK_EXPORT','0')=='1')
        preemption_events = []
        if args.audit_preemption:
            lookup = {tuple(ids): i for i, ids in enumerate(prompts)}
            if len(lookup) != len(prompts):
                raise ValueError("Preemption audit requires unique tokenized prompts")
            original_preempt = llm.scheduler.preempt

            def observe_preempt(seq):
                index = lookup[tuple(seq.token_ids[:seq.initial_num_prompt_tokens])]
                preemption_events.append(dict(question=index,
                    output_position=seq.num_completion_tokens, prefix_ids=list(seq.token_ids)))
                original_preempt(seq)
            llm.scheduler.preempt = observe_preempt
        for temperature in args.temperatures:
            for batch in args.batches:
                # Captured capacity stays fixed; only the scheduler limit changes.
                llm.scheduler.max_num_seqs = batch
                warm = SamplingParams(temperature=temperature, max_new_tokens=16,
                                      ignore_eos=args.ignore_eos)
                if observer:observer.active=False
                llm.generate(prompts[:batch], warm, use_tqdm=False)
                for seed in args.seeds:
                    preemption_events.clear()
                    random.seed(seed)
                    torch.manual_seed(seed)
                    # SSD_SEED seeds draft process at startup. Each cell records
                    # that separately: resetting rank0 is NOT a draft RNG reset.
                    params = [SamplingParams(temperature=temperature,
                              max_new_tokens=n, ignore_eos=args.ignore_eos) for n in caps]
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    if observer:observer.active=True
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
                                summary=summarize(metrics, outputs, wall, args.mode != "ar"))
                    report["cells"].append(cell)
                    if args.audit_preemption:
                        cell["preemption_events"] = list(preemption_events)
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
        if llm is not None:
            llm.exit(hard=False)


if __name__ == "__main__":
    main()
