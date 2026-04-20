"""MESA-SSD detailed latency analysis.
Measures target/draft breakdown with CUDA events (no sync overhead).
Usage: python -O bench/mesa_analysis.py
"""
import os, sys, json, time
os.environ["SSD_CUDA_ARCH"] = "8.6"
os.environ["SSD_HF_CACHE"] = "/data2/chokwans99/models"
os.environ["SSD_DATASET_DIR"] = "/tmp"

import ssd.paths
from ssd import LLM, SamplingParams
from ssd.engine.llm_engine import METRICS

TARGET = "/data2/chokwans99/models/layerskip-llama3-8B"
DRAFT = "/data2/chokwans99/models/Llama-3.2-1B-Instruct"

def run_experiment(mesa_enabled, exit_layer=None, draft_fan_out=None,
                   num_seqs=10, output_len=256, temp=0.6, fan_out=3, k=4):
    """Run one experiment configuration and return metrics."""
    kwargs = dict(
        model=TARGET, draft=DRAFT, num_gpus=2,
        speculate=True, speculate_k=k, draft_async=True,
        async_fan_out=fan_out, max_num_seqs=1, max_model_len=4096,
        jit_speculate=True,
    )
    if mesa_enabled:
        kwargs["mesa_enabled"] = True
        if exit_layer is not None:
            kwargs["mesa_exit_layer"] = exit_layer
        if draft_fan_out is not None:
            kwargs["mesa_draft_fan_out"] = draft_fan_out

    sp = SamplingParams(temperature=temp, max_new_tokens=output_len)

    # Generate random prompts
    from random import randint, seed
    seed(42)
    prompts = [[randint(0, 127999) for _ in range(128)] for _ in range(num_seqs)]

    llm = LLM(**kwargs)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, [sp] * num_seqs)
    t1 = time.perf_counter()

    total_tokens = sum(len(o.get("token_ids", [])) if isinstance(o, dict) else 0 for o in outputs)
    if total_tokens == 0:
        total_tokens = num_seqs * output_len  # fallback

    metrics = dict(METRICS)
    result = {
        "mesa_enabled": mesa_enabled,
        "exit_layer": exit_layer,
        "draft_fan_out": draft_fan_out,
        "fan_out": fan_out,
        "k": k,
        "total_tokens": total_tokens,
        "total_time": t1 - t0,
        "throughput": total_tokens / (t1 - t0),
        "avg_target_step_ms": sum(metrics.get("target_step_times", [])) / max(len(metrics.get("target_step_times", [])), 1) * 1000 if metrics.get("target_step_times") else None,
        "avg_target_verify_ms": sum(metrics.get("target_verify_times", [])) / max(len(metrics.get("target_verify_times", [])), 1) * 1000 if metrics.get("target_verify_times") else None,
        "avg_draft_step_ms": sum(metrics.get("draft_step_times", [])) / max(len(metrics.get("draft_step_times", [])), 1) * 1000 if metrics.get("draft_step_times") else None,
        "avg_accept_rate": sum(metrics.get("accepted_suffix_lens_with_recovery", [])) / max(len(metrics.get("accepted_suffix_lens_with_recovery", [])), 1) / (k + 1) if metrics.get("accepted_suffix_lens_with_recovery") else None,
        "cache_hit_rate": sum(metrics.get("cache_hits", [])) / max(len(metrics.get("cache_hits", [])), 1) if metrics.get("cache_hits") else None,
        "avg_tok_per_step": sum(metrics.get("accepted_suffix_lens_with_recovery", [])) / max(len(metrics.get("accepted_suffix_lens_with_recovery", [])), 1) if metrics.get("accepted_suffix_lens_with_recovery") else None,
    }
    return result


def main():
    results = []

    configs = [
        # Baseline
        {"mesa_enabled": False, "label": "Baseline SSD"},
        # MESA with different exit layers
        {"mesa_enabled": True, "exit_layer": 10, "label": "MESA exit=10 (31%)"},
        {"mesa_enabled": True, "exit_layer": 16, "label": "MESA exit=16 (50%)"},
        {"mesa_enabled": True, "exit_layer": 21, "label": "MESA exit=21 (66%)"},
        {"mesa_enabled": True, "exit_layer": 26, "label": "MESA exit=26 (81%)"},
        # MESA with different draft_fan_out
        {"mesa_enabled": True, "exit_layer": 21, "draft_fan_out": 1, "label": "MESA exit=21 draft_fo=1"},
        {"mesa_enabled": True, "exit_layer": 21, "draft_fan_out": 2, "label": "MESA exit=21 draft_fo=2"},
    ]

    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"Config: {cfg}")
        print(f"{'='*60}")

        try:
            result = run_experiment(**cfg, num_seqs=5, output_len=256)
            result["label"] = label
            results.append(result)

            print(f"  Throughput: {result['throughput']:.1f} tok/s")
            print(f"  Cache hit: {result['cache_hit_rate']:.3f}" if result['cache_hit_rate'] else "  Cache hit: N/A")
            print(f"  Tok/Step: {result['avg_tok_per_step']:.2f}" if result['avg_tok_per_step'] else "  Tok/Step: N/A")
            print(f"  Target verify: {result['avg_target_verify_ms']:.1f}ms" if result['avg_target_verify_ms'] else "  Target verify: N/A")
            print(f"  Draft step: {result['avg_draft_step_ms']:.1f}ms" if result['avg_draft_step_ms'] else "  Draft step: N/A")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"label": label, "error": str(e)})

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Config':<35} {'Throughput':>12} {'CacheHit':>10} {'Tok/Step':>10} {'Verify':>10} {'Draft':>10}")
    print(f"{'='*100}")
    for r in results:
        if "error" in r:
            print(f"{r['label']:<35} ERROR: {r['error'][:50]}")
        else:
            tp = f"{r['throughput']:.1f}" if r['throughput'] else "N/A"
            ch = f"{r['cache_hit_rate']:.3f}" if r['cache_hit_rate'] else "N/A"
            ts = f"{r['avg_tok_per_step']:.2f}" if r['avg_tok_per_step'] else "N/A"
            tv = f"{r['avg_target_verify_ms']:.1f}ms" if r['avg_target_verify_ms'] else "N/A"
            ds = f"{r['avg_draft_step_ms']:.1f}ms" if r['avg_draft_step_ms'] else "N/A"
            print(f"{r['label']:<35} {tp:>12} {ch:>10} {ts:>10} {tv:>10} {ds:>10}")

    # Save results
    with open("/tmp/mesa_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to /tmp/mesa_analysis_results.json")


if __name__ == "__main__":
    main()
