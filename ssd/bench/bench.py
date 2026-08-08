import os
import ssd.paths  # noqa: F401 — sets TORCH_CUDA_ARCH_LIST before flashinfer import
import sys
import time
import argparse
import json
from random import randint, seed
from ssd import LLM, SamplingParams
from ssd.engine.llm_engine import METRICS
from transformers import AutoTokenizer
import wandb
from bench_helpers import get_model_paths, generate_benchmark_inputs


def parse_arguments():
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark SSD performance (API similar to example.py)")

    # Model configuration
    parser.add_argument("--size", type=str, choices=["0.6", "1.7", "4", "8", "14", "32", "1", "3", "70"], default="70",
                        help="Model size in billions of parameters (0.6, 1.7, 4, 8, 14, 32, 1, 3, 70)")
    parser.add_argument("--llama", action="store_true", default=True, help="Use Llama models (default)")
    parser.add_argument("--qwen", action="store_true", help="Use Qwen models instead of Llama")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model size (0.6 for Qwen-0.6B, 1 for Llama-1B) or path to draft model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Direct path to target model (overrides --size/--llama/--qwen)")
    parser.add_argument("--draft_path", type=str, default=None,
                        help="Direct path to draft model (overrides --draft)")

    # Execution configuration
    parser.add_argument("--eager", action="store_true", help="Use eager execution (disable CUDA graphs)")
    parser.add_argument("--gpus", type=int, default=1, help="Total number of gpus")

    # Speculative decoding configuration
    parser.add_argument("--spec", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--eagle", action="store_true", help="Enable eagle speculative decoding (implies --spec, uses default eagle draft for model)")
    parser.add_argument("--k", type=int, default=6, help="Speculative decoding k value")
    parser.add_argument("--async", action="store_true", help="Enable async speculative decoding")
    parser.add_argument("--f", type=int, default=3, help="Async fan out value")
    parser.add_argument("--flh", type=int, nargs='+', default=None, help="Fan out list (e.g., --flh 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flm", type=int, nargs='+', default=None, help="Fan out list miss (e.g., --flm 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--backup", type=str, choices=["jit", "fast"], default="jit", help="Backup strategy (jit or fast)")

    # Memory and batching configuration
    parser.add_argument("--block_sz", type=int, default=256, help="KV cache block size (see config.py: kvcache_block_size)")
    parser.add_argument("--b", type=int, default=1, help="Maximum number of sequences in batch")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                        help="Fraction of free GPU memory to reserve for KV cache. "
                             "Useful for sync SD where target rank 0 also hosts the draft model.")

    # Generation configuration
    parser.add_argument("--input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--output_len", type=int, default=512, help="Maximum output length")
    parser.add_argument("--numseqs", type=int, default=128, help="Number of sequences to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--dtemp", type=float, default=None, help="Draft async temperature (overrides --temp)")
    parser.add_argument("--x", type=float, default=None, help="Sampler x for generation (Saguaro sampling coefficient)")

    # Example mode
    parser.add_argument("--example", action="store_true", help="Use real prompts like in example.py and print generations (supports up to batch size 8)")
    parser.add_argument("--humaneval", action="store_true", help="Use HumanEval prompts")
    parser.add_argument("--alpaca", action="store_true", help="Use Alpaca prompts")
    parser.add_argument("--c4", action="store_true", help="Use C4 prompts")
    parser.add_argument("--ultrafeedback", action="store_true", help="Use UltraFeedback prompts")
    parser.add_argument("--random", action="store_true", help="Use random tokens instead of dataset prompts")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed Python random + torch. Reproducible prompts (with --random) and sampling.")
    parser.add_argument("--prompt_offset", type=int, default=0, help="Skip first N prompts per dataset (for variance testing)")
    parser.add_argument("--all", action="store_true", help="Use numseqs from each dataset (union dataset with numseqs*4 total)")
    parser.add_argument("--chat_template", action="store_true", help="Wrap dataset prompts in chat template before tokenizing")

    # Debugging and logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (saves draft inputs during prefill)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps to run")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    parser.add_argument("--group", type=str, default=None, help="Wandb group name")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name")

    # DUET-SSD configuration
    parser.add_argument("--duet", action="store_true", help="Enable DUET-SSD (early-exit proxy)")
    parser.add_argument("--duet_exit_layer", type=int, default=None,
                        help="Early-exit layer index (default: 2*L//3)")
    parser.add_argument("--duet_proxy_top_k", type=int, default=3,
                        help="Number of proxy correction tokens per position")
    parser.add_argument("--duet_draft_fan_out", type=int, default=None,
                        help="Draft-sourced branches per position (default: fan_out//2)")
    parser.add_argument("--duet_policy", choices=("a", "b"), default="b",
                        help="Phase-2 budget: 'b' = unified K+1 (default since 2026-04-29; "
                             "see docs/duet/05-policy-b-fix.md). 'a' = legacy h_i proportional (dead branch).")
    parser.add_argument("--duet_phase1_k", type=int, default=None,
                        help="v1 hybrid: Phase 1 forward depth K1 (must sum with --duet_phase2_k to --k). "
                             "When set, enables hybrid v1 (proxy K2 + verify dispatch).")
    parser.add_argument("--duet_phase2_k", type=int, default=None,
                        help="v1 hybrid: Phase 2 forward depth K2 = K_short. Required iff --duet_phase1_k is set.")
    parser.add_argument("--duet_split_phase1_fan_out_list", type=str, default=None,
                        help="Split-only K1/K2: comma-separated per-position fan_out list "
                             "for Phase 1 (length must be K1+1). E.g. '4,4,3,3,2,2,1,1,1' "
                             "for K1=8. Default = uniform [draft_fo]*(K1+1).")
    # Tier-1 canonical names (docs/duet/16). Old flags above still work and
    # print a deprecation note; passing both with different values is an error.
    parser.add_argument("--duet_k1", type=int, default=None,
                        help="Canonical alias for --duet_phase1_k (Phase 1 depth K1).")
    parser.add_argument("--duet_k2", type=int, default=None,
                        help="Canonical alias for --duet_phase2_k (Phase 2 depth K2).")
    parser.add_argument("--duet_p1_fanout", type=int, default=None,
                        help="Canonical alias for --duet_draft_fan_out.")
    parser.add_argument("--duet_p1_fanout_list", type=str, default=None,
                        help="Canonical alias for --duet_split_phase1_fan_out_list "
                             "(comma list, length K1+1).")
    parser.add_argument("--duet_no_jit_short", action="store_true",
                        help="Disable JIT-short (miss JIT at K2 depth). Default is ON "
                             "(champion standard; replaces SSD_DUET_JIT_SHORT=1).")
    parser.add_argument("--duet_p2_budget", type=int, default=None,
                        help="Direct Phase-2 seed budget (docs/duet/16 Tier-3). "
                             "Default: derived pfo*(K1+1) where pfo = f - p1_fanout.")
    parser.add_argument("--duet_p1_tree_policy", choices=("off", "on"),
                        default="off",
                        help="Enable DUET's dynamic Phase-1 tree.")
    parser.add_argument("--duet_p2_tree_policy", choices=("off", "on"),
                        default="on",
                        help="Enable DUET's dynamic Phase-2 tree.")
    # Old experiment scripts remain runnable, but this option is intentionally
    # absent from --help.  New runs use the phase-specific on/off switches.
    parser.add_argument("--duet_tree_policy",
                        choices=("off", "on", "eagle", "adaptive",
                                 "coverage", "confidence", "level",
                                 "frontier"), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--duet_tree_c_tensor", type=int, default=3,
                        help="Dynamic tree: maximum ordered child samples per parent.")
    parser.add_argument("--duet_p1_tree_max_nodes", type=int, default=18,
                        help="Maximum P1 tree nodes generated per cache root.")
    parser.add_argument("--duet_p2_tree_max_nodes", type=int, default=8,
                        help="Maximum P2 tree nodes generated per cache root.")
    parser.add_argument("--duet_p1_tree_verify_nodes", type=int, default=None,
                        help="Maximum P1 tree nodes reranked and sent on a hit. "
                             "Default: all generated P1 nodes.")
    parser.add_argument("--duet_p2_tree_verify_nodes", type=int, default=None,
                        help="Maximum P2 tree nodes reranked and sent on a hit. "
                             "Default: all generated P2 nodes.")
    parser.add_argument("--duet_p1_roots_per_position", type=int, default=2,
                        help="Uniform number of P1 root candidates per glue position.")
    parser.add_argument("--duet_p1_tree_forward_scale", type=float,
                        default=1.0,
                        help="P1 captured forward width / root count. The dynamic "
                             "default is 1; values above 1 explicitly spend "
                             "more compute on additional frontier lanes.")
    parser.add_argument("--duet_tree_nv", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--duet_tree_fanout_policy", type=str,
                        default="backbone", choices=["backbone", "ctensor"])
    parser.add_argument("--duet_tree_root_count", type=int, default=None,
                        help="Number R of P2 roots. Dynamic P2 defaults to R=W; "
                             "R must not exceed the P2 forward width W. "
                             "Legacy level/frontier also honor this override; "
                             "confidence derives R and coverage keeps R=W.")
    parser.add_argument("--duet_tree_proxy_threshold", type=float,
                        default=0.01,
                        help="Dynamic P2 tree: keep every root's first "
                             "forward, but do not expand deeper below roots "
                             "with a smaller calibrated proxy score.")
    parser.add_argument("--duet_tree_conf_threshold", type=float,
                        default=0.03,
                        help="Dynamic P2 tree: keep the sampled child as a leaf, "
                             "but do not expand deeper below children with "
                             "smaller calibrated draft confidence.")
    parser.add_argument("--duet_p1_tree_start_threshold", type=float,
                        default=0.0,
                        help="Dynamic P1 tree: minimum glue-reach*root-q start "
                             "score for expansion after round zero. Default 0 "
                             "keeps P1 unfiltered until phase-specific calibration.")
    parser.add_argument("--duet_p1_tree_conf_threshold", type=float,
                        default=0.0,
                        help="Dynamic P1 tree: minimum local draft confidence "
                             "for expansion after round zero. Default 0.")
    parser.add_argument("--duet_tree_beta", type=float, default=0.5,
                        help="P2-tree: budget allocation exponent (E1 default 0.5).")

    # Weight-only quantization (target only) — AWQ Marlin is the supported path.
    # The legacy torchao backends (int4_wo_tile / int8_wo) remain in tree as an
    # unsupported internal fallback; their CLI was removed in this revision.
    # See `INT8-WEIGHT-ONLY-PLAN-v2.md` and `INT8-v2-IMPL-ISSUE.md`.
    parser.add_argument("--quant_lm_head", action="store_true",
                        help="[ignored for AWQ — kept dense by default for throughput/accept]")
    parser.add_argument("--quant_awq", action="store_true",
                        help="Enable AWQ-style W4A16 on TARGET via sgl-kernel Marlin (plan v2). "
                             "Requires --quant_awq_artifact or --quant_awq_external.")
    parser.add_argument("--quant_awq_artifact", type=str, default=None,
                        help="TARGET SSD-native AWQ artifact prefix (expects `<prefix>.rank{r}.awq.pt`).")
    parser.add_argument("--quant_awq_external", type=str, default=None,
                        help="TARGET external AutoAWQ HF checkpoint directory (reads qweight/qzeros/scales).")
    parser.add_argument("--quant_group_size", type=int, default=128,
                        help="AWQ group size (default 128; must evenly divide every RowParallel in-shard).")
    # Draft-side AWQ — independent of target
    parser.add_argument("--quant_awq_draft", action="store_true",
                        help="Enable AWQ W4A16 on DRAFT model (plan v2 extension). "
                             "Requires --quant_awq_draft_artifact. Llama-family, tp=1 only.")
    parser.add_argument("--quant_awq_draft_artifact", type=str, default=None,
                        help="DRAFT SSD-native AWQ artifact prefix.")
    parser.add_argument("--quant_awq_draft_external", type=str, default=None,
                        help="DRAFT external AutoAWQ HF checkpoint directory.")
    parser.add_argument("--quant_awq_draft_group_size", type=int, default=128,
                        help="DRAFT AWQ group size.")

    # Sweep mode: load engine once, run multiple configs
    parser.add_argument("--sweep", type=str, default=None,
                        help="JSON list of override dicts. Sweepable keys: temp, b. "
                             "Each dict also supports 'name' for wandb run name.")

    args = parser.parse_args()
    assert not (args.qwen and '--llama' in sys.argv), "--llama and --qwen are mutually exclusive"
    if args.qwen:
        args.llama = False
    if args.eagle:
        args.spec = True
        assert args.llama, "Eagle currently only supports llama models"
        assert args.temp == 0.0 and args.dtemp is None, "Eagle currently only supports greedy decoding (temp=0)"
        assert getattr(args, 'async', False), "Eagle currently only supports async speculative decoding"

    # DUET Tier-1 alias normalization (docs/duet/16).
    if getattr(args, 'duet', False):
        def _merge_alias(new_name, old_name):
            new_v = getattr(args, new_name)
            old_v = getattr(args, old_name)
            if new_v is not None and old_v is not None and new_v != old_v:
                raise SystemExit(
                    f"[bench] --{new_name}={new_v} and --{old_name}={old_v} "
                    f"disagree; pass only one.")
            if new_v is not None:
                setattr(args, old_name, new_v)
            elif old_v is not None:
                print(f"[bench][deprecated] --{old_name} -> use --{new_name}",
                      flush=True)
        _merge_alias("duet_k1", "duet_phase1_k")
        _merge_alias("duet_k2", "duet_phase2_k")
        _merge_alias("duet_p1_fanout", "duet_draft_fan_out")
        _merge_alias("duet_p1_fanout_list", "duet_split_phase1_fan_out_list")
        # --k omission: derive k = K1 + K2 (Config asserts equality when
        # --k is passed explicitly).
        if (args.duet_phase1_k is not None and args.duet_phase2_k is not None
                and '--k' not in sys.argv):
            args.k = args.duet_phase1_k + args.duet_phase2_k
            print(f"[bench] --k omitted -> derived k = K1+K2 = {args.k}",
                  flush=True)
        # --f omission with a direct budget: keeps the legacy pfo = f - dfo
        # derivation consistent (Config still computes pfo; the direct
        # budget overrides the total via duet_p2_budget).
        if (args.duet_p2_budget is not None and '--f' not in sys.argv
                and args.duet_draft_fan_out is not None
                and args.duet_phase1_k is not None):
            import math
            args.f = args.duet_draft_fan_out + math.ceil(
                args.duet_p2_budget / (args.duet_phase1_k + 1))
            print(f"[bench] --f omitted -> derived f = p1_fanout + "
                  f"ceil(budget/(K1+1)) = {args.f}", flush=True)
    return args


def create_run_name(args):
    """Create a descriptive run name for wandb logging."""
    spec_mode_str = "spec" if args.spec else "normal"
    async_mode_str = "_async" if getattr(args, 'async', False) else ""
    jit_mode_str = "_jit" if args.backup == "jit" else ""
    model_type = "llama" if args.llama else "qwen"
    example_str = "_example" if args.example else ""
    humaneval_str = "_humaneval" if args.humaneval else ""
    alpaca_str = "_alpaca" if args.alpaca else ""
    c4_str = "_c4" if args.c4 else ""
    ultrafeedback_str = "_ultrafeedback" if args.ultrafeedback else ""
    random_str = "_random" if args.random else ""
    all_str = "_all" if args.all else ""
    gsm_str = "_gsm" if not args.example and not args.humaneval and not args.alpaca and not args.c4 and not args.ultrafeedback and not args.random and not args.all else ""
    sampler_x_str = f"_sampler_x{args.x}" if args.x else ""

    temp_str = f"_temp{args.temp}"
    if args.dtemp is not None:
        temp_str += f"_dtemp{args.dtemp}"

    draft_str = f"_draft{args.draft}" if args.draft is not None else "_nodraft"
    k_str = f"_k{args.k}"
    f_str = f"_f{args.f}"

    return args.name if args.name else f"{model_type}_size{args.size}_{spec_mode_str}{async_mode_str}{jit_mode_str}_b{args.b}{k_str}{f_str}{draft_str}{temp_str}{sampler_x_str}{example_str}{humaneval_str}{alpaca_str}{c4_str}{ultrafeedback_str}{random_str}{all_str}{gsm_str}"


def initialize_wandb(args, run_name):
    """Initialize wandb logging if requested."""
    if not args.wandb:
        return

    wandb.init(
        project="ssd",
        name=run_name,
        group=args.group,
        config={
            "model_size": args.size,
            "gpus": args.gpus,
            "speculative_decoding": args.spec,
            "async_speculative": getattr(args, 'async', False),
            "jit_speculative": args.backup == "jit",
            "k": args.k if args.spec else None,
            "f": args.f,
            "fan_out_list": args.flh,
            "fan_out_list_miss": args.flm,
            "llama": args.llama,
            "max_model_len": args.max_model_len,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "numseqs": args.numseqs,
            "draft_model": args.draft,
            "b": args.b,
            "block_size": args.block_sz,
            "eager": args.eager,
            "example_mode": args.example,
            "humaneval_mode": args.humaneval,
            "alpaca_mode": args.alpaca,
            "c4_mode": args.c4,
            "ultrafeedback_mode": args.ultrafeedback,
            "random_mode": args.random,
            "all_mode": args.all,
            "sampler_x": args.x,
            "implementation": "ssd",
            "max_steps": args.max_steps,
        }
    )


def create_llm_kwargs(args, draft_path):
    """Create LLM initialization arguments."""
    llm_kwargs = dict(
        enforce_eager=args.eager,
        num_gpus=args.gpus,
        speculate=args.spec,
        speculate_k=args.k,
        draft_async=getattr(args, 'async', False),
        async_fan_out=args.f,
        verbose=args.verbose,
        draft=draft_path,
        kvcache_block_size=args.block_sz,
        max_num_seqs=args.b,
        max_model_len=args.max_model_len,
        sampler_x=args.x,
        jit_speculate=(args.backup == "jit"),
        max_steps=args.max_steps,
    )
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization

    # DUET-SSD
    if getattr(args, 'duet', False):
        llm_kwargs["duet_enabled"] = True
        llm_kwargs["jit_speculate"] = True  # DUET requires jit_speculate
        if args.duet_exit_layer is not None:
            llm_kwargs["duet_exit_layer"] = args.duet_exit_layer
        llm_kwargs["duet_proxy_top_k"] = args.duet_proxy_top_k
        if args.duet_draft_fan_out is not None:
            llm_kwargs["duet_draft_fan_out"] = args.duet_draft_fan_out
        llm_kwargs["duet_policy"] = args.duet_policy
        llm_kwargs["duet_jit_short"] = not args.duet_no_jit_short
        if args.duet_p2_budget is not None:
            llm_kwargs["duet_p2_budget"] = args.duet_p2_budget
        llm_kwargs["duet_p1_tree_policy"] = args.duet_p1_tree_policy
        llm_kwargs["duet_p2_tree_policy"] = args.duet_p2_tree_policy
        if args.duet_tree_policy is not None:
            # Deprecated input only.  ``on`` is accepted as a convenience
            # and normalized to the old internal dynamic selector name.
            llm_kwargs["duet_tree_policy"] = (
                "eagle" if args.duet_tree_policy == "on"
                else args.duet_tree_policy)
        _any_tree = (args.duet_p1_tree_policy == "on"
                     or args.duet_p2_tree_policy == "on"
                     or args.duet_tree_policy not in (None, "off"))
        if _any_tree:
            llm_kwargs["duet_tree_c_tensor"] = args.duet_tree_c_tensor
            llm_kwargs["duet_p1_tree_max_nodes"] = \
                args.duet_p1_tree_max_nodes
            llm_kwargs["duet_p2_tree_max_nodes"] = \
                args.duet_p2_tree_max_nodes
            if args.duet_p1_tree_verify_nodes is not None:
                llm_kwargs["duet_p1_tree_verify_nodes"] = \
                    args.duet_p1_tree_verify_nodes
            if args.duet_p2_tree_verify_nodes is not None:
                llm_kwargs["duet_p2_tree_verify_nodes"] = \
                    args.duet_p2_tree_verify_nodes
            llm_kwargs["duet_p1_roots_per_position"] = \
                args.duet_p1_roots_per_position
            llm_kwargs["duet_p1_tree_forward_scale"] = \
                args.duet_p1_tree_forward_scale
            if args.duet_tree_nv is not None:
                llm_kwargs["duet_tree_nv"] = args.duet_tree_nv
            llm_kwargs["duet_tree_beta"] = args.duet_tree_beta
            llm_kwargs["duet_tree_fanout_policy"] = args.duet_tree_fanout_policy
            llm_kwargs["duet_tree_proxy_threshold"] = \
                args.duet_tree_proxy_threshold
            llm_kwargs["duet_tree_conf_threshold"] = \
                args.duet_tree_conf_threshold
            llm_kwargs["duet_p1_tree_start_threshold"] = \
                args.duet_p1_tree_start_threshold
            llm_kwargs["duet_p1_tree_conf_threshold"] = \
                args.duet_p1_tree_conf_threshold
            if args.duet_tree_root_count is not None:
                llm_kwargs["duet_tree_root_count"] = args.duet_tree_root_count
        if args.duet_phase1_k is not None:
            llm_kwargs["duet_phase1_k"] = args.duet_phase1_k
        if args.duet_phase2_k is not None:
            llm_kwargs["duet_phase2_k"] = args.duet_phase2_k
        if args.duet_split_phase1_fan_out_list is not None:
            llm_kwargs["duet_split_phase1_fan_out_list"] = [
                int(x) for x in args.duet_split_phase1_fan_out_list.split(",")
            ]

    # AWQ W4A16 (Marlin) — plan v2 primary direction (TARGET)
    if getattr(args, 'quant_awq', False):
        if not (getattr(args, 'quant_awq_artifact', None) or
                getattr(args, 'quant_awq_external', None)):
            raise SystemExit(
                "[bench] --quant_awq requires either --quant_awq_artifact "
                "<SSD-native prefix> or --quant_awq_external <AutoAWQ HF dir>."
            )
        llm_kwargs["target_quant_enabled"] = True
        llm_kwargs["target_quant_backend"] = "awq_marlin"
        if getattr(args, 'quant_awq_artifact', None):
            llm_kwargs["target_quant_awq_artifact"] = args.quant_awq_artifact
        if getattr(args, 'quant_awq_external', None):
            llm_kwargs["target_quant_external_awq_path"] = args.quant_awq_external
        if getattr(args, 'quant_group_size', 128) != 128:
            llm_kwargs["target_quant_group_size"] = args.quant_group_size
        if getattr(args, 'quant_lm_head', False):
            print("[bench] --quant_lm_head is ignored for the AWQ path "
                  "(lm_head stays dense).", flush=True)

    # AWQ on DRAFT — independent from target (can be enabled alone or together)
    if getattr(args, 'quant_awq_draft', False):
        if not (getattr(args, 'quant_awq_draft_artifact', None) or
                getattr(args, 'quant_awq_draft_external', None)):
            raise SystemExit(
                "[bench] --quant_awq_draft requires either "
                "--quant_awq_draft_artifact <SSD-native prefix> or "
                "--quant_awq_draft_external <AutoAWQ HF dir>."
            )
        llm_kwargs["draft_quant_enabled"] = True
        llm_kwargs["draft_quant_backend"] = "awq_marlin"
        if getattr(args, 'quant_awq_draft_artifact', None):
            llm_kwargs["draft_quant_awq_artifact"] = args.quant_awq_draft_artifact
        if getattr(args, 'quant_awq_draft_external', None):
            llm_kwargs["draft_quant_external_awq_path"] = args.quant_awq_draft_external
        if getattr(args, 'quant_awq_draft_group_size', 128) != 128:
            llm_kwargs["draft_quant_group_size"] = args.quant_awq_draft_group_size

    if args.flh is not None:
        llm_kwargs["fan_out_list"] = args.flh
    if args.flm is not None:
        llm_kwargs["fan_out_list_miss"] = args.flm

    return llm_kwargs


def log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, mode, run_name):
    """Log metrics to wandb if enabled."""
    if not args.wandb:
        return

    wandb_metrics = {
        "official_total_tokens": total_tokens,
        "official_total_time": total_time,
        "official_end_to_end_throughput": throughput,
        "model_name": model_name,
        "mode": mode,
        "run_name": run_name,
    }

    if metrics:
        if "prefill_total_time" in metrics and "prefill_total_tokens" in metrics:
            if metrics["prefill_total_time"] > 0:
                wandb_metrics["metrics_prefill_throughput"] = metrics["prefill_total_tokens"] / metrics["prefill_total_time"]

        if "decode_total_time" in metrics and "decode_total_tokens" in metrics:
            if metrics["decode_total_time"] > 0:
                wandb_metrics["metrics_decode_throughput"] = metrics["decode_total_tokens"] / metrics["decode_total_time"]

        if "target_step_times" in metrics and metrics["target_step_times"]:
            avg_target_step_time_ms = sum(metrics["target_step_times"]) * 1000 / len(metrics["target_step_times"])
            wandb_metrics["metrics_avg_target_step_time_ms"] = avg_target_step_time_ms

        if "cache_hits" in metrics and metrics["cache_hits"]:
            wandb_metrics["metrics_avg_cache_hits"] = sum(metrics["cache_hits"]) / len(metrics["cache_hits"])

        if "accepted_suffix_lens_with_recovery" in metrics and metrics["accepted_suffix_lens_with_recovery"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_with_recovery"] = sum(metrics["accepted_suffix_lens_with_recovery"]) / len(metrics["accepted_suffix_lens_with_recovery"])
            wandb_metrics["metrics_accepted_suffix_lens_with_recovery_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_with_recovery"])

        if "accepted_suffix_lens_on_hit" in metrics and metrics["accepted_suffix_lens_on_hit"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_on_hit"] = sum(metrics["accepted_suffix_lens_on_hit"]) / len(metrics["accepted_suffix_lens_on_hit"])
            wandb_metrics["metrics_accepted_suffix_lens_on_hit_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_on_hit"])

    wandb.log(wandb_metrics)


def run_benchmark(args, llm, prompts, sampling_params):
    """Run the actual benchmark and return results."""
    if args.wandb:
        wandb.log({"sequences_processed": 0, "total_sequences": len(prompts)})

    start_time = time.time()
    outputs, metrics = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    if args.wandb:
        wandb.log({"sequences_processed": len(prompts), "total_sequences": len(prompts)})

    return outputs, total_time, metrics


def reset_metrics():
    """Reset the global METRICS dict between sweep runs."""
    for k, v in METRICS.items():
        if isinstance(v, list):
            v.clear()
        else:
            METRICS[k] = 0


def reconfigure_engine(llm, b=None):
    """Reconfigure a live engine for a new sweep run without reloading weights."""
    if b is not None:
        assert b <= llm.config.max_num_seqs, f"b={b} > initial max_num_seqs={llm.config.max_num_seqs}"
        llm.config.max_num_seqs = b
        llm.scheduler.max_num_seqs = b


def main():
    args = parse_arguments()
    seed(0)

    if args.example and args.numseqs > 8:
        print("Warning: --example mode supports up to 8 sequences, reducing numseqs from {} to 8".format(args.numseqs))
        args.numseqs = 8

    model_name, model_path, draft_path = get_model_paths(args)
    # Direct path overrides
    if args.model_path:
        model_path = args.model_path
        model_name = os.path.basename(model_path)
    if args.draft_path:
        draft_path = args.draft_path

    # Seed BEFORE generate_benchmark_inputs so --random prompts are reproducible.
    if args.seed is not None:
        import random as _random
        import torch as _torch
        try:
            import numpy as _np
            _np.random.seed(args.seed)
        except ImportError:
            pass
        _random.seed(args.seed)
        _torch.manual_seed(args.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.seed)
        print(f"[bench] seeded with {args.seed}", flush=True)

    string_prompts, prompt_token_ids, original_prompts = generate_benchmark_inputs(args, model_path)
    prompts = string_prompts if string_prompts is not None else prompt_token_ids

    if prompts:
        num_reqs = len(prompts)
    else:
        num_reqs = args.numseqs
    sampling_params = [SamplingParams(
        temperature=args.temp,
        draft_temperature=args.dtemp,
        ignore_eos=True,
        max_new_tokens=args.output_len,
    ) for _ in range(num_reqs)]

    if prompts:
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                print(f'Prompt: {prompt}')
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                num_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            elif isinstance(prompt, list):
                num_tokens = len(prompt)
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}")

    # Create LLM (once, reused across sweep configs)
    llm_kwargs = create_llm_kwargs(args, draft_path)
    if args.eagle:
        llm_kwargs['use_eagle'] = True
    if args.debug:
        llm_kwargs['debug_mode'] = True

    llm = LLM(model_path, **llm_kwargs)

    # Build sweep configs
    if args.sweep:
        sweep_configs = json.loads(args.sweep)
        assert isinstance(sweep_configs, list), "--sweep must be a JSON list of dicts"
    else:
        sweep_configs = [{}]

    for si, sweep_cfg in enumerate(sweep_configs):
        bad_keys = {"backup", "flh", "flm"} & set(sweep_cfg.keys())
        assert not bad_keys, f"Cannot sweep {bad_keys} — draft process won't see changes."

        temp = sweep_cfg.get("temp", args.temp)
        b = sweep_cfg.get("b", args.b)
        run_name_override = sweep_cfg.get("name", None)

        reconfigure_engine(llm, b=b)

        cur_sampling_params = [SamplingParams(
            temperature=temp,
            draft_temperature=args.dtemp,
            ignore_eos=True,
            max_new_tokens=args.output_len,
        ) for _ in range(num_reqs)]

        reset_metrics()

        if run_name_override:
            cur_run_name = run_name_override
        elif args.sweep:
            cur_run_name = f"{create_run_name(args)}_sweep{si}"
        else:
            cur_run_name = create_run_name(args)

        orig_temp, orig_name, orig_b = args.temp, args.name, args.b
        args.temp, args.b = temp, b
        if run_name_override:
            args.name = run_name_override
        initialize_wandb(args, cur_run_name)
        args.temp, args.name, args.b = orig_temp, orig_name, orig_b

        try:
            print(f"\n{'='*60}")
            print(f"SWEEP [{si+1}/{len(sweep_configs)}] temp={temp} b={b}")
            print(f"{'='*60}")

            outputs, total_time, metrics = run_benchmark(args, llm, prompts, cur_sampling_params)

            total_tokens = sum(sp.max_new_tokens for sp in cur_sampling_params)
            throughput = total_tokens / total_time

            mode = "Eager" if args.eager else "CUDA Graphs"
            spec_mode = f" + Speculative(k={args.k})" if args.spec else ""
            async_mode = " + Async" if getattr(args, 'async', False) else ""
            jit_mode = " + JIT" if args.backup == "jit" else ""
            x_mode = f" + X({args.x})" if args.x else ""
            full_mode = mode + spec_mode + async_mode + jit_mode + x_mode

            print(f"Model: {model_name}, Mode: {full_mode}, Total: {total_tokens}tok, Time: {total_time:.2f}s, Total Throughput: {throughput:.2f}tok/s")

            if not args.random and si == 0:
                print("\n" + "="*80)
                print("GENERATIONS:")
                print("="*80)

                tokenizer = AutoTokenizer.from_pretrained(model_path)

                for i, (prompt, output) in enumerate(zip(prompts, outputs)):
                    if i >= 10:
                        break
                    if isinstance(prompt, list):
                        decoded_prompt = tokenizer.decode(prompt, skip_special_tokens=True)
                    else:
                        decoded_prompt = prompt
                    if original_prompts and i < len(original_prompts):
                        display_prompt = original_prompts[i]
                    else:
                        display_prompt = decoded_prompt
                    print(f"\nPrompt {i+1}: {display_prompt!r}")
                    print(f"Generation: {output['text']!r}")
                    print("-" * 40)

            log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, full_mode, cur_run_name)

            if args.wandb:
                wandb.finish()
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
            break

    print(f'Engine exited!')
    sys.exit(0)


if __name__ == "__main__":
    main()
