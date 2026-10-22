"""Read-only weights, native TP2 target A/B; isolated outputs only.

Run with torchrun --nproc_per_node=2. This measures the target component,
not an online DUET step. Baseline is the unmodified v1.4 method, imported
from a separately pinned clean checkout; both paths use identical weights/KV.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
import torch
import torch.distributed as dist
from ssd.config import Config
from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.sampling_params import SamplingParams
from ssd.utils.context import set_context, reset_context
from ssd.engine.helpers.cudagraph_helpers import capture_tree_verify_cudagraph


class TargetRunner(ModelRunner):
    def loop(self):
        # Lifecycle adapter only: both torchrun ranks coordinate direct calls.
        return


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def wire(valid, cap, branched):
    parents = [(-1 if i < 2 else (i - 2) // 2) if branched else i - 1
               for i in range(valid)]
    pad = cap - valid
    return tuple([valid, valid, 1, *range(300, 300 + valid), *([0] * pad),
                  *parents, *([-1] * pad),
                  *([i % 2 if branched else 0 for i in range(valid)]), *([0] * pad),
                  *[p + 1 for p in parents], *([0] * pad)])


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--inputs", type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    args.out.mkdir(parents=True, exist_ok=True)
    legacy_path = args.baseline / "ssd/ssd/engine/model_runner.py"
    spec = importlib.util.spec_from_file_location("ssd.engine.padding_baseline", legacy_path)
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    legacy = legacy_module.ModelRunner._run_tree_verify
    cfg = Config(model=args.target, draft=args.draft, dtype="bfloat16", num_gpus=2,
                 max_num_seqs=1, max_model_len=2048, max_num_batched_tokens=2048,
                 gpu_memory_utilization=.25, speculate=True, speculate_k=15,
                 draft_async=False, enforce_eager=True)
    r = TargetRunner(cfg, rank, None, num_tp_gpus=2)
    cfg.duet_enabled = True
    cfg.duet_exit_layer = 56
    cfg.duet_tree_c_tensor = 2
    cfg.duet_p1_tree_policy = cfg.duet_p2_tree_policy = "on"
    cfg.duet_p1_tree_verify_nodes = cfg.duet_p2_tree_verify_nodes = 12
    r.kv_cache.zero_()
    allocated_before = torch.cuda.memory_allocated()
    r._init_tree_verify_wrappers()
    assert r._tree_verify_buckets == (2, 4, 12), r._tree_verify_buckets
    dtype = torch.get_default_dtype()
    torch.set_default_device(r.device)
    torch.set_default_dtype(torch.bfloat16)
    try:
        started = time.perf_counter()
        r._tree_verify_cg, pool = capture_tree_verify_cudagraph(r)
        r._warmup_tree_topology_gpu()
        capture_seconds = time.perf_counter() - started
    finally:
        torch.set_default_device("cpu")
        torch.set_default_dtype(dtype)
    r.enforce_eager = False
    r._tree_proxy_graphs = {}
    r._duet_tree_wire_gpu_source = None
    r._duet_validated_tree = None
    all_wrappers = dict(r.tree_verify_wrappers)
    all_buckets = r._tree_verify_buckets
    if rank == 0:
        write(args.out / "ENVIRONMENT.json", dict(
            device=torch.cuda.get_device_name(), torch=torch.__version__,
            baseline_sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
            buckets=all_buckets, graph_count=len(r._tree_verify_cg),
            capture_seconds=capture_seconds,
            graph_allocation_bytes=torch.cuda.memory_allocated()-allocated_before,
            query_cap_nonroot=12, prefix_uid="317_t0", repeats=15, warmup=3,
            target_only=True, online_duet_step=False, kv_shape=list(r.kv_cache.shape)))
    prefixes = json.loads(args.inputs.read_text())["317_t0"]
    parity = []
    cells = [(v, c, t) for v in (2, 4, 6, 8, 10, 11, 12)
             for c in (128, 253, 255, 512, 1536) for t in (False, True)]
    random.Random(42).shuffle(cells)
    for cell_index, (valid, prefix, branched) in enumerate(cells):
        seq = Sequence(prefixes[:prefix], SamplingParams(temperature=.7, max_new_tokens=32))
        seq.block_table = list(range(r.config.max_blocks))
        tid, pos = r.prepare_prefill([seq])
        r.run_model(tid, pos, True, last_only=True)
        reset_context()
        n = valid + 1
        token = torch.tensor([prefixes[prefix], *range(300, 300+valid)], device=r.device, dtype=torch.int64)
        positions = torch.arange(prefix, prefix+n, device=r.device, dtype=torch.int64)
        slots = positions.to(torch.int32)
        lens = torch.tensor([prefix+n], device=r.device, dtype=torch.int32)
        blocks = torch.arange(r.config.max_blocks, device=r.device, dtype=torch.int32).view(1, -1)
        r._duet_verify_context_len_cpu = prefix+n
        r._duet_tree_meta = wire(valid, 12, branched)
        def invoke(mode):
            r.tree_verify_wrappers = {12:all_wrappers[12]} if mode != "sparse" else all_wrappers
            r._tree_verify_buckets = (12,) if mode != "sparse" else all_buckets
            set_context(is_prefill=False, slot_mapping=slots, context_lens=lens, block_tables=blocks)
            fn = legacy if mode == "legacy" else ModelRunner._run_tree_verify
            return fn(r, token, positions, False)
        # Save clones: subsequent graph replays overwrite persistent outputs.
        outputs = {}
        for mode in ("legacy", "trim_only", "sparse"):
            y = invoke(mode)
            if rank == 0:
                assert y.shape == (n, cfg.hf_config.vocab_size)
                assert torch.isfinite(y).all()
                outputs[mode] = y.clone()
        if rank == 0:
            ref = outputs["legacy"]
            p_ref = torch.softmax(ref.float()/.7, dim=-1)
            case = dict(valid=valid, prefix=prefix, branched=branched)
            for mode in ("trim_only", "sparse"):
                y = outputs[mode]
                tv = .5*(p_ref - torch.softmax(y.float()/.7, dim=-1)).abs().sum(-1)
                torch.testing.assert_close(y, ref, atol=.25, rtol=.02)
                case[mode] = dict(max_logit_error=(y-ref).abs().max().item(),
                    max_tvd=tv.max().item(), mean_tvd=tv.mean().item(),
                    argmax_agreement=(y.argmax(-1)==ref.argmax(-1)).float().mean().item(),
                    bitwise_equal=torch.equal(y, ref))
            parity.append(case)
        # Alternate order across cells and repeats to control drift.
        modes = ["legacy", "trim_only", "sparse"]
        random.Random(cell_index).shuffle(modes)
        values = {mode:[] for mode in modes}
        for mode in modes:
            for _ in range(3): invoke(mode)
        for repeat in range(15):
            for mode in modes[repeat % 3:] + modes[:repeat % 3]:
                torch.cuda.synchronize()
                dist.barrier(group=r.tp_pg, device_ids=[rank])
                start = time.perf_counter()
                y = invoke(mode)
                torch.cuda.synchronize()
                values[mode].append((time.perf_counter()-start)*1000)
        parts = [None, None]
        dist.all_gather_object(parts, dict(rank=rank, samples=values), group=r.tp_pg)
        if rank == 0:
            row = dict(valid=valid, query_rows=n, prefix=prefix, branched=branched,
                physical_rows={mode:(13 if mode != "sparse" else next(b+1 for b in all_buckets if b>=valid)) for mode in modes},
                rank_samples=parts,
                medians_ms={mode:statistics.median(values[mode]) for mode in modes})
            with (args.out / "samples.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            write(args.out / "STATUS.json", dict(state="measuring", completed=cell_index+1, total=len(cells)))
            print(json.dumps({k:v for k,v in row.items() if k!="rank_samples"}), flush=True)
    torch.cuda.synchronize()
    dist.barrier(group=r.tp_pg, device_ids=[rank])
    if rank == 0:
        write(args.out / "PARITY.json", dict(state="passed", cases=parity))
        write(args.out / "STATUS.json", dict(state="complete", completed=len(cells), total=len(cells)))
    if rank == 0:
        r.shm.close()
        r.shm.unlink()
    dist.barrier(group=r.tp_pg, device_ids=[rank])
    if rank != 0:
        r.shm.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
