"""Full draft weights, synthetic or real prefixes: tree execution only.

This is not end-to-end DUET throughput and does not test the cache protocol.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8])
    ap.add_argument('--repeats', type=int, default=50)
    ap.add_argument('--real-prefix', action='store_true', help='Prefill natural-text prefixes instead of zero KV')
    ap.add_argument('--diverse-prefix', action='store_true',
                    help='Use different natural-text prefixes for different requests')
    ap.add_argument('--dtype',choices=['auto','fp16','bf16'],default='auto')
    ap.add_argument('--strict-matmul',action='store_true')
    ap.add_argument('--plausible-roots',action='store_true')
    ap.add_argument('--pad-linear-rows',type=int,default=0,
                    help='Diagnostic: hold GEMM row shape fixed in serial and batched execution')
    ap.add_argument('--trace-layers',action='store_true',
                    help='Compare serial/batch first-round intermediates in eager mode')
    ap.add_argument('--hf-reference',action='store_true',
                    help='Compare root distributions to independent float32 HF eager attention')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    import torch
    from transformers import AutoConfig
    from ssd.models.llama3 import LlamaForCausalLM
    from ssd.models.qwen2 import Qwen2ForCausalLM
    from ssd.utils.loader import load_model
    from ssd.layers.attention import Attention
    from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
    from ssd.engine.helpers.batched_tree_executor import BatchedTreeExecutor
    torch.manual_seed(2026)
    device = torch.device('cuda:0')
    cfg = AutoConfig.from_pretrained(a.model)
    dtype = torch.bfloat16 if cfg.model_type == 'qwen2' else torch.float16
    if a.dtype != 'auto': dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[a.dtype]
    if a.strict_matmul:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    if a.pad_linear_rows:
        import torch.nn.functional as functional
        original_linear=functional.linear
        def fixed_rows_linear(x,weight,bias=None):
            n=x.shape[0]
            if x.ndim==2 and n<a.pad_linear_rows:
                padded=functional.pad(x,(0,0,0,a.pad_linear_rows-n))
                return original_linear(padded,weight,bias)[:n]
            return original_linear(x,weight,bias)
        functional.linear=fixed_rows_linear
    torch.set_default_dtype(dtype)
    cls = Qwen2ForCausalLM if cfg.model_type == 'qwen2' else LlamaForCausalLM
    with torch.device(device):
        model = cls(cfg, draft=True, speculate=True, draft_async=True, spec_k=2, async_fan_out=1)
    load_model(model, a.model)
    model.eval()
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    page, blocks = 256, 3
    for layer in model.modules():
        if isinstance(layer, Attention):
            layer.k_cache = torch.zeros(max(a.batches) * blocks, page,
                                        cfg.num_key_value_heads, hd, dtype=dtype, device=device)
            layer.v_cache = torch.zeros_like(layer.k_cache)
    if a.real_prefix:
        from transformers import AutoTokenizer
        from ssd.utils.context import set_context,reset_context
        tok=AutoTokenizer.from_pretrained(a.model)
        prefix=tok.encode('The following example explains how to reason about a computer system. '*40)[:128]
        prefix_texts = [
            'The following example explains how to reason about a computer system. ',
            'In Python, a dictionary maps keys to values. A function can return multiple values. ',
            'To solve the equation, subtract the constant and divide both sides by the coefficient. ',
            'A city library keeps books organized by author and subject for visitors to find. ',
            'The experimental measurements depend on the temperature and pressure of the gas. ',
            'The chef chopped the vegetables and prepared a warm soup for the evening meal. ',
            'A fictional traveler arrived at the village and asked for directions to the river. ',
            'For each item in the sorted list, compare the current value with the previous value. ',
        ]
        prefixes = [tok.encode(prefix_texts[b % len(prefix_texts)]*40)[:128]
                    if a.diverse_prefix else prefix for b in range(max(a.batches))]
        assert all(len(p)==128 for p in prefixes)
        with torch.inference_mode():
            Bmax=max(a.batches)
            ids=torch.tensor(prefixes,dtype=torch.int64,device=device).flatten()
            positions=torch.arange(128,device=device).repeat(Bmax)
            slots=(torch.arange(Bmax,device=device)[:,None]*blocks*page+
                   torch.arange(128,device=device)[None,:]).reshape(-1).int()
            cu=torch.arange(Bmax+1,device=device,dtype=torch.int32)*128
            set_context(True,cu_seqlens_q=cu,cu_seqlens_k=cu,
                        max_seqlen_q=128,max_seqlen_k=128,slot_mapping=slots)
            hidden=model(ids,positions)
            root_logits=model.compute_logits(hidden,True)
            plausible_roots=root_logits.topk(5,dim=-1).indices
            reset_context()
    tree = SimpleNamespace(duet_proxy_total_budget=5, duet_p2_seed_count=5,
                           duet_phase1_k=4, duet_phase2_k=2, duet_tree_c_tensor=3,
                           duet_p2_tree_max_nodes=4, duet_tree_policy='dynamic',
                           greedy_only=True, sampler_x=None, async_fan_out=3)
    def measure(fn):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(a.repeats): fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / a.repeats
    out = dict(model=a.model, dtype=str(dtype), synthetic_prefix_tokens=128,
               width_per_request=5, forwards=2, repeats=a.repeats,
               gpu=torch.cuda.get_device_name(0),cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
               torch_version=torch.__version__,real_prefix=a.real_prefix,
               diverse_prefix=a.diverse_prefix,
               plausible_roots=a.plausible_roots,strict_matmul=a.strict_matmul,
               pad_linear_rows=a.pad_linear_rows,cells=[])
    with torch.inference_mode():
        for B in a.batches:
            arenas = [P2TreeExecutor(model, model.compute_logits, tree, device,
                       page, blocks, cfg.vocab_size, cfg.num_attention_heads,
                       cfg.num_key_value_heads, hd, dtype=dtype,
                       materialize_backbone_logits=False) for _ in range(B)]
            for b, ex in enumerate(arenas):
                ex.prime_capture_inputs(1)
                ex.in_root_tok.copy_(torch.arange(5, device=device) + 100 + b * 10)
                if a.plausible_roots:
                    if not a.real_prefix: raise ValueError('Plausible roots require --real-prefix')
                    ex.in_root_tok.copy_(plausible_roots[b])
                ex.in_root_piv.copy_(torch.tensor([.4,.25,.15,.12,.08], device=device))
                ex.in_prefix_len.fill_(128)
                ex.in_rope_base.fill_(128)
                ex.in_temps.zero_()
                for f in range(ex.F):
                    ex.in_slot[f].copy_(b * blocks * page + 128 + ex.round_offsets[f] + ex.lane_w)
                    ex.wrappers[1][f]._paged_kv_indices_buf.copy_(
                        torch.arange(1 + ex.canvas_extra_pages, dtype=torch.int32, device=device) + b * blocks)
                ex.capture(1)
            def serial():
                for ex in arenas: ex.replay(1)
            serial()
            refs = [(ex.cell_logits.clone(), ex.view_tok.clone(), ex.view_par.clone()) for ex in arenas]
            for ex in arenas: ex.run_once(1)
            serial_eager_diff = max(float((ex.cell_logits-ref[0]).abs().max())
                                    for ex,ref in zip(arenas,refs))
            batch = BatchedTreeExecutor(arenas)
            batch.run_once(1)
            batched_eager_first_diff = max(float((ex.cell_logits[:ex.W]-ref[0][:ex.W]).abs().max())
                                           for ex,ref in zip(arenas,refs))
            batch.capture(1)
            batch.replay(1)
            max_diff = max(float((ex.cell_logits-ref[0]).abs().max()) for ex,ref in zip(arenas,refs))
            first_round_diff = max(float((ex.cell_logits[:ex.W]-ref[0][:ex.W]).abs().max())
                                   for ex,ref in zip(arenas,refs))
            first_round_top3 = [torch.equal(ex.cell_logits[:ex.W].topk(3,-1).indices,
                                           ref[0][:ex.W].topk(3,-1).indices)
                                for ex,ref in zip(arenas,refs)]
            first_logit_a = torch.cat([ref[0][:ex.W] for ex, ref in zip(arenas, refs)])
            first_logit_b = torch.cat([ex.cell_logits[:ex.W] for ex in arenas])
            probability_tv = (first_logit_a.softmax(-1)-first_logit_b.softmax(-1)).abs().sum(-1)/2
            first_top1 = first_logit_a.argmax(-1)==first_logit_b.argmax(-1)
            worst_row, worst_token = divmod(int((first_logit_a-first_logit_b).abs().argmax()), cfg.vocab_size)
            worst_logit = dict(row=worst_row,token=worst_token,
                serial_logit=float(first_logit_a[worst_row,worst_token]),
                batched_logit=float(first_logit_b[worst_row,worst_token]),
                serial_probability=float(first_logit_a[worst_row].softmax(-1)[worst_token]))
            same = all(torch.equal(ex.view_tok,ref[1]) and torch.equal(ex.view_par,ref[2])
                       for ex,ref in zip(arenas,refs))
            serial_ms = measure(serial)
            batched_ms = measure(lambda: batch.replay(1))
            row = dict(batch=B, serial_ms=serial_ms, batched_ms=batched_ms,
                       speedup=serial_ms/batched_ms, same_topology_and_tokens=same,
                       max_logit_difference=max_diff, first_round_max_difference=first_round_diff,
                       first_round_top3_exact=first_round_top3,
                       first_round_top1_agreement=float(first_top1.float().mean()),
                       first_round_probability_tv_mean=float(probability_tv.mean()),
                       first_round_probability_tv_max=float(probability_tv.max()),
                       worst_first_round_logit=worst_logit,
                       serial_eager_diff=serial_eager_diff,
                       batched_eager_first_diff=batched_eager_first_diff)
            if a.hf_reference:
                if not a.real_prefix:
                    raise ValueError('--hf-reference requires --real-prefix')
                from transformers import AutoModelForCausalLM
                # Load after timings. Process roots in small chunks, avoiding
                # a full [batch,prefix,vocabulary] logits allocation.
                hf = AutoModelForCausalLM.from_pretrained(
                    a.model, torch_dtype=torch.float32,
                    attn_implementation='eager').to(device).eval()
                root_ids = torch.cat([ex.in_root_tok for ex in arenas])
                hf_rows = []
                for start in range(0, root_ids.numel(), 5):
                    roots = root_ids[start:start+5]
                    prompt = torch.tensor([prefixes[j//5]
                        for j in range(start,start+roots.numel())],device=device)
                    ids = torch.cat((prompt, roots[:,None]),dim=1)
                    h = hf.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
                    hf_rows.append(hf.lm_head(h).float())
                hf_logits = torch.cat(hf_rows)
                hf_probs = hf_logits.softmax(-1)
                hf_summary = {}
                for label, values in [('serial',first_logit_a),('batched',first_logit_b)]:
                    tv = (values.softmax(-1)-hf_probs).abs().sum(-1)/2
                    chosen = values.argmax(-1)
                    gap = hf_logits.max(-1).values-hf_logits.gather(1,chosen[:,None]).squeeze(1)
                    hf_summary[label] = dict(tv_mean=float(tv.mean()),tv_max=float(tv.max()),
                        argmax_agreement=float((chosen==hf_logits.argmax(-1)).float().mean()),
                        chosen_token_reference_logit_deficit_max=float(gap.max()))
                row['hf_float32_reference'] = hf_summary
                del hf, hf_rows, hf_logits, hf_probs
                gc.collect()
                torch.cuda.empty_cache()
            if a.trace_layers:
                traces = {}
                handles = []
                def trace_hook(name):
                    def hook(module, args, value):
                        # Only the first forward (tree roots), before differing
                        # token choices can propagate into later rounds.
                        if name not in traces:
                            values = value if isinstance(value, tuple) else (value,)
                            traces[name] = [t.detach().float().cpu().clone()
                                            for t in values if torch.is_tensor(t)]
                    return hook
                for name, module in model.named_modules():
                    if name.startswith('model.'):
                        handles.append(module.register_forward_hook(trace_hook(name)))
                arenas[0].run_once(1)
                serial_trace = traces
                traces = {}
                batch.run_once(1)
                for handle in handles:
                    handle.remove()
                differences = []
                for name, tensors in serial_trace.items():
                    for ti, t in enumerate(tensors):
                        other = traces[name][ti][:t.shape[0]]
                        diff = (t-other).abs()
                        differences.append(dict(module=name,output=ti,
                            shape=list(t.shape),max_difference=float(diff.max()),
                            rms_difference=float(diff.square().mean().sqrt()),
                            reference_absmax=float(t.abs().max())))
                row['layer_differences'] = differences
            out['cells'].append(row)
            print({k:v for k,v in row.items() if k!='layer_differences'}, flush=True)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(out,indent=2)+'\n')
            del batch, arenas, refs, ex
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
