"""
LayerSkip Early-Exit Distribution Analysis
==========================================
Compare three distribution sources against Llama-3.1-8B final output:
  1. Llama-3.1-8B final layer (= ground truth p_T)
  2. Llama-3.2-1B final layer (= draft model p_D)
  3. LayerSkip-Llama3-8B per-layer early-exit (= p_E^k via logit lens)

For each GSM8K sample, generate 10 tokens greedily from the 8B model,
then compare distributions at each token position.

Metrics per layer:
  - JSD(p_T, p_E^k)  /  KL(p_T || p_E^k)
  - Top-k token overlap: |topk(p_E^k) ∩ topk(p_T)| / k
  - Top-k mass coverage: Σ p_T(v) for v ∈ topk(p_E^k)
  - Top-1 match: argmax(p_E^k) == argmax(p_T)

Draft baseline (scalar per position):
  - Same metrics but p_D (1B final) vs p_T (8B final)
"""

import argparse, json, os, random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

N_TOKENS = 10
N_SAMPLES = 50


# ─── Utilities ─────────────────────────────────────────────────────────────

def js_div(p, q, eps=1e-10):
    p, q = p.clamp(min=eps), q.clamp(min=eps)
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())

def kl_div(p, q, eps=1e-10):
    p = p.clamp(min=0); q = q.clamp(min=eps)
    mask = p > 0
    return float((p[mask] * (p[mask] / q[mask]).log()).sum())

def topk_overlap(p, q, k=5):
    """Token overlap between top-k of p and top-k of q."""
    tp = set(p.topk(k).indices.tolist())
    tq = set(q.topk(k).indices.tolist())
    return len(tp & tq) / k

def topk_mass(p_ref, p_pred, k=5):
    """How much p_ref mass is covered by top-k tokens of p_pred."""
    tk = p_pred.topk(k).indices.tolist()
    return float(p_ref[tk].sum())


def get_final_norm(model):
    if hasattr(model.model, "norm"):  return model.model.norm
    if hasattr(model.model, "ln_f"):  return model.model.ln_f
    raise ValueError(f"Unknown norm: {model.__class__.__name__}")


@torch.no_grad()
def batch_hidden_to_probs(hiddens, norm, lm_head, norm_device):
    """[N, D] -> [N, V] probability distributions.
    Applies norm to all hidden states (for layers 0..N-1, which are pre-norm).
    NOTE: hidden_states[-1] (final layer) is already post-norm in HF models,
    so caller should handle it separately or pass only pre-norm layers here.
    """
    dtype = next(lm_head.parameters()).dtype
    head_device = next(lm_head.parameters()).device
    normed = norm(hiddens.to(norm_device))
    logits = lm_head(normed.to(head_device).to(dtype))
    return F.softmax(logits.float(), dim=-1).cpu()


@torch.no_grad()
def hidden_to_probs_no_norm(hidden, lm_head):
    """For already-normed hidden state (e.g. hidden_states[-1]). [D] -> [V]."""
    dtype = next(lm_head.parameters()).dtype
    head_device = next(lm_head.parameters()).device
    logits = lm_head(hidden.to(head_device).to(dtype).unsqueeze(0)).squeeze()
    return F.softmax(logits.float(), dim=-1).cpu()


def load_model(path, cache_dir=None, device_map="auto"):
    print(f"  Loading: {path}")
    tok = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device_map, cache_dir=cache_dir,
    )
    model.eval()
    norm = get_final_norm(model)
    norm_dev = next(norm.parameters()).device
    embed_dev = model.model.embed_tokens.weight.device
    n_layers = model.config.num_hidden_layers
    print(f"    layers={n_layers}  embed@{embed_dev}  norm@{norm_dev}")
    return model, tok, norm, model.lm_head, norm_dev, embed_dev, n_layers


def load_gsm8k(n, seed=42, cache_dir=None):
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    rng = random.Random(seed)
    return [ds[i]["question"] for i in rng.sample(range(len(ds)), n)]


# ─── Core analysis ─────────────────────────────────────────────────────────

@torch.no_grad()
def get_target_distributions(model, tok, norm, lm_head, norm_dev, embed_dev, n_layers, prompt_ids, n_tokens):
    """
    Greedy-generate n_tokens from the model.
    Returns:
      gen_ids: [n_tokens] generated token ids
      final_probs: list of n_tokens [V] tensors (final layer probs)
      layer_probs: list of n_tokens [n_layers+1, V] tensors (all layers via logit lens)
    """
    input_ids = prompt_ids.to(embed_dev)
    gen_ids = []
    final_probs = []
    layer_probs_list = []

    past_kv = None
    for step in range(n_tokens):
        inp = input_ids if step == 0 else torch.tensor([[gen_ids[-1]]], device=embed_dev)
        out = model(inp, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values

        # Final layer probs
        logits = out.logits[0, -1, :].float()
        p_final = F.softmax(logits, dim=-1).cpu()
        final_probs.append(p_final)

        # All layer probs via logit lens
        # hidden_states[0..n_layers-1] are pre-norm, hidden_states[n_layers] is post-norm
        pre_norm_hidden = torch.stack([hs[0, -1, :] for hs in out.hidden_states[:-1]])  # [n_layers, D]
        pre_norm_p = batch_hidden_to_probs(pre_norm_hidden, norm, lm_head, norm_dev)     # [n_layers, V]
        post_norm_p = hidden_to_probs_no_norm(out.hidden_states[-1][0, -1, :], lm_head)  # [V]
        all_p = torch.cat([pre_norm_p, post_norm_p.unsqueeze(0)], dim=0)  # [n_layers+1, V]
        layer_probs_list.append(all_p)

        tk = p_final.argmax().item()
        gen_ids.append(tk)

    return gen_ids, final_probs, layer_probs_list


@torch.no_grad()
def get_layerskip_distributions(model, tok, norm, lm_head, norm_dev, embed_dev, n_layers, prompt, gen_ids):
    """
    Get per-layer distributions for positions predicting gen_ids[0..N-1].
    Uses KV cache to avoid OOM: prefill prompt (no hidden states), then
    process one token at a time with hidden states.

    Tokens fed:          prompt[:-1]  | prompt[-1] | gen[0] | gen[1] | ... | gen[N-2]
    Predicts:                         | gen[0]     | gen[1] | gen[2] | ... | gen[N-1]
    Hidden states needed:             | yes        | yes    | yes    | ... | yes
    """
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(embed_dev)

    # Prefill prompt[:-1] without hidden states
    if prompt_ids.shape[1] > 1:
        prefill_out = model(prompt_ids[:, :-1], use_cache=True, output_hidden_states=False)
        past_kv = prefill_out.past_key_values
    else:
        past_kv = None

    # Tokens to feed one-by-one: prompt[-1], gen[0], ..., gen[N-2]
    tokens_to_feed = [prompt_ids[0, -1].item()] + list(gen_ids[:-1])

    layer_probs_list = []
    for tid in tokens_to_feed:
        inp = torch.tensor([[tid]], device=embed_dev)
        out = model(inp, past_key_values=past_kv, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values

        pre_norm_hidden = torch.stack([hs[0, -1, :] for hs in out.hidden_states[:-1]])
        pre_norm_p = batch_hidden_to_probs(pre_norm_hidden, norm, lm_head, norm_dev)
        post_norm_p = hidden_to_probs_no_norm(out.hidden_states[-1][0, -1, :], lm_head)
        all_p = torch.cat([pre_norm_p, post_norm_p.unsqueeze(0)], dim=0)
        layer_probs_list.append(all_p)

    return layer_probs_list


@torch.no_grad()
def get_draft_distributions(model, tok, embed_dev, context_ids, gen_ids):
    """
    Feed context + gen_ids to draft model, get p_D at each generated position.
    Returns list of n_tokens [V] tensors.
    """
    full = torch.cat([
        context_ids,
        torch.tensor([gen_ids], dtype=torch.long),
    ], dim=1).to(embed_dev)

    out = model(full)
    L = context_ids.shape[1]
    draft_probs = []
    for i in range(len(gen_ids)):
        logits = out.logits[0, L + i - 1, :].float()  # predicts position L+i
        p = F.softmax(logits, dim=-1).cpu()
        draft_probs.append(p)
    return draft_probs


# ─── Main ──────────────────────────────────────────────────────────────────

def _gc_gpu():
    """Force garbage collection and free GPU memory."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def compute_layer_metrics(p_T, p_E):
    """Compute all metrics for a single (p_T, p_E) pair."""
    return {
        "jsd": js_div(p_T, p_E),
        "kl": kl_div(p_T, p_E),
        "top1": float(p_E.argmax().item() == p_T.argmax().item()),
        "top5_olap": topk_overlap(p_T, p_E, 5),
        "top10_olap": topk_overlap(p_T, p_E, 10),
        "top5_mass": topk_mass(p_T, p_E, 5),
        "top10_mass": topk_mass(p_T, p_E, 10),
    }


def run(args):
    cache_dir = args.cache_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    questions = load_gsm8k(args.n_samples, cache_dir=cache_dir)
    draft_paths = args.draft if args.draft else []

    # ── Phase 1: Target model ─────────────────────────────────────────────
    print("="*60)
    print("Phase 1: Loading target model...")
    t_model, t_tok, t_norm, t_head, t_norm_dev, t_embed_dev, t_nlayers = \
        load_model(args.target, cache_dir=cache_dir)

    # Store per-sample: (gen_ids, final_probs, layer_probs, prompt)
    sample_cache = []
    for q in tqdm(questions, desc="Target samples"):
        prompt = f"Question: {q}\nAnswer: "
        prompt_ids = t_tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        gen_ids, t_final_probs, t_layer_probs = get_target_distributions(
            t_model, t_tok, t_norm, t_head, t_norm_dev, t_embed_dev, t_nlayers,
            prompt_ids, N_TOKENS,
        )
        sample_cache.append((gen_ids, t_final_probs, t_layer_probs, prompt))

    print(f"Target done. Unloading to free GPU memory...")
    del t_model, t_tok, t_norm, t_head
    _gc_gpu()

    # ── Phase 2: LayerSkip model ──────────────────────────────────────────
    print("="*60)
    print("Phase 2: Loading LayerSkip model...")
    ls_model, ls_tok, ls_norm, ls_head, ls_norm_dev, ls_embed_dev, ls_nlayers = \
        load_model(args.layerskip, cache_dir=cache_dir)

    print(f"\nTarget: {t_nlayers} layers, LayerSkip: {ls_nlayers} layers")

    # Accumulators
    n_pos = 0

    t_jsd_sum     = np.zeros(t_nlayers + 1)
    t_kl_sum      = np.zeros(t_nlayers + 1)
    t_top1_sum    = np.zeros(t_nlayers + 1)
    t_top5_olap_sum  = np.zeros(t_nlayers + 1)
    t_top10_olap_sum = np.zeros(t_nlayers + 1)
    t_top5_mass_sum  = np.zeros(t_nlayers + 1)
    t_top10_mass_sum = np.zeros(t_nlayers + 1)

    ls_jsd_sum    = np.zeros(ls_nlayers + 1)
    ls_kl_sum     = np.zeros(ls_nlayers + 1)
    ls_top1_sum   = np.zeros(ls_nlayers + 1)
    ls_top5_olap_sum  = np.zeros(ls_nlayers + 1)
    ls_top10_olap_sum = np.zeros(ls_nlayers + 1)
    ls_top5_mass_sum  = np.zeros(ls_nlayers + 1)
    ls_top10_mass_sum = np.zeros(ls_nlayers + 1)

    for idx, (gen_ids, t_final_probs, t_layer_probs, prompt) in enumerate(
            tqdm(sample_cache, desc="LayerSkip samples")):

        # Get LayerSkip per-layer distributions using KV cache (avoids OOM)
        ls_layer_probs = get_layerskip_distributions(
            ls_model, ls_tok, ls_norm, ls_head, ls_norm_dev, ls_embed_dev,
            ls_nlayers, prompt, gen_ids,
        )

        for i in range(N_TOKENS):
            p_T = t_final_probs[i]
            n_pos += 1

            # Target self early-exit (from cached data)
            t_all_p = t_layer_probs[i]
            for k in range(t_nlayers + 1):
                m = compute_layer_metrics(p_T, t_all_p[k])
                t_jsd_sum[k]       += m["jsd"]
                t_kl_sum[k]        += m["kl"]
                t_top1_sum[k]      += m["top1"]
                t_top5_olap_sum[k] += m["top5_olap"]
                t_top10_olap_sum[k]+= m["top10_olap"]
                t_top5_mass_sum[k] += m["top5_mass"]
                t_top10_mass_sum[k]+= m["top10_mass"]

            # LayerSkip per-layer (from KV-cache based computation)
            ls_all_p = ls_layer_probs[i]
            for k in range(ls_nlayers + 1):
                m = compute_layer_metrics(p_T, ls_all_p[k])
                ls_jsd_sum[k]       += m["jsd"]
                ls_kl_sum[k]        += m["kl"]
                ls_top1_sum[k]      += m["top1"]
                ls_top5_olap_sum[k] += m["top5_olap"]
                ls_top10_olap_sum[k]+= m["top10_olap"]
                ls_top5_mass_sum[k] += m["top5_mass"]
                ls_top10_mass_sum[k]+= m["top10_mass"]

    print(f"LayerSkip done. Unloading...")
    del ls_model, ls_tok, ls_norm, ls_head
    _gc_gpu()

    # ── Phase 3: Draft models (optional, sequential) ────────────────────
    draft_results = []  # list of {name, n_layers, metrics}
    for di, draft_path in enumerate(draft_paths):
        print("="*60)
        print(f"Phase 3-{di}: Loading draft model: {draft_path}")
        d_model, d_tok, _, _, _, d_embed_dev, d_nlayers = \
            load_model(draft_path, cache_dir=cache_dir)
        d_name = os.path.basename(draft_path.rstrip("/"))

        d_jsd_sum = 0.0; d_kl_sum = 0.0; d_top1_sum = 0.0
        d_top5_olap_sum_s = 0.0; d_top10_olap_sum_s = 0.0
        d_top5_mass_sum_s = 0.0; d_top10_mass_sum_s = 0.0

        for idx, (gen_ids, t_final_probs, _, prompt) in enumerate(
                tqdm(sample_cache, desc=f"Draft[{d_name}]")):
            d_prompt_ids = d_tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
            d_probs = get_draft_distributions(d_model, d_tok, d_embed_dev, d_prompt_ids, gen_ids)
            for i in range(N_TOKENS):
                p_T = t_final_probs[i]
                p_D = d_probs[i]
                d_jsd_sum += js_div(p_T, p_D)
                d_kl_sum  += kl_div(p_T, p_D)
                d_top1_sum += float(p_D.argmax().item() == p_T.argmax().item())
                d_top5_olap_sum_s  += topk_overlap(p_T, p_D, 5)
                d_top10_olap_sum_s += topk_overlap(p_T, p_D, 10)
                d_top5_mass_sum_s  += topk_mass(p_T, p_D, 5)
                d_top10_mass_sum_s += topk_mass(p_T, p_D, 10)

        draft_results.append({
            "path": draft_path,
            "name": d_name,
            "n_layers": d_nlayers,
            "jsd":           d_jsd_sum / n_pos,
            "kl":            d_kl_sum / n_pos,
            "top1_match":    d_top1_sum / n_pos,
            "top5_overlap":  d_top5_olap_sum_s / n_pos,
            "top10_overlap": d_top10_olap_sum_s / n_pos,
            "top5_mass":     d_top5_mass_sum_s / n_pos,
            "top10_mass":    d_top10_mass_sum_s / n_pos,
        })
        print(f"  Done. JSD={d_jsd_sum/n_pos:.4f}")
        del d_model, d_tok
        _gc_gpu()

    # ── Finalize ──────────────────────────────────────────────────────────
    n = max(n_pos, 1)

    result = {
        "target": args.target,
        "layerskip": args.layerskip,
        "drafts": [dp for dp in draft_paths],
        "n_samples": args.n_samples,
        "n_tokens": N_TOKENS,
        "n_positions": n_pos,
        "target_n_layers": t_nlayers,
        "layerskip_n_layers": ls_nlayers,
        "target_early_exit": {
            "jsd":           (t_jsd_sum / n).tolist(),
            "kl":            (t_kl_sum / n).tolist(),
            "top1_match":    (t_top1_sum / n).tolist(),
            "top5_overlap":  (t_top5_olap_sum / n).tolist(),
            "top10_overlap": (t_top10_olap_sum / n).tolist(),
            "top5_mass":     (t_top5_mass_sum / n).tolist(),
            "top10_mass":    (t_top10_mass_sum / n).tolist(),
        },
        "layerskip_early_exit": {
            "jsd":           (ls_jsd_sum / n).tolist(),
            "kl":            (ls_kl_sum / n).tolist(),
            "top1_match":    (ls_top1_sum / n).tolist(),
            "top5_overlap":  (ls_top5_olap_sum / n).tolist(),
            "top10_overlap": (ls_top10_olap_sum / n).tolist(),
            "top5_mass":     (ls_top5_mass_sum / n).tolist(),
            "top10_mass":    (ls_top10_mass_sum / n).tolist(),
        },
        "draft_baselines": draft_results if draft_results else None,
    }

    json_path = os.path.join(out_dir, "layerskip_analysis.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {json_path}")

    plot_results(result, out_dir)
    print_summary(result)
    return result


def plot_results(r, out_dir):
    t_ee  = r["target_early_exit"]
    ls_ee = r["layerskip_early_exit"]
    drafts = r.get("draft_baselines") or []
    t_nl  = r["target_n_layers"]
    ls_nl = r["layerskip_n_layers"]
    t_name  = os.path.basename(r["target"].rstrip("/"))
    ls_name = os.path.basename(r["layerskip"].rstrip("/"))

    draft_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    draft_styles = ["--", ":", "-.", (0,(3,1,1,1)), (0,(5,2))]

    metrics = [
        ("jsd",          "JSD ↓",               (0, 0.75)),
        ("kl",           "KL Divergence ↓",      None),
        ("top1_match",   "Top-1 Match Rate ↑",   (-0.02, 1.05)),
        ("top5_overlap", "Top-5 Token Overlap ↑", (-0.02, 1.05)),
        ("top5_mass",    "Top-5 Mass Coverage ↑", (-0.02, 1.05)),
        ("top10_overlap","Top-10 Token Overlap ↑",(-0.02, 1.05)),
        ("top10_mass",   "Top-10 Mass Coverage ↑",(-0.02, 1.05)),
    ]

    for key, ylabel, ylim in metrics:
        fig, ax = plt.subplots(figsize=(12, 5))

        t_vals = t_ee[key]
        t_layers = np.arange(len(t_vals))
        ax.plot(t_layers / t_nl * 100, t_vals,
                label=f"{t_name} early-exit ({t_nl}L)", color="#1f77b4", linewidth=1.5)

        ls_vals = ls_ee[key]
        ls_layers = np.arange(len(ls_vals))
        ax.plot(ls_layers / ls_nl * 100, ls_vals,
                label=f"{ls_name} early-exit ({ls_nl}L)", color="#d62728", linewidth=1.5)

        for di, d in enumerate(drafts):
            d_val = d[key]
            ax.axhline(d_val, color=draft_colors[di % len(draft_colors)],
                        linewidth=1.5, linestyle=draft_styles[di % len(draft_styles)],
                        label=f"{d['name']} ({d['n_layers']}L) = {d_val:.4f}")

        ax.set_xlabel("Layer Depth (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Early-Exit vs {t_name} Final — {ylabel}")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"layerskip_{key}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Combined overview: JSD + top5_overlap + top5_mass in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, ylabel, ylim) in zip(axes, [
        ("jsd", "JSD ↓", (0, 0.75)),
        ("top5_overlap", "Top-5 Overlap ↑", (-0.02, 1.05)),
        ("top5_mass", "Top-5 Mass ↑", (-0.02, 1.05)),
    ]):
        t_vals = t_ee[key]
        ls_vals = ls_ee[key]
        ax.plot(np.arange(len(t_vals)) / t_nl * 100, t_vals,
                label=t_name, color="#1f77b4", linewidth=1.5)
        ax.plot(np.arange(len(ls_vals)) / ls_nl * 100, ls_vals,
                label=ls_name, color="#d62728", linewidth=1.5)
        for di, d in enumerate(drafts):
            ax.axhline(d[key], color=draft_colors[di % len(draft_colors)],
                        linewidth=1.5, linestyle=draft_styles[di % len(draft_styles)],
                        label=f"{d['name']} = {d[key]:.3f}")
        ax.set_xlabel("Layer Depth (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"LayerSkip vs Standard Early-Exit (all vs {t_name} final)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layerskip_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {out_dir}/")


def print_summary(r):
    t_ee  = r["target_early_exit"]
    ls_ee = r["layerskip_early_exit"]
    drafts = r.get("draft_baselines") or []
    t_nl  = r["target_n_layers"]
    ls_nl = r["layerskip_n_layers"]
    t_name  = os.path.basename(r["target"].rstrip("/"))
    ls_name = os.path.basename(r["layerskip"].rstrip("/"))

    print(f"\n{'='*90}")
    print(f"SUMMARY  ({r['n_samples']} samples × {r['n_tokens']} tokens = {r['n_positions']} positions)")
    print(f"  Target: {r['target']} ({t_nl}L)")
    print(f"  LayerSkip: {r['layerskip']} ({ls_nl}L)")
    for d in drafts:
        print(f"  Draft: {d['path']} ({d['n_layers']}L)")
    print(f"{'='*90}")

    print(f"\n{'Source':<30} | {'JSD':>8} | {'KL':>8} | {'Top1':>6} | {'T5olap':>7} | {'T5mass':>7} | {'T10olap':>8} | {'T10mass':>8}")
    print("-" * 95)

    i = -2
    print(f"{t_name+' L[-2]':<30} | {t_ee['jsd'][i]:>8.4f} | {t_ee['kl'][i]:>8.4f} | "
          f"{t_ee['top1_match'][i]:>6.3f} | {t_ee['top5_overlap'][i]:>7.3f} | "
          f"{t_ee['top5_mass'][i]:>7.3f} | {t_ee['top10_overlap'][i]:>8.3f} | {t_ee['top10_mass'][i]:>8.3f}")

    print(f"{ls_name+' L[-2]':<30} | {ls_ee['jsd'][i]:>8.4f} | {ls_ee['kl'][i]:>8.4f} | "
          f"{ls_ee['top1_match'][i]:>6.3f} | {ls_ee['top5_overlap'][i]:>7.3f} | "
          f"{ls_ee['top5_mass'][i]:>7.3f} | {ls_ee['top10_overlap'][i]:>8.3f} | {ls_ee['top10_mass'][i]:>8.3f}")

    for d in drafts:
        label = f"{d['name']} ({d['n_layers']}L)"
        print(f"{label:<30} | {d['jsd']:>8.4f} | {d['kl']:>8.4f} | "
              f"{d['top1_match']:>6.3f} | {d['top5_overlap']:>7.3f} | "
              f"{d['top5_mass']:>7.3f} | {d['top10_overlap']:>8.3f} | {d['top10_mass']:>8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--layerskip", default="facebook/layerskip-llama3-8B")
    parser.add_argument("--draft", nargs="*", default=None,
                        help="Optional draft model(s). Pass multiple for comparison.")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", default="/home/chokwans99/Parallel_SD/layer_skip")
    parser.add_argument("--cache_dir", default="/data2/shared/huggingface_cache")
    args = parser.parse_args()
    run(args)
