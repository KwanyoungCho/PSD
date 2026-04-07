"""
Correction Distribution Feasibility Analysis
============================================
핵심 질문: SD에서 reject된 위치의 correction distribution을
target model early-exit layer로 얼마나 잘 근사할 수 있는가?

  r_true(v)  = [p^T_final(v) - p^D(v)]_+   normalized  (실제 correction)
  r_hat_k(v) = [p^E_k(v)    - p^D(v)]_+   normalized  (layer k 근사)

Checkpoint (target greedy 생성 기준):
  cp0:   context = question  (prefill 직후)
  cp128: context = question + target[:128]
  cp256: context = question + target[:256]
  cp512: context = question + target[:512]
  answer가 cp보다 짧으면 해당 cp skip.

Draft window size = 10 (greedy 생성, 각 위치 p^D 기록)
모든 reject 위치에서 correction 분석 (probabilistic SD rejection).

Baseline 비교:
  draft_topk:  draft p^D 자체의 top-k 후보가 correction token을 포함하는지
  proxy_topk:  early-exit residual [p^E_k - p^D]_+ 의 top-k 후보가 포함하는지
"""

import argparse, json, os, random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

WINDOW_SIZE = 10
CHECKPOINTS = [0, 128, 256, 512]
CP_LABELS   = ["cp0", "cp128", "cp256", "cp512"]


# ─── Model utilities ────────────────────────────────────────────────────────

def get_final_norm(model):
    if hasattr(model.model, "norm"):  return model.model.norm
    if hasattr(model.model, "ln_f"): return model.model.ln_f
    raise ValueError(f"Unknown final norm: {model.__class__.__name__}")


@torch.no_grad()
def hidden_to_probs(hidden, norm, lm_head, norm_device):
    """hidden [hidden_dim] → prob [vocab_size] CPU float32."""
    dtype  = next(lm_head.parameters()).dtype
    normed = norm(hidden.to(norm_device).float().unsqueeze(0))
    logits = lm_head(normed.to(dtype)).squeeze()
    return F.softmax(logits.float(), dim=-1).cpu()


@torch.no_grad()
def batch_hidden_to_probs(hiddens, norm, lm_head, norm_device):
    """hiddens [N, hidden_dim] → probs [N, vocab_size] CPU float32.
    Single GPU sync instead of N sequential syncs."""
    dtype  = next(lm_head.parameters()).dtype
    normed = norm(hiddens.to(norm_device).float())
    logits = lm_head(normed.to(dtype))
    return F.softmax(logits.float(), dim=-1).cpu()


def js_div(p, q, eps=1e-10):
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())


def kl_div(p, q, eps=1e-10):
    """KL(p || q) — forward KL from true distribution p to proxy q."""
    p = p.clamp(min=0)
    q = q.clamp(min=eps)
    mask = p > 0
    return float((p[mask] * (p[mask] / q[mask]).log()).sum())


def residual_dist(p, q, eps=1e-10):
    """[p - q]_+, normalized. Returns None if zero-mass."""
    r = (p - q).clamp(min=0)
    s = r.sum().item()
    return (r / s) if s > eps else None


def load_model(path):
    print(f"  Loading: {path}")
    tok   = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    norm      = get_final_norm(model)
    norm_dev  = next(norm.parameters()).device
    embed_dev = model.model.embed_tokens.weight.device
    n_layers  = model.config.num_hidden_layers
    print(f"  Layers={n_layers}  embed@{embed_dev}  norm@{norm_dev}")
    return model, tok, norm, model.lm_head, norm_dev, embed_dev, n_layers


# ─── Dataset ────────────────────────────────────────────────────────────────

def load_gsm8k(n, seed=42, cache_dir=None):
    ds  = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    rng = random.Random(seed)
    return [(ds[i]["question"], ds[i]["answer"]) for i in rng.sample(range(len(ds)), n)]


# ─── Generation helpers ─────────────────────────────────────────────────────

@torch.no_grad()
def generate_target(model, embed_dev, input_ids, max_new=512):
    """Greedy generation up to max_new tokens. Returns new tokens [1, N] on CPU."""
    out = model.generate(
        input_ids.to(embed_dev),
        max_new_tokens=max_new,
        do_sample=False,
    )
    return out[:, input_ids.shape[1]:].cpu()


@torch.no_grad()
def generate_draft_window(model, embed_dev, context_ids, window_size):
    """
    Greedy draft generation of window_size tokens with KV cache.
    Returns:
      draft_ids   [window_size] LongTensor
      draft_probs list of window_size CPU float32 tensors [vocab_size]
    """
    ids, probs = [], []
    cur = context_ids.to(embed_dev)
    past_kv = None
    for step in range(window_size):
        inp = cur if step == 0 else torch.tensor([[ids[-1]]], device=embed_dev)
        out = model(inp, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :].float()
        p  = F.softmax(logits, dim=-1).cpu()
        tk = p.argmax().item()
        ids.append(tk)
        probs.append(p)
    return torch.tensor(ids), probs


# ─── Core analysis ──────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_window(
    t_model, t_norm, t_head, t_norm_dev, t_embed_dev,
    context_ids,   # [1, L] CPU
    draft_ids,     # [W]    LongTensor CPU
    draft_probs,   # list[W] of [vocab_size] CPU float32
    n_layers,
):
    """
    Runs target model on context + draft window.
    Uses probabilistic SD rejection: at each position, reject with
    probability  1 - min(1, p_T(y)/p_D(y)).
    Analyzes ALL rejected positions (not just the first).

    Indexing note:
      hidden_states[k][0, L+i-1, :] → lm_head → P(draft_i | context, draft_0..draft_{i-1})
      For i=0: uses last context position (L-1).

    Returns list of per-position result dicts, or empty list if no reject.
    """
    L    = context_ids.shape[1]
    W    = len(draft_ids)
    full = torch.cat([context_ids, draft_ids.unsqueeze(0)], dim=1).to(t_embed_dev)
    out  = t_model(full, output_hidden_states=True)
    # out.hidden_states: tuple of (n_layers+1) x [1, L+W, D]

    results = []
    for i in range(W):
        hs_pos = L + i - 1   # hidden state position that predicts draft[i]
        p_T = hidden_to_probs(
            out.hidden_states[-1][0, hs_pos, :], t_norm, t_head, t_norm_dev
        )
        p_D = draft_probs[i]
        y_i = draft_ids[i].item()

        # ── Probabilistic SD rejection ──
        ratio = p_T[y_i].item() / max(p_D[y_i].item(), 1e-10)
        accept_prob = min(1.0, ratio)
        if random.random() < accept_prob:
            continue  # accepted

        # ── Rejected at position i: compute correction ──
        r_true = residual_dist(p_T, p_D)
        if r_true is None:
            break  # no correction mass → stop window

        true_top1  = r_true.argmax().item()
        true_top5  = set(r_true.topk(5).indices.tolist())

        # ── Draft-only baseline: does p_D top-k already cover correction? ──
        # Exclude y_i (the rejected token) from draft candidates
        p_D_masked = p_D.clone()
        p_D_masked[y_i] = 0.0
        draft_top5  = set(p_D_masked.topk(5).indices.tolist())
        draft_top10 = set(p_D_masked.topk(10).indices.tolist())
        draft_baseline = {
            "top1_hit": p_D_masked.argmax().item() == true_top1,
            "top5_hit": true_top1 in draft_top5,
            "top10_hit": true_top1 in draft_top10,
            "top5_recall":  len(true_top5 & draft_top5) / len(true_top5),
            "top10_recall": len(true_top5 & draft_top10) / len(true_top5),
        }

        # ── Per-layer early-exit proxy (batched) ──
        all_hidden = torch.stack([hs[0, hs_pos, :] for hs in out.hidden_states])  # [n_layers+1, D]
        all_probs  = batch_hidden_to_probs(all_hidden, t_norm, t_head, t_norm_dev)  # [n_layers+1, V]

        # ── Raw distribution metrics: early-exit p_E vs final p_T ──
        final_top5  = set(p_T.topk(5).indices.tolist())
        final_top10 = set(p_T.topk(10).indices.tolist())

        # ── Per-layer metrics ──
        jsd_list  = []
        kl_list   = []
        top1_list = []
        top5_list = []
        top5_recall_list = []
        # Raw distribution divergence (p_E vs p_T, not residual)
        raw_jsd_list = []
        raw_kl_list  = []
        # Mirror-SD style top-k mass overlap
        topk_overlap_5_list  = []   # |top5(p_E) ∩ top5(p_T)| / 5
        topk_overlap_10_list = []   # |top10(p_E) ∩ top10(p_T)| / 10
        topk_mass_5_list  = []      # Σ p_T(v) for v in top5(p_E)
        topk_mass_10_list = []      # Σ p_T(v) for v in top10(p_E)

        for k in range(all_probs.shape[0]):
            p_E   = all_probs[k]

            # Raw distribution: early-exit vs final
            raw_jsd_list.append(js_div(p_T, p_E))
            raw_kl_list.append(kl_div(p_T, p_E))

            # Mirror-SD top-k mass overlap
            ek_top5  = set(p_E.topk(5).indices.tolist())
            ek_top10 = set(p_E.topk(10).indices.tolist())
            topk_overlap_5_list.append(len(ek_top5 & final_top5) / 5)
            topk_overlap_10_list.append(len(ek_top10 & final_top10) / 10)
            topk_mass_5_list.append(float(p_T[list(ek_top5)].sum()))
            topk_mass_10_list.append(float(p_T[list(ek_top10)].sum()))

            # Residual distribution: correction proxy
            r_hat = residual_dist(p_E, p_D)
            if r_hat is None:
                jsd_list.append(float("nan"))
                kl_list.append(float("nan"))
                top1_list.append(False)
                top5_list.append(False)
                top5_recall_list.append(0.0)
            else:
                jsd_list.append(js_div(r_true, r_hat))
                kl_list.append(kl_div(r_true, r_hat))
                rhat_top5 = set(r_hat.topk(5).indices.tolist())
                top1_list.append(r_hat.argmax().item() == true_top1)
                top5_list.append(true_top1 in rhat_top5)
                top5_recall_list.append(len(true_top5 & rhat_top5) / len(true_top5))

        results.append({
            "reject_pos":     i,
            "accept_prob":    accept_prob,
            # Correction (residual) distribution metrics
            "jsd_layers":     jsd_list,
            "kl_layers":      kl_list,
            "top1_match":     top1_list,
            "top5_cover":     top5_list,
            "top5_recall":    top5_recall_list,
            "draft_baseline": draft_baseline,
            # Raw distribution metrics (p_E vs p_T)
            "raw_jsd_layers": raw_jsd_list,
            "raw_kl_layers":  raw_kl_list,
            # Mirror-SD top-k overlap
            "topk_overlap_5":  topk_overlap_5_list,
            "topk_overlap_10": topk_overlap_10_list,
            "topk_mass_5":     topk_mass_5_list,
            "topk_mass_10":    topk_mass_10_list,
        })
        break  # SD stops at first reject

    return results


# ─── Main experiment ────────────────────────────────────────────────────────

def run(target_path, draft_path, n_samples, output_dir, cache_dir, suffix=""):
    os.makedirs(output_dir, exist_ok=True)

    t_model, t_tok, t_norm, t_head, t_norm_dev, t_embed_dev, n_layers = load_model(target_path)
    d_model, d_tok, _,      _,      _,           d_embed_dev, _        = load_model(draft_path)

    print(f"\nLoading GSM8K ({n_samples} samples)...")
    qa_pairs = load_gsm8k(n_samples, cache_dir=cache_dir)

    def make_acc(nl):
        return {
            "jsd_sum":    np.zeros(nl + 1),
            "jsd_cnt":    np.zeros(nl + 1),
            "kl_sum":     np.zeros(nl + 1),
            "kl_cnt":     np.zeros(nl + 1),
            "top1_sum":   np.zeros(nl + 1),
            "top5_sum":   np.zeros(nl + 1),
            "top5_recall_sum": np.zeros(nl + 1),
            # Raw distribution (p_E vs p_T)
            "raw_jsd_sum": np.zeros(nl + 1),
            "raw_kl_sum":  np.zeros(nl + 1),
            "raw_cnt":     np.zeros(nl + 1),
            # Mirror-SD top-k overlap
            "topk_overlap_5_sum":  np.zeros(nl + 1),
            "topk_overlap_10_sum": np.zeros(nl + 1),
            "topk_mass_5_sum":     np.zeros(nl + 1),
            "topk_mass_10_sum":    np.zeros(nl + 1),
            #
            "count":      0,
            "all_accept_count": 0,
            "reject_pos": [],
            "accept_probs": [],
            # Draft baseline accumulators
            "draft_top1_sum":  0.0,
            "draft_top5_sum":  0.0,
            "draft_top10_sum": 0.0,
            "draft_top5_recall_sum":  0.0,
            "draft_top10_recall_sum": 0.0,
        }

    acc = {lbl: make_acc(n_layers) for lbl in CP_LABELS}

    for question, _ in tqdm(qa_pairs, desc="Samples"):
        q_fmt = f"Question: {question}\nAnswer: "
        q_ids = t_tok(q_fmt, return_tensors="pt", add_special_tokens=True).input_ids

        # Target greedy generation (ground truth context for cp128/256/512)
        target_toks = generate_target(t_model, t_embed_dev, q_ids, max_new=512)
        # target_toks: [1, N_generated] on CPU

        for cp, lbl in zip(CHECKPOINTS, CP_LABELS):
            # Skip if not enough generated tokens for this checkpoint
            if cp > 0 and target_toks.shape[1] < cp:
                continue

            ctx = q_ids if cp == 0 else torch.cat([q_ids, target_toks[:, :cp]], dim=1)

            # Draft window
            draft_ids, draft_probs = generate_draft_window(
                d_model, d_embed_dev, ctx, WINDOW_SIZE
            )

            # Target verify + correction distribution analysis
            results = analyze_window(
                t_model, t_norm, t_head, t_norm_dev, t_embed_dev,
                ctx, draft_ids, draft_probs, n_layers,
            )
            d = acc[lbl]
            if not results:
                d["all_accept_count"] += 1
                continue

            for result in results:
                d["count"] += 1
                d["reject_pos"].append(result["reject_pos"])
                d["accept_probs"].append(result["accept_prob"])
                # Draft baseline
                db = result["draft_baseline"]
                d["draft_top1_sum"]  += float(db["top1_hit"])
                d["draft_top5_sum"]  += float(db["top5_hit"])
                d["draft_top10_sum"] += float(db["top10_hit"])
                d["draft_top5_recall_sum"]  += db["top5_recall"]
                d["draft_top10_recall_sum"] += db["top10_recall"]
                # Per-layer
                for k in range(n_layers + 1):
                    jv = result["jsd_layers"][k]
                    if jv == jv:  # not NaN
                        d["jsd_sum"][k] += jv
                        d["jsd_cnt"][k] += 1
                    kv = result["kl_layers"][k]
                    if kv == kv:  # not NaN
                        d["kl_sum"][k] += kv
                        d["kl_cnt"][k] += 1
                    d["top1_sum"][k] += float(result["top1_match"][k])
                    d["top5_sum"][k] += float(result["top5_cover"][k])
                    d["top5_recall_sum"][k] += result["top5_recall"][k]
                    # Raw distribution
                    d["raw_jsd_sum"][k] += result["raw_jsd_layers"][k]
                    d["raw_kl_sum"][k]  += result["raw_kl_layers"][k]
                    d["raw_cnt"][k]     += 1
                    # Mirror-SD top-k
                    d["topk_overlap_5_sum"][k]  += result["topk_overlap_5"][k]
                    d["topk_overlap_10_sum"][k] += result["topk_overlap_10"][k]
                    d["topk_mass_5_sum"][k]     += result["topk_mass_5"][k]
                    d["topk_mass_10_sum"][k]    += result["topk_mass_10"][k]

    # Finalize averages
    for lbl in CP_LABELS:
        d = acc[lbl]
        c = max(d["count"], 1)
        d["jsd_layers"] = np.where(
            d["jsd_cnt"] > 0, d["jsd_sum"] / np.maximum(d["jsd_cnt"], 1), float("nan")
        )
        d["kl_layers"] = np.where(
            d["kl_cnt"] > 0, d["kl_sum"] / np.maximum(d["kl_cnt"], 1), float("nan")
        )
        d["top1_match"]  = d["top1_sum"] / c
        d["top5_cover"]  = d["top5_sum"] / c
        d["top5_recall"] = d["top5_recall_sum"] / c
        # Raw distribution averages
        rc = np.maximum(d["raw_cnt"], 1)
        d["raw_jsd_layers"] = d["raw_jsd_sum"] / rc
        d["raw_kl_layers"]  = d["raw_kl_sum"] / rc
        # Mirror-SD top-k averages
        d["topk_overlap_5"]  = d["topk_overlap_5_sum"] / rc
        d["topk_overlap_10"] = d["topk_overlap_10_sum"] / rc
        d["topk_mass_5"]     = d["topk_mass_5_sum"] / rc
        d["topk_mass_10"]    = d["topk_mass_10_sum"] / rc
        # Draft baseline averages
        d["draft_top1"]  = d["draft_top1_sum"] / c
        d["draft_top5"]  = d["draft_top5_sum"] / c
        d["draft_top10"] = d["draft_top10_sum"] / c
        d["draft_top5_recall"]  = d["draft_top5_recall_sum"] / c
        d["draft_top10_recall"] = d["draft_top10_recall_sum"] / c

    # Save JSON
    target_name = os.path.basename(target_path.rstrip("/"))
    draft_name  = os.path.basename(draft_path.rstrip("/"))
    save = {
        "target": target_path,
        "draft":  draft_path,
        "n_samples":   n_samples,
        "n_layers":    n_layers,
        "window_size": WINDOW_SIZE,
        "data": {
            lbl: {
                # Correction (residual) distribution metrics
                "jsd_layers": acc[lbl]["jsd_layers"].tolist(),
                "kl_layers":  acc[lbl]["kl_layers"].tolist(),
                "top1_match": acc[lbl]["top1_match"].tolist(),
                "top5_cover": acc[lbl]["top5_cover"].tolist(),
                "top5_recall": acc[lbl]["top5_recall"].tolist(),
                # Raw distribution (p_E vs p_T) metrics
                "raw_jsd_layers": acc[lbl]["raw_jsd_layers"].tolist(),
                "raw_kl_layers":  acc[lbl]["raw_kl_layers"].tolist(),
                # Mirror-SD top-k overlap
                "topk_overlap_5":  acc[lbl]["topk_overlap_5"].tolist(),
                "topk_overlap_10": acc[lbl]["topk_overlap_10"].tolist(),
                "topk_mass_5":     acc[lbl]["topk_mass_5"].tolist(),
                "topk_mass_10":    acc[lbl]["topk_mass_10"].tolist(),
                "count":      acc[lbl]["count"],
                "all_accept_count": acc[lbl]["all_accept_count"],
                "mean_first_reject": (
                    float(np.mean(acc[lbl]["reject_pos"]))
                    if acc[lbl]["reject_pos"] else None
                ),
                "mean_accept_prob": (
                    float(np.mean(acc[lbl]["accept_probs"]))
                    if acc[lbl]["accept_probs"] else None
                ),
                "reject_pos_dist": {
                    str(k): acc[lbl]["reject_pos"].count(k)
                    for k in range(WINDOW_SIZE)
                } if acc[lbl]["reject_pos"] else {},
                "draft_baseline": {
                    "top1_match": acc[lbl]["draft_top1"],
                    "top5_cover": acc[lbl]["draft_top5"],
                    "top10_cover": acc[lbl]["draft_top10"],
                    "top5_recall":  acc[lbl]["draft_top5_recall"],
                    "top10_recall": acc[lbl]["draft_top10_recall"],
                },
            }
            for lbl in CP_LABELS
        },
    }
    json_path = os.path.join(output_dir, f"correction_{target_name}{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {json_path}")

    plot(acc, CP_LABELS, n_layers, target_name, draft_name, output_dir, suffix=suffix)
    return acc


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot(acc, CP_LABELS, n_layers, target_name, draft_name, output_dir, suffix=""):
    layers = np.arange(n_layers + 1)
    n_rows = 9
    fig, axes = plt.subplots(n_rows, len(CP_LABELS), figsize=(22, n_rows * 4))
    fig.suptitle(
        f"Correction Distribution Feasibility  (window={WINDOW_SIZE}, probabilistic SD)\n"
        f"Target: {target_name}  |  Draft: {draft_name}",
        fontsize=13, fontweight="bold",
    )

    rows = [
        # Correction (residual) distribution metrics
        ("jsd_layers",  "JSD(r_true, r̂_k) ↓",    (0.0, 0.75)),
        ("kl_layers",   "KL(r_true ∥ r̂_k) ↓",    None),
        ("top1_match",  "Top-1 Match Rate ↑",      (-0.05, 1.05)),
        ("top5_cover",  "Top-5 Coverage ↑",         (-0.05, 1.05)),
        ("top5_recall", "Top-5 Recall ↑",           (-0.05, 1.05)),
        # Raw distribution (p_E vs p_T) metrics
        ("raw_jsd_layers", "JSD(p_T, p_E) ↓",      (0.0, 0.75)),
        ("raw_kl_layers",  "KL(p_T ∥ p_E) ↓",      None),
        # Mirror-SD top-k overlap
        ("topk_overlap_5",  "Top-5 Overlap ↑",      (-0.05, 1.05)),
        ("topk_mass_5",     "Top-5 Mass Coverage ↑", (-0.05, 1.05)),
    ]
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#6a4c93", "#1982c4"]
    n_cp = len(CP_LABELS)

    for row, (key, ylabel, ylim) in enumerate(rows):
        for col, lbl in enumerate(CP_LABELS):
            ax = axes[row][col] if n_cp > 1 else axes[row]
            d  = acc[lbl]
            if d["count"] == 0:
                ax.text(0.5, 0.5, "no data\n(skipped)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
                if row == 0:
                    ax.set_title(lbl, fontsize=10)
                continue

            vals = np.array(d[key], dtype=float)
            ax.plot(layers, vals, lw=1.5, color=colors[col], label="proxy")
            # Final layer dashed reference
            final_val = vals[-1] if not np.isnan(vals[-1]) else np.nanmin(vals)
            ax.axhline(final_val, color="gray", lw=0.8, ls="--", alpha=0.6,
                       label=f"final={final_val:.3f}")

            # Draft baseline horizontal line (for top1/top5/recall rows)
            baseline_map = {
                "top1_match": "draft_top1",
                "top5_cover": "draft_top5",
                "top5_recall": "draft_top5_recall",
            }
            if key in baseline_map:
                bl = d[baseline_map[key]]
                ax.axhline(bl, color="red", lw=1.2, ls=":", alpha=0.8,
                           label=f"draft={bl:.3f}")

            if row == 0:
                mr = np.mean(d["reject_pos"]) if d["reject_pos"] else float("nan")
                ax.set_title(f"{lbl}  (n={d['count']}, rej@{mr:.1f})", fontsize=9)

            ax.set_ylabel(ylabel if col == 0 else "", fontsize=8)
            ax.set_xlabel("Layer" if row == len(rows) - 1 else "", fontsize=8)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_xlim(0, n_layers)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right" if row == 0 else "lower right")

    plt.tight_layout()
    p = os.path.join(output_dir, f"correction_{target_name}{suffix}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {p}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",      required=True)
    parser.add_argument("--draft",       required=True)
    parser.add_argument("--n_samples",   type=int, default=200)
    parser.add_argument("--output_dir",  default="/home/chokwans99/Parallel_SD/results")
    parser.add_argument("--cache_dir",   default="/data2/shared/huggingface_cache")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None,
                        help="Override checkpoint positions. e.g. --checkpoints 0 1")
    parser.add_argument("--suffix",      type=str, default="",
                        help="Suffix for output filename. e.g. --suffix _cp1")
    args = parser.parse_args()

    if args.checkpoints is not None:
        global CHECKPOINTS, CP_LABELS
        CHECKPOINTS = args.checkpoints
        CP_LABELS   = [f"cp{c}" for c in CHECKPOINTS]

    acc = run(args.target, args.draft, args.n_samples, args.output_dir, args.cache_dir,
              suffix=args.suffix)

    # Console summary
    print("\n" + "=" * 100)
    print(f"SUMMARY  (window={WINDOW_SIZE}, probabilistic SD rejection)")
    hdr = (f"{'CP':<8} | {'n':>5} | {'skip':>5} | {'mean_rej':>9} "
           f"| {'JSD@-2':>8} | {'KL@-2':>8} | {'prx_t1@-2':>10} | {'prx_t5@-2':>10} "
           f"| {'drf_t1':>7} | {'drf_t5':>7}"
           f"| {'rawJSD':>8} | {'rawKL':>8} | {'tk5olap':>8} | {'tk5mass':>8}")
    print(hdr)
    print("=" * 140)
    n_layers = acc[CP_LABELS[0]]["jsd_layers"].shape[0] - 1
    for lbl in CP_LABELS:
        d = acc[lbl]
        if d["count"] == 0:
            print(f"{lbl:<8} | {'skip':>5}")
            continue
        mr  = np.mean(d["reject_pos"]) if d["reject_pos"] else float("nan")
        jf  = d["jsd_layers"][-2]
        kf  = d["kl_layers"][-2]
        t1f = d["top1_match"][-2]
        t5f = d["top5_cover"][-2]
        dt1 = d["draft_top1"]
        dt5 = d["draft_top5"]
        rjf = d["raw_jsd_layers"][-2]
        rkf = d["raw_kl_layers"][-2]
        to5 = d["topk_overlap_5"][-2]
        tm5 = d["topk_mass_5"][-2]
        print(f"{lbl:<8} | {d['count']:>5} | {d['all_accept_count']:>5} "
              f"| {mr:>9.2f} | {jf:>8.4f} | {kf:>8.4f} | {t1f:>10.3f} | {t5f:>10.3f} "
              f"| {dt1:>7.3f} | {dt5:>7.3f}"
              f"| {rjf:>8.4f} | {rkf:>8.4f} | {to5:>8.3f} | {tm5:>8.3f}")
    print("\n  proxy = early-exit residual top-k  |  drf = draft p_D top-k")
    print("  rawJSD/rawKL = JSD/KL(p_T, p_E)  |  tk5olap/mass = Mirror-SD top-5 overlap/mass")
    print("  @-2 = second-to-last layer")


if __name__ == "__main__":
    main()
