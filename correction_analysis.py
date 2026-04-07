"""
Correction Distribution Feasibility Analysis (70B Compatible)
=============================================================
핵심 질문: SD에서 reject된 위치의 correction distribution을
target model early-exit layer로 얼마나 잘 근사할 수 있는가?

  r_true(v)  = [p^T_final(v) - p^D(v)]_+   normalized  (실제 correction)
  r_hat_k(v) = [p^E_k(v)    - p^D(v)]_+   normalized  (layer k 근사)

다중 draft 모델 지원: target 한번 로드, draft 순차 교체.
KV-cache 기반 verification으로 70B+ 모델 OOM 방지.

Metrics (per layer):
  Correction: JSD, KL, TVD of r_hat vs r_true
  Correction top-k: top1 match, top5 cover, top5 recall
  Raw distribution: JSD, KL, TVD of p_E vs p_T
  Top-k: overlap (5, 10), mass coverage (5, 10)

Draft baseline (scalar):
  Correction: top1/5/10 hit, top5/10 recall
  Raw: JSD, KL, TVD, topk overlap/mass
"""

import argparse, json, os, random, gc
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

WINDOW_SIZE = 10
CHECKPOINTS = [0, 128, 256, 512]
CP_LABELS   = ["cp0", "cp128", "cp256", "cp512"]


# ─── Utilities ─────────────────────────────────────────────────────────────

def js_div(p, q, eps=1e-10):
    p, q = p.clamp(min=eps), q.clamp(min=eps)
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())

def kl_div(p, q, eps=1e-10):
    p = p.clamp(min=0); q = q.clamp(min=eps)
    mask = p > 0
    return float((p[mask] * (p[mask] / q[mask]).log()).sum())

def tvd(p, q):
    """Total Variation Distance."""
    return float(0.5 * (p - q).abs().sum())

def topk_overlap(p, q, k=5):
    tp = set(p.topk(k).indices.tolist())
    tq = set(q.topk(k).indices.tolist())
    return len(tp & tq) / k

def topk_mass(p_ref, p_pred, k=5):
    tk = p_pred.topk(k).indices.tolist()
    return float(p_ref[tk].sum())

def residual_dist(p, q, eps=1e-10):
    r = (p - q).clamp(min=0)
    s = r.sum().item()
    return (r / s) if s > eps else None

def get_final_norm(model):
    if hasattr(model.model, "norm"):  return model.model.norm
    if hasattr(model.model, "ln_f"):  return model.model.ln_f
    raise ValueError(f"Unknown norm: {model.__class__.__name__}")

@torch.no_grad()
def batch_hidden_to_probs(hiddens, norm, lm_head, norm_device):
    """[N, D] → [N, V]. Applies norm then lm_head (for pre-norm hidden states)."""
    dtype = next(lm_head.parameters()).dtype
    head_device = next(lm_head.parameters()).device
    normed = norm(hiddens.to(norm_device))
    logits = lm_head(normed.to(head_device).to(dtype))
    return F.softmax(logits.float(), dim=-1).cpu()

@torch.no_grad()
def hidden_to_probs_no_norm(hidden, lm_head):
    """For already-normed hidden (hidden_states[-1]). [D] → [V]."""
    dtype = next(lm_head.parameters()).dtype
    head_device = next(lm_head.parameters()).device
    logits = lm_head(hidden.to(head_device).to(dtype).unsqueeze(0)).squeeze()
    return F.softmax(logits.float(), dim=-1).cpu()

def load_model(path, cache_dir=None):
    print(f"  Loading: {path}")
    tok = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir,
    )
    model.eval()
    norm = get_final_norm(model)
    norm_dev = next(norm.parameters()).device
    embed_dev = model.model.embed_tokens.weight.device
    n_layers = model.config.num_hidden_layers
    print(f"    layers={n_layers}  embed@{embed_dev}  norm@{norm_dev}")
    return model, tok, norm, model.lm_head, norm_dev, embed_dev, n_layers

def _gc_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def load_gsm8k(n, seed=42, cache_dir=None):
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    rng = random.Random(seed)
    return [(ds[i]["question"], ds[i]["answer"]) for i in rng.sample(range(len(ds)), n)]


# ─── Generation ────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_target(model, embed_dev, input_ids, max_new=512):
    out = model.generate(input_ids.to(embed_dev), max_new_tokens=max_new, do_sample=False)
    return out[:, input_ids.shape[1]:].cpu()

@torch.no_grad()
def generate_draft_window(model, embed_dev, context_ids, window_size):
    ids, probs = [], []
    cur = context_ids.to(embed_dev)
    past_kv = None
    for step in range(window_size):
        inp = cur if step == 0 else torch.tensor([[ids[-1]]], device=embed_dev)
        out = model(inp, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :].float()
        p = F.softmax(logits, dim=-1).cpu()
        ids.append(p.argmax().item())
        probs.append(p)
    return torch.tensor(ids), probs


# ─── KV-cache based verification ──────────────────────────────────────────

@torch.no_grad()
def verify_window_kv(t_model, t_norm, t_head, t_norm_dev, t_embed_dev,
                     context_ids, draft_ids, draft_probs, n_layers):
    """
    Target verifies draft window using KV cache to avoid OOM.
    Prefills context without hidden states, then feeds draft tokens
    one-by-one with hidden states.

    Tokens fed:     ctx[:-1] (prefill) | ctx[-1] | draft[0] | ... | draft[W-2]
    Predicts:                          | draft[0]| draft[1] | ... | draft[W-1]
    Hidden states:                     | yes     | yes      | ... | yes

    Returns list of result dicts (at most 1 for standard SD with break).
    """
    L = context_ids.shape[1]
    W = len(draft_ids)
    ctx = context_ids.to(t_embed_dev)

    # Prefill context[:-1] without hidden states
    if L > 1:
        prefill = t_model(ctx[:, :-1], use_cache=True, output_hidden_states=False)
        past_kv = prefill.past_key_values
    else:
        past_kv = None

    # Feed one-by-one: ctx[-1], draft[0], ..., draft[W-2]
    tokens_to_feed = [ctx[0, -1].item()] + [draft_ids[j].item() for j in range(W - 1)]

    results = []
    for i, tid in enumerate(tokens_to_feed):
        inp = torch.tensor([[tid]], device=t_embed_dev)
        out = t_model(inp, past_key_values=past_kv, use_cache=True,
                      output_hidden_states=True)
        past_kv = out.past_key_values

        # p_T from logits directly (avoids double-norm on hidden_states[-1])
        logits = out.logits[0, -1, :].float()
        p_T = F.softmax(logits, dim=-1).cpu()

        p_D = draft_probs[i]
        y_i = draft_ids[i].item()

        # ── Probabilistic SD rejection ──
        ratio = p_T[y_i].item() / max(p_D[y_i].item(), 1e-10)
        accept_prob = min(1.0, ratio)
        if random.random() < accept_prob:
            continue  # accepted

        # ── Rejected: compute correction ──
        r_true = residual_dist(p_T, p_D)
        if r_true is None:
            break

        true_top1 = r_true.argmax().item()
        true_top5 = set(r_true.topk(5).indices.tolist())

        # ── Draft baseline (correction) ──
        p_D_masked = p_D.clone(); p_D_masked[y_i] = 0.0
        d5  = set(p_D_masked.topk(5).indices.tolist())
        d10 = set(p_D_masked.topk(10).indices.tolist())
        draft_baseline = {
            "top1_hit":     p_D_masked.argmax().item() == true_top1,
            "top5_hit":     true_top1 in d5,
            "top10_hit":    true_top1 in d10,
            "top5_recall":  len(true_top5 & d5)  / len(true_top5),
            "top10_recall": len(true_top5 & d10) / len(true_top5),
        }

        # ── Draft raw baseline (p_D vs p_T) ──
        final_top5  = set(p_T.topk(5).indices.tolist())
        final_top10 = set(p_T.topk(10).indices.tolist())
        d5r  = set(p_D.topk(5).indices.tolist())
        d10r = set(p_D.topk(10).indices.tolist())
        draft_raw = {
            "jsd":  js_div(p_T, p_D),
            "kl":   kl_div(p_T, p_D),
            "tvd":  tvd(p_T, p_D),
            "topk_overlap_5":  len(d5r  & final_top5)  / 5,
            "topk_overlap_10": len(d10r & final_top10) / 10,
            "topk_mass_5":     float(p_T[list(d5r)].sum()),
            "topk_mass_10":    float(p_T[list(d10r)].sum()),
        }

        # ── Per-layer early-exit ──
        pre_norm_hidden = torch.stack([hs[0, -1, :] for hs in out.hidden_states[:-1]])
        pre_norm_probs  = batch_hidden_to_probs(pre_norm_hidden, t_norm, t_head, t_norm_dev)
        post_norm_prob  = hidden_to_probs_no_norm(out.hidden_states[-1][0, -1, :], t_head)
        all_probs = torch.cat([pre_norm_probs, post_norm_prob.unsqueeze(0)], dim=0)

        # ── Per-layer metrics ──
        corr_jsd, corr_kl, corr_tvd = [], [], []
        corr_top1, corr_top5, corr_top5_recall = [], [], []
        raw_jsd, raw_kl, raw_tvd = [], [], []
        tk_olap_5, tk_olap_10, tk_mass_5, tk_mass_10 = [], [], [], []

        for k in range(all_probs.shape[0]):
            p_E = all_probs[k]

            # Raw distribution: p_E vs p_T
            raw_jsd.append(js_div(p_T, p_E))
            raw_kl.append(kl_div(p_T, p_E))
            raw_tvd.append(tvd(p_T, p_E))

            ek5  = set(p_E.topk(5).indices.tolist())
            ek10 = set(p_E.topk(10).indices.tolist())
            tk_olap_5.append(len(ek5  & final_top5)  / 5)
            tk_olap_10.append(len(ek10 & final_top10) / 10)
            tk_mass_5.append(float(p_T[list(ek5)].sum()))
            tk_mass_10.append(float(p_T[list(ek10)].sum()))

            # Correction distribution: r_hat vs r_true
            r_hat = residual_dist(p_E, p_D)
            if r_hat is None:
                corr_jsd.append(float("nan"))
                corr_kl.append(float("nan"))
                corr_tvd.append(float("nan"))
                corr_top1.append(False)
                corr_top5.append(False)
                corr_top5_recall.append(0.0)
            else:
                corr_jsd.append(js_div(r_true, r_hat))
                corr_kl.append(kl_div(r_true, r_hat))
                corr_tvd.append(tvd(r_true, r_hat))
                rh5 = set(r_hat.topk(5).indices.tolist())
                corr_top1.append(r_hat.argmax().item() == true_top1)
                corr_top5.append(true_top1 in rh5)
                corr_top5_recall.append(len(true_top5 & rh5) / len(true_top5))

        results.append({
            "reject_pos": i, "accept_prob": accept_prob,
            "corr_jsd": corr_jsd, "corr_kl": corr_kl, "corr_tvd": corr_tvd,
            "corr_top1": corr_top1, "corr_top5": corr_top5,
            "corr_top5_recall": corr_top5_recall,
            "raw_jsd": raw_jsd, "raw_kl": raw_kl, "raw_tvd": raw_tvd,
            "tk_olap_5": tk_olap_5, "tk_olap_10": tk_olap_10,
            "tk_mass_5": tk_mass_5, "tk_mass_10": tk_mass_10,
            "draft_baseline": draft_baseline, "draft_raw": draft_raw,
        })
        break  # SD stops at first reject

    return results


# ─── Accumulator ───────────────────────────────────────────────────────────

def make_acc(nl):
    return {
        # Correction per layer
        "corr_jsd_sum": np.zeros(nl+1), "corr_jsd_cnt": np.zeros(nl+1),
        "corr_kl_sum":  np.zeros(nl+1), "corr_kl_cnt":  np.zeros(nl+1),
        "corr_tvd_sum": np.zeros(nl+1), "corr_tvd_cnt": np.zeros(nl+1),
        "corr_top1_sum":       np.zeros(nl+1),
        "corr_top5_sum":       np.zeros(nl+1),
        "corr_top5_recall_sum": np.zeros(nl+1),
        # Raw distribution per layer
        "raw_jsd_sum": np.zeros(nl+1), "raw_kl_sum": np.zeros(nl+1),
        "raw_tvd_sum": np.zeros(nl+1), "raw_cnt":    np.zeros(nl+1),
        # Top-k per layer
        "tk_olap_5_sum": np.zeros(nl+1), "tk_olap_10_sum": np.zeros(nl+1),
        "tk_mass_5_sum": np.zeros(nl+1), "tk_mass_10_sum": np.zeros(nl+1),
        # Scalar
        "count": 0, "all_accept_count": 0,
        "reject_pos": [], "accept_probs": [],
        # Draft baseline correction
        "db_top1": 0.0, "db_top5": 0.0, "db_top10": 0.0,
        "db_top5_recall": 0.0, "db_top10_recall": 0.0,
        # Draft raw baseline
        "dr_jsd": 0.0, "dr_kl": 0.0, "dr_tvd": 0.0,
        "dr_tk_olap_5": 0.0, "dr_tk_olap_10": 0.0,
        "dr_tk_mass_5": 0.0, "dr_tk_mass_10": 0.0,
    }


def accumulate(d, result, n_layers):
    d["count"] += 1
    d["reject_pos"].append(result["reject_pos"])
    d["accept_probs"].append(result["accept_prob"])

    # Draft baseline correction
    db = result["draft_baseline"]
    d["db_top1"]  += float(db["top1_hit"])
    d["db_top5"]  += float(db["top5_hit"])
    d["db_top10"] += float(db["top10_hit"])
    d["db_top5_recall"]  += db["top5_recall"]
    d["db_top10_recall"] += db["top10_recall"]

    # Draft raw baseline
    dr = result["draft_raw"]
    d["dr_jsd"] += dr["jsd"]; d["dr_kl"] += dr["kl"]; d["dr_tvd"] += dr["tvd"]
    d["dr_tk_olap_5"]  += dr["topk_overlap_5"]
    d["dr_tk_olap_10"] += dr["topk_overlap_10"]
    d["dr_tk_mass_5"]  += dr["topk_mass_5"]
    d["dr_tk_mass_10"] += dr["topk_mass_10"]

    # Per-layer
    for k in range(n_layers + 1):
        jv = result["corr_jsd"][k]
        if jv == jv:
            d["corr_jsd_sum"][k] += jv; d["corr_jsd_cnt"][k] += 1
        kv = result["corr_kl"][k]
        if kv == kv:
            d["corr_kl_sum"][k] += kv; d["corr_kl_cnt"][k] += 1
        tv = result["corr_tvd"][k]
        if tv == tv:
            d["corr_tvd_sum"][k] += tv; d["corr_tvd_cnt"][k] += 1
        d["corr_top1_sum"][k]       += float(result["corr_top1"][k])
        d["corr_top5_sum"][k]       += float(result["corr_top5"][k])
        d["corr_top5_recall_sum"][k] += result["corr_top5_recall"][k]

        d["raw_jsd_sum"][k] += result["raw_jsd"][k]
        d["raw_kl_sum"][k]  += result["raw_kl"][k]
        d["raw_tvd_sum"][k] += result["raw_tvd"][k]
        d["raw_cnt"][k]     += 1

        d["tk_olap_5_sum"][k]  += result["tk_olap_5"][k]
        d["tk_olap_10_sum"][k] += result["tk_olap_10"][k]
        d["tk_mass_5_sum"][k]  += result["tk_mass_5"][k]
        d["tk_mass_10_sum"][k] += result["tk_mass_10"][k]


def finalize_acc(acc, n_layers):
    out = {}
    for lbl, d in acc.items():
        c  = max(d["count"], 1)
        rc = np.maximum(d["raw_cnt"], 1)
        def safe_avg(s, cnt):
            return np.where(cnt > 0, s / np.maximum(cnt, 1), float("nan")).tolist()

        out[lbl] = {
            "corr_jsd": safe_avg(d["corr_jsd_sum"], d["corr_jsd_cnt"]),
            "corr_kl":  safe_avg(d["corr_kl_sum"],  d["corr_kl_cnt"]),
            "corr_tvd": safe_avg(d["corr_tvd_sum"], d["corr_tvd_cnt"]),
            "corr_top1":       (d["corr_top1_sum"] / c).tolist(),
            "corr_top5":       (d["corr_top5_sum"] / c).tolist(),
            "corr_top5_recall": (d["corr_top5_recall_sum"] / c).tolist(),

            "raw_jsd": (d["raw_jsd_sum"] / rc).tolist(),
            "raw_kl":  (d["raw_kl_sum"]  / rc).tolist(),
            "raw_tvd": (d["raw_tvd_sum"] / rc).tolist(),

            "tk_olap_5":  (d["tk_olap_5_sum"]  / rc).tolist(),
            "tk_olap_10": (d["tk_olap_10_sum"] / rc).tolist(),
            "tk_mass_5":  (d["tk_mass_5_sum"]  / rc).tolist(),
            "tk_mass_10": (d["tk_mass_10_sum"] / rc).tolist(),

            "count": d["count"],
            "all_accept_count": d["all_accept_count"],
            "accept_rate": d["all_accept_count"] / max(d["all_accept_count"] + d["count"], 1),
            "mean_first_reject": float(np.mean(d["reject_pos"])) if d["reject_pos"] else None,
            "mean_accept_prob":  float(np.mean(d["accept_probs"])) if d["accept_probs"] else None,
            "reject_pos_dist": (
                {str(k): d["reject_pos"].count(k) for k in range(WINDOW_SIZE)}
                if d["reject_pos"] else {}
            ),
            "draft_baseline": {
                "top1":  d["db_top1"] / c,  "top5":  d["db_top5"] / c,
                "top10": d["db_top10"] / c,
                "top5_recall":  d["db_top5_recall"] / c,
                "top10_recall": d["db_top10_recall"] / c,
            },
            "draft_raw": {
                "jsd": d["dr_jsd"] / c, "kl": d["dr_kl"] / c, "tvd": d["dr_tvd"] / c,
                "tk_olap_5":  d["dr_tk_olap_5"] / c,
                "tk_olap_10": d["dr_tk_olap_10"] / c,
                "tk_mass_5":  d["dr_tk_mass_5"] / c,
                "tk_mass_10": d["dr_tk_mass_10"] / c,
            },
        }
    return out


# ─── Main experiment ───────────────────────────────────────────────────────

def run(args):
    cache_dir = args.cache_dir
    out_dir   = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    draft_paths = args.draft if args.draft else []

    if not draft_paths:
        print("ERROR: at least one --draft model required.")
        return

    # ── Load target (stays loaded throughout) ──
    print("=" * 60)
    print("Loading target model...")
    t_model, t_tok, t_norm, t_head, t_norm_dev, t_embed_dev, n_layers = \
        load_model(args.target, cache_dir=cache_dir)

    # ── Load dataset ──
    print(f"\nLoading GSM8K ({args.n_samples} samples)...")
    qa_pairs = load_gsm8k(args.n_samples, cache_dir=cache_dir)

    # ── Phase 1: Target greedy generation ──
    print("\nPhase 1: Target greedy generation...")
    target_cache = []  # list of (q_fmt, q_ids, target_toks)
    for question, _ in tqdm(qa_pairs, desc="Target gen"):
        q_fmt = f"Question: {question}\nAnswer: "
        q_ids = t_tok(q_fmt, return_tensors="pt", add_special_tokens=True).input_ids
        target_toks = generate_target(t_model, t_embed_dev, q_ids, max_new=512)
        target_cache.append((q_fmt, q_ids, target_toks))

    # ── Phase 2: For each draft model ──
    all_draft_data = []  # list of {name, path, n_layers, data: {cp: metrics}}
    for di, draft_path in enumerate(draft_paths):
        d_name = os.path.basename(draft_path.rstrip("/"))
        print(f"\n{'=' * 60}")
        print(f"Phase 2-{di}: Draft = {d_name}")

        d_model, d_tok, _, _, _, d_embed_dev, d_nlayers = \
            load_model(draft_path, cache_dir=cache_dir)

        acc = {lbl: make_acc(n_layers) for lbl in CP_LABELS}

        for q_fmt, q_ids, target_toks in tqdm(target_cache, desc=f"[{d_name}]"):
            for cp, lbl in zip(CHECKPOINTS, CP_LABELS):
                if cp > 0 and target_toks.shape[1] < cp:
                    continue

                ctx = q_ids if cp == 0 else torch.cat([q_ids, target_toks[:, :cp]], dim=1)

                # Draft generates window
                draft_ids, draft_probs = generate_draft_window(
                    d_model, d_embed_dev, ctx, WINDOW_SIZE,
                )

                # Target verifies with KV cache
                results = verify_window_kv(
                    t_model, t_norm, t_head, t_norm_dev, t_embed_dev,
                    ctx, draft_ids, draft_probs, n_layers,
                )

                d = acc[lbl]
                if not results:
                    d["all_accept_count"] += 1
                    continue
                for result in results:
                    accumulate(d, result, n_layers)

        del d_model, d_tok
        _gc_gpu()

        finalized = finalize_acc(acc, n_layers)
        all_draft_data.append({
            "name": d_name, "path": draft_path,
            "n_layers": d_nlayers, "data": finalized,
        })
        # Print quick summary for this draft
        for lbl in CP_LABELS:
            fd = finalized[lbl]
            if fd["count"] > 0:
                print(f"  {lbl}: n={fd['count']} accept_rate={fd['accept_rate']:.2f} "
                      f"corr_jsd@L[-2]={fd['corr_jsd'][-2]:.4f} "
                      f"raw_jsd@L[-2]={fd['raw_jsd'][-2]:.4f}")

    del t_model, t_tok, t_norm, t_head
    _gc_gpu()

    # ── Save JSON ──
    t_name = os.path.basename(args.target.rstrip("/"))
    save = {
        "target": args.target,
        "target_name": t_name,
        "n_layers": n_layers,
        "n_samples": args.n_samples,
        "window_size": WINDOW_SIZE,
        "checkpoints": CHECKPOINTS,
        "drafts": all_draft_data,
    }
    json_path = os.path.join(out_dir, f"correction_{t_name}.json")
    with open(json_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Plot ──
    plot_all(save, out_dir)
    print_summary(save)
    return save


# ─── Plotting ──────────────────────────────────────────────────────────────

DRAFT_COLORS = ["#e63946", "#2a9d8f", "#f4a261", "#6a4c93", "#1982c4"]
DRAFT_STYLES = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 2))]


def plot_all(save, out_dir):
    """Generate per-draft detail plots + cross-draft comparison overview."""
    n_layers = save["n_layers"]
    t_name   = save["target_name"]
    drafts   = save["drafts"]
    cp_labels = [f"cp{c}" for c in save["checkpoints"]]
    layers   = np.arange(n_layers + 1)

    # ── Per-draft detailed plot ──
    for di, draft in enumerate(drafts):
        d_name = draft["name"]
        data   = draft["data"]
        n_cp   = len(cp_labels)

        rows_spec = [
            # (key,             ylabel,                          ylim,          is_corr)
            ("corr_jsd",        "Corr JSD(r̂, r_true) ↓",       (0, 0.75),     True),
            ("corr_kl",         "Corr KL(r_true ∥ r̂) ↓",       None,          True),
            ("corr_tvd",        "Corr TVD ↓",                   (0, 1.0),      True),
            ("corr_top1",       "Corr Top-1 Match ↑",           (-0.05, 1.05), True),
            ("corr_top5",       "Corr Top-5 Cover ↑",           (-0.05, 1.05), True),
            ("corr_top5_recall","Corr Top-5 Recall ↑",          (-0.05, 1.05), True),
            ("raw_jsd",         "Raw JSD(p_T, p_E) ↓",          (0, 0.75),     False),
            ("raw_kl",          "Raw KL(p_T ∥ p_E) ↓",          None,          False),
            ("raw_tvd",         "Raw TVD ↓",                    (0, 1.0),      False),
            ("tk_olap_5",       "Top-5 Overlap ↑",              (-0.05, 1.05), False),
            ("tk_mass_5",       "Top-5 Mass ↑",                 (-0.05, 1.05), False),
            ("tk_olap_10",      "Top-10 Overlap ↑",             (-0.05, 1.05), False),
        ]
        n_rows = len(rows_spec)
        fig, axes = plt.subplots(n_rows, n_cp, figsize=(5 * n_cp, 3.5 * n_rows))
        if n_cp == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle(
            f"Correction Analysis: {t_name} (target) × {d_name} (draft)\n"
            f"window={save['window_size']}, probabilistic SD",
            fontsize=13, fontweight="bold",
        )

        cp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for row, (key, ylabel, ylim, is_corr) in enumerate(rows_spec):
            for col, lbl in enumerate(cp_labels):
                ax = axes[row, col]
                fd = data[lbl]

                if fd["count"] == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes)
                    if row == 0:
                        ax.set_title(lbl)
                    continue

                vals = np.array(fd[key], dtype=float)
                ax.plot(layers, vals, lw=1.5, color=cp_colors[col % 4], label="proxy")

                # Final layer reference
                fv = vals[-1] if not np.isnan(vals[-1]) else np.nanmin(vals)
                ax.axhline(fv, color="gray", lw=0.8, ls="--", alpha=0.6,
                           label=f"final={fv:.3f}")

                # Draft baselines
                corr_bl_map = {
                    "corr_top1": "top1", "corr_top5": "top5",
                    "corr_top5_recall": "top5_recall",
                }
                raw_bl_map = {
                    "raw_jsd": "jsd", "raw_kl": "kl", "raw_tvd": "tvd",
                    "tk_olap_5": "tk_olap_5", "tk_olap_10": "tk_olap_10",
                    "tk_mass_5": "tk_mass_5", "tk_mass_10": "tk_mass_10",
                }
                if key in corr_bl_map:
                    bl = fd["draft_baseline"][corr_bl_map[key]]
                    ax.axhline(bl, color="red", lw=1.2, ls=":", alpha=0.8,
                               label=f"draft={bl:.3f}")
                elif key in raw_bl_map:
                    bl = fd["draft_raw"][raw_bl_map[key]]
                    ax.axhline(bl, color="red", lw=1.2, ls=":", alpha=0.8,
                               label=f"draft={bl:.3f}")

                if row == 0:
                    mr = fd["mean_first_reject"]
                    mr_s = f"{mr:.1f}" if mr is not None else "?"
                    ax.set_title(f"{lbl} (n={fd['count']}, rej@{mr_s})", fontsize=9)
                ax.set_ylabel(ylabel if col == 0 else "", fontsize=8)
                ax.set_xlabel("Layer" if row == n_rows - 1 else "", fontsize=8)
                if ylim:
                    ax.set_ylim(*ylim)
                ax.set_xlim(0, n_layers)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6, loc="best")

        plt.tight_layout()
        p = os.path.join(out_dir, f"correction_{t_name}_{d_name}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {p}")

    # ── Cross-draft comparison overview (cp0 only) ──
    if len(drafts) < 2:
        return

    overview_keys = [
        ("corr_jsd", "Corr JSD ↓",     (0, 0.75)),
        ("corr_tvd", "Corr TVD ↓",     (0, 1.0)),
        ("corr_top1","Corr Top-1 ↑",   (-0.05, 1.05)),
        ("raw_jsd",  "Raw JSD ↓",      (0, 0.75)),
        ("tk_olap_5","Top-5 Overlap ↑", (-0.05, 1.05)),
        ("tk_mass_5","Top-5 Mass ↑",    (-0.05, 1.05)),
    ]
    n_ov = len(overview_keys)
    n_cp = len(cp_labels)
    fig, axes = plt.subplots(n_ov, n_cp, figsize=(5 * n_cp, 3.5 * n_ov))
    if n_cp == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle(
        f"Draft Comparison: {t_name} target, {len(drafts)} drafts",
        fontsize=14, fontweight="bold",
    )

    for row, (key, ylabel, ylim) in enumerate(overview_keys):
        for col, lbl in enumerate(cp_labels):
            ax = axes[row, col]
            for di, draft in enumerate(drafts):
                fd = draft["data"][lbl]
                if fd["count"] == 0:
                    continue
                vals = np.array(fd[key], dtype=float)
                color = DRAFT_COLORS[di % len(DRAFT_COLORS)]
                ax.plot(layers, vals, lw=1.5, color=color,
                        label=f"{draft['name']} proxy")

                # Draft baseline as horizontal line
                raw_bl_map = {
                    "raw_jsd": "jsd", "raw_tvd": "tvd",
                    "tk_olap_5": "tk_olap_5", "tk_olap_10": "tk_olap_10",
                    "tk_mass_5": "tk_mass_5", "tk_mass_10": "tk_mass_10",
                }
                if key in raw_bl_map:
                    bl = fd["draft_raw"][raw_bl_map[key]]
                    ax.axhline(bl, color=color, lw=1.0,
                               ls=DRAFT_STYLES[di % len(DRAFT_STYLES)],
                               alpha=0.7, label=f"{draft['name']} draft={bl:.3f}")

            if row == 0:
                ax.set_title(lbl, fontsize=10)
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=8)
            ax.set_xlabel("Layer" if row == n_ov - 1 else "", fontsize=8)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xlim(0, n_layers)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc="best")

    plt.tight_layout()
    p = os.path.join(out_dir, f"correction_{t_name}_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {p}")


# ─── Summary ──────────────────────────────────────────────────────────────

def print_summary(save):
    t_name   = save["target_name"]
    n_layers = save["n_layers"]
    drafts   = save["drafts"]
    cp_labels = [f"cp{c}" for c in save["checkpoints"]]

    print(f"\n{'='*110}")
    print(f"SUMMARY  target={t_name} ({n_layers}L)  window={save['window_size']}")
    for d in drafts:
        print(f"  Draft: {d['name']} ({d['n_layers']}L)")
    print(f"{'='*110}")

    hdr = (f"{'Draft':<25} {'CP':<6} | {'n':>4} {'skip':>4} {'aRate':>5} {'rej@':>5}"
           f" | {'cJSD':>7} {'cKL':>7} {'cTVD':>7} {'cT1':>5} {'cT5':>5}"
           f" | {'rJSD':>7} {'rKL':>7} {'rTVD':>7}"
           f" | {'t5ol':>5} {'t5ms':>5}")
    print(hdr)
    print("-" * 120)

    for d in drafts:
        for lbl in cp_labels:
            fd = d["data"][lbl]
            if fd["count"] == 0:
                print(f"{d['name']:<25} {lbl:<6} | {'skip':>4}")
                continue
            i = -2  # second-to-last layer
            ar = fd["accept_rate"]
            mr = fd["mean_first_reject"]
            mr_s = f"{mr:.1f}" if mr is not None else "?"
            print(
                f"{d['name']:<25} {lbl:<6} "
                f"| {fd['count']:>4} {fd['all_accept_count']:>4} {ar:>5.2f} {mr_s:>5}"
                f" | {fd['corr_jsd'][i]:>7.4f} {fd['corr_kl'][i]:>7.4f} "
                f"{fd['corr_tvd'][i]:>7.4f} {fd['corr_top1'][i]:>5.3f} {fd['corr_top5'][i]:>5.3f}"
                f" | {fd['raw_jsd'][i]:>7.4f} {fd['raw_kl'][i]:>7.4f} "
                f"{fd['raw_tvd'][i]:>7.4f}"
                f" | {fd['tk_olap_5'][i]:>5.3f} {fd['tk_mass_5'][i]:>5.3f}"
            )
        print()

    # Draft raw baselines
    print("Draft raw baselines (p_D vs p_T at rejected positions):")
    for d in drafts:
        for lbl in cp_labels:
            fd = d["data"][lbl]
            if fd["count"] == 0:
                continue
            dr = fd["draft_raw"]
            print(f"  {d['name']:<22} {lbl:<6} "
                  f"JSD={dr['jsd']:.4f} KL={dr['kl']:.4f} TVD={dr['tvd']:.4f} "
                  f"t5ol={dr['tk_olap_5']:.3f} t5ms={dr['tk_mass_5']:.3f}")


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Correction distribution feasibility analysis (70B compatible)")
    parser.add_argument("--target", required=True, help="Target model path")
    parser.add_argument("--draft", nargs="+", required=True,
                        help="Draft model path(s). Multiple for comparison.")
    parser.add_argument("--n_samples",  type=int, default=200)
    parser.add_argument("--output_dir", default="/home/chokwans99/Parallel_SD/results")
    parser.add_argument("--cache_dir",  default="/data2/shared/huggingface_cache")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None,
                        help="Override checkpoints. e.g. --checkpoints 0 128")

    args = parser.parse_args()

    if args.checkpoints is not None:
        global CHECKPOINTS, CP_LABELS
        CHECKPOINTS = args.checkpoints
        CP_LABELS   = [f"cp{c}" for c in CHECKPOINTS]

    run(args)


if __name__ == "__main__":
    main()
