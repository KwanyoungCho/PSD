"""
Early-Exit Verification Feasibility Analysis (MESA-SSD)
========================================================
Target의 early-exit 분포(P_E)가 SD verification에서 P_T를 대체할 수 있는지 검증.

검증 항목:
  1. α̂ 정확도: min(1, P_E/P_D) vs min(1, P_T/P_D)
  2. ĥ 예측: P_E 기반 first-rejection 분포 정확도
  3. Anchor coverage: ĥ_E 상위-K가 실제 reject 위치 커버 비율
  4. Token hit: r̂∝[P_E-P_D]+ correction token 예측 정확도
  5. Cheap scalar: entropy/margin/disagreement → reject risk 예측
"""

import argparse, json, os, random, gc, math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

WINDOW_SIZE = 10
CHECKPOINTS = [1]
CP_LABELS = ["cp1"]


# ─── Utilities ──────────────────────────────────────────────────────────

def js_div(p, q, eps=1e-10):
    p, q = p.clamp(min=eps), q.clamp(min=eps)
    m = 0.5 * (p + q)
    return float(0.5*(p*(p/m).log()).sum() + 0.5*(q*(q/m).log()).sum())

def residual_dist(p, q, eps=1e-10):
    r = (p - q).clamp(min=0)
    s = r.sum().item()
    return (r / s) if s > eps else None

def get_final_norm(model):
    if hasattr(model.model, "norm"): return model.model.norm
    if hasattr(model.model, "ln_f"): return model.model.ln_f
    raise ValueError(f"Unknown norm: {model.__class__.__name__}")

@torch.no_grad()
def batch_hidden_to_probs(hiddens, norm, lm_head, norm_device):
    dtype = next(lm_head.parameters()).dtype
    head_device = next(lm_head.parameters()).device
    normed = norm(hiddens.to(norm_device))
    logits = lm_head(normed.to(head_device).to(dtype))
    return F.softmax(logits.float(), dim=-1).cpu()

@torch.no_grad()
def hidden_to_probs_no_norm(hidden, lm_head):
    dtype = next(lm_head.parameters()).dtype
    head_device = next(lm_head.parameters()).device
    logits = lm_head(hidden.to(head_device).to(dtype).unsqueeze(0)).squeeze()
    return F.softmax(logits.float(), dim=-1).cpu()

def load_model(path, cache_dir=None):
    print(f"  Loading: {path}")
    tok = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=cache_dir)
    model.eval()
    norm = get_final_norm(model)
    norm_dev = next(norm.parameters()).device
    embed_dev = model.model.embed_tokens.weight.device
    n_layers = model.config.num_hidden_layers
    print(f"    layers={n_layers}  embed@{embed_dev}  norm@{norm_dev}")
    return model, tok, norm, model.lm_head, norm_dev, embed_dev, n_layers

def _gc_gpu():
    gc.collect(); torch.cuda.empty_cache()

def load_gsm8k(n, seed=42, cache_dir=None):
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    rng = random.Random(seed)
    return [(ds[i]["question"], ds[i]["answer"]) for i in rng.sample(range(len(ds)), n)]


# ─── Generation ─────────────────────────────────────────────────────────

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
        p = F.softmax(out.logits[0, -1, :].float(), dim=-1).cpu()
        ids.append(p.argmax().item())
        probs.append(p)
    return torch.tensor(ids), probs


# ─── Full window verification (no early stop) ──────────────────────────

@torch.no_grad()
def verify_window_full(t_model, t_norm, t_head, t_norm_dev, t_embed_dev,
                       context_ids, draft_ids, draft_probs, n_layers):
    """Process entire draft window, collect per-position per-layer metrics."""
    L = context_ids.shape[1]
    W = len(draft_ids)
    ctx = context_ids.to(t_embed_dev)
    nl1 = n_layers + 1

    if L > 1:
        prefill = t_model(ctx[:, :-1], use_cache=True, output_hidden_states=False)
        past_kv = prefill.past_key_values
    else:
        past_kv = None

    tokens_to_feed = [ctx[0, -1].item()] + [draft_ids[j].item() for j in range(W-1)]
    positions = []

    for i, tid in enumerate(tokens_to_feed):
        inp = torch.tensor([[tid]], device=t_embed_dev)
        out = t_model(inp, past_key_values=past_kv, use_cache=True,
                      output_hidden_states=True)
        past_kv = out.past_key_values

        p_T = F.softmax(out.logits[0, -1, :].float(), dim=-1).cpu()
        pre_h = torch.stack([hs[0, -1, :] for hs in out.hidden_states[:-1]])
        pre_probs = batch_hidden_to_probs(pre_h, t_norm, t_head, t_norm_dev)
        post_prob = hidden_to_probs_no_norm(out.hidden_states[-1][0, -1, :], t_head)
        all_probs = torch.cat([pre_probs, post_prob.unsqueeze(0)], dim=0)

        p_D = draft_probs[i]
        y_i = draft_ids[i].item()
        pD_yi = max(p_D[y_i].item(), 1e-10)

        # Alpha values
        alpha_T = min(1.0, p_T[y_i].item() / pD_yi)
        alpha_E = [min(1.0, all_probs[l][y_i].item() / pD_yi) for l in range(nl1)]

        # Correction distributions
        r_true = residual_dist(p_T, p_D)
        true_top1 = r_true.argmax().item() if r_true is not None else -1
        true_top5 = set(r_true.topk(5).indices.tolist()) if r_true is not None else set()

        corr_jsd, corr_top1, corr_top5r = [], [], []
        for l in range(nl1):
            r_hat = residual_dist(all_probs[l], p_D)
            if r_true is not None and r_hat is not None:
                corr_jsd.append(js_div(r_true, r_hat))
                corr_top1.append(r_hat.argmax().item() == true_top1)
                rh5 = set(r_hat.topk(5).indices.tolist())
                corr_top5r.append(len(true_top5 & rh5) / max(len(true_top5), 1))
            else:
                corr_jsd.append(float('nan'))
                corr_top1.append(False)
                corr_top5r.append(0.0)

        # Cheap scalars
        entropy, margin, disagree = [], [], []
        for l in range(nl1):
            pE = all_probs[l]
            entropy.append(-(pE.clamp(min=1e-10) * pE.clamp(min=1e-10).log()).sum().item())
            sv = pE.sort(descending=True).values
            margin.append((sv[0] - sv[1]).item())
            disagree.append(abs(pE[y_i].item() - p_D[y_i].item()))

        positions.append({
            'alpha_T': alpha_T, 'alpha_E': alpha_E,
            'corr_jsd': corr_jsd, 'corr_top1': corr_top1,
            'corr_top5r': corr_top5r,
            'entropy': entropy, 'margin': margin, 'disagree': disagree,
        })

    return positions


# ─── Hazard / Analysis ──────────────────────────────────────────────────

def compute_hazard(alphas):
    """ĥ_i = (∏_{j<i} α_j)(1-α_i). Returns [W+1] (last = all-accept)."""
    W = len(alphas)
    h = np.zeros(W + 1)
    cum = 1.0
    for i in range(W):
        h[i] = cum * (1.0 - alphas[i])
        cum *= alphas[i]
    h[W] = cum
    return h


def analyze_window(positions, n_layers):
    """Compute per-layer metrics for one window."""
    W = len(positions)
    nl1 = n_layers + 1

    alpha_T = np.array([p['alpha_T'] for p in positions])
    alpha_E = np.array([[p['alpha_E'][l] for p in positions] for l in range(nl1)])

    h_T = compute_hazard(alpha_T)
    h_E = np.array([compute_hazard(alpha_E[l]) for l in range(nl1)])

    idx = np.arange(W + 1)
    E_len_T = float(np.dot(h_T, idx))
    E_len_E = [float(np.dot(h_E[l], idx)) for l in range(nl1)]

    eps = 1e-10
    h_kl, h_tvd, alpha_mae = [], [], []
    for l in range(nl1):
        ht = h_T + eps; he = h_E[l] + eps
        ht_n = ht / ht.sum(); he_n = he / he.sum()
        h_kl.append(float(np.sum(ht_n * np.log(ht_n / he_n))))
        h_tvd.append(float(0.5 * np.abs(h_T - h_E[l]).sum()))
        alpha_mae.append(float(np.mean(np.abs(alpha_T - alpha_E[l]))))

    # Anchor coverage
    anchor_cov = {k: [] for k in [1, 3, 5]}
    for l in range(nl1):
        ranked = np.argsort(-h_E[l][:W])
        for K in [1, 3, 5]:
            top_k = set(ranked[:K].tolist())
            anchor_cov[K].append(sum(h_T[i] for i in top_k))

    # Weighted correction metrics (by h_T rejection prob)
    h_rej = h_T[:W]
    h_rej_n = h_rej / max(h_rej.sum(), eps)

    w_corr_jsd, w_corr_top1, w_corr_top5r = [], [], []
    for l in range(nl1):
        jsd_l = np.array([p['corr_jsd'][l] for p in positions])
        top1_l = np.array([float(p['corr_top1'][l]) for p in positions])
        top5r_l = np.array([p['corr_top5r'][l] for p in positions])
        valid = ~np.isnan(jsd_l)
        if valid.any():
            w = h_rej_n.copy(); w[~valid] = 0; ws = w.sum()
            w_corr_jsd.append(float(np.dot(w, np.nan_to_num(jsd_l)) / ws) if ws > 0 else float('nan'))
        else:
            w_corr_jsd.append(float('nan'))
        w_corr_top1.append(float(np.dot(h_rej_n, top1_l)))
        w_corr_top5r.append(float(np.dot(h_rej_n, top5r_l)))

    return {
        'alpha_mae': alpha_mae, 'h_kl': h_kl, 'h_tvd': h_tvd,
        'E_len_T': E_len_T, 'E_len_E': E_len_E,
        'anchor_cov_1': anchor_cov[1], 'anchor_cov_3': anchor_cov[3],
        'anchor_cov_5': anchor_cov[5],
        'w_corr_jsd': w_corr_jsd, 'w_corr_top1': w_corr_top1,
        'w_corr_top5r': w_corr_top5r,
        'all_accept_prob': float(h_T[-1]),
    }


# ─── Accumulator ────────────────────────────────────────────────────────

def make_acc(nl):
    nl1 = nl + 1
    return {
        'n_windows': 0, 'nl1': nl1,
        # Per-window metric sums
        'alpha_mae_sum': np.zeros(nl1), 'h_kl_sum': np.zeros(nl1),
        'h_tvd_sum': np.zeros(nl1), 'E_len_E_sum': np.zeros(nl1),
        'E_len_T_sum': 0.0, 'all_accept_sum': 0.0,
        'acov1_sum': np.zeros(nl1), 'acov3_sum': np.zeros(nl1),
        'acov5_sum': np.zeros(nl1),
        'wcj_sum': np.zeros(nl1), 'wcj_cnt': np.zeros(nl1),
        'wct1_sum': np.zeros(nl1), 'wct5r_sum': np.zeros(nl1),
        # Online correlation: alpha_T vs alpha_E  (per-position)
        'cn': 0,
        'at_s': 0.0, 'at_s2': 0.0,
        'ae_s': np.zeros(nl1), 'ae_s2': np.zeros(nl1), 'ate_s': np.zeros(nl1),
        # Online correlation: cheap scalars vs reject_prob
        'rp_s': 0.0, 'rp_s2': 0.0,
        'ent_s': np.zeros(nl1), 'ent_s2': np.zeros(nl1), 'ent_rp': np.zeros(nl1),
        'mar_s': np.zeros(nl1), 'mar_s2': np.zeros(nl1), 'mar_rp': np.zeros(nl1),
        'dis_s': np.zeros(nl1), 'dis_s2': np.zeros(nl1), 'dis_rp': np.zeros(nl1),
    }


def accumulate(acc, analysis, positions, nl):
    nl1 = nl + 1
    acc['n_windows'] += 1
    acc['all_accept_sum'] += analysis['all_accept_prob']
    acc['E_len_T_sum'] += analysis['E_len_T']

    for key, src in [('alpha_mae_sum','alpha_mae'), ('h_kl_sum','h_kl'),
                     ('h_tvd_sum','h_tvd'), ('E_len_E_sum','E_len_E'),
                     ('acov1_sum','anchor_cov_1'), ('acov3_sum','anchor_cov_3'),
                     ('acov5_sum','anchor_cov_5'),
                     ('wct1_sum','w_corr_top1'), ('wct5r_sum','w_corr_top5r')]:
        for l in range(nl1):
            acc[key][l] += analysis[src][l]

    for l in range(nl1):
        v = analysis['w_corr_jsd'][l]
        if not math.isnan(v):
            acc['wcj_sum'][l] += v; acc['wcj_cnt'][l] += 1

    # Per-position online stats
    for p in positions:
        aT = p['alpha_T']; rp = 1.0 - aT
        acc['cn'] += 1
        acc['at_s'] += aT; acc['at_s2'] += aT*aT
        acc['rp_s'] += rp; acc['rp_s2'] += rp*rp

        aE = np.array(p['alpha_E'])
        acc['ae_s'] += aE; acc['ae_s2'] += aE*aE; acc['ate_s'] += aT*aE

        ent = np.array(p['entropy'])
        acc['ent_s'] += ent; acc['ent_s2'] += ent*ent; acc['ent_rp'] += ent*rp

        mar = np.array(p['margin'])
        acc['mar_s'] += mar; acc['mar_s2'] += mar*mar; acc['mar_rp'] += mar*rp

        dis = np.array(p['disagree'])
        acc['dis_s'] += dis; acc['dis_s2'] += dis*dis; acc['dis_rp'] += dis*rp


def _online_corr(n, sx, sx2, sy, sy2, sxy):
    """Pearson r from online sums."""
    if n < 2: return float('nan')
    num = n*sxy - sx*sy
    den = np.sqrt(np.maximum((n*sx2 - sx*sx) * (n*sy2 - sy*sy), 1e-30))
    r = num / den
    return np.where(den > 1e-15, r, float('nan'))


def finalize_acc(all_acc, n_layers):
    out = {}
    nl1 = n_layers + 1
    for lbl, a in all_acc.items():
        nw = max(a['n_windows'], 1)
        n = a['cn']

        # Alpha correlation (per layer)
        alpha_corr = _online_corr(
            n, a['at_s'], a['at_s2'], a['ae_s'], a['ae_s2'], a['ate_s'])

        # Cheap scalar correlations (vs reject_prob)
        ent_corr = _online_corr(
            n, a['ent_s'], a['ent_s2'], a['rp_s'], a['rp_s2'], a['ent_rp'])
        mar_corr = _online_corr(
            n, a['mar_s'], a['mar_s2'], a['rp_s'], a['rp_s2'], a['mar_rp'])
        dis_corr = _online_corr(
            n, a['dis_s'], a['dis_s2'], a['rp_s'], a['rp_s2'], a['dis_rp'])

        def to_list(x):
            return x.tolist() if hasattr(x, 'tolist') else [float(v) for v in x]

        out[lbl] = {
            'n_windows': a['n_windows'],
            'mean_all_accept_prob': a['all_accept_sum'] / nw,
            'mean_E_len_T': a['E_len_T_sum'] / nw,
            'alpha_mae': (a['alpha_mae_sum'] / nw).tolist(),
            'alpha_corr': to_list(alpha_corr),
            'h_kl': (a['h_kl_sum'] / nw).tolist(),
            'h_tvd': (a['h_tvd_sum'] / nw).tolist(),
            'mean_E_len_E': (a['E_len_E_sum'] / nw).tolist(),
            'anchor_cov_1': (a['acov1_sum'] / nw).tolist(),
            'anchor_cov_3': (a['acov3_sum'] / nw).tolist(),
            'anchor_cov_5': (a['acov5_sum'] / nw).tolist(),
            'weighted_corr_jsd': np.where(
                a['wcj_cnt'] > 0, a['wcj_sum'] / np.maximum(a['wcj_cnt'], 1),
                float('nan')).tolist(),
            'weighted_corr_top1': (a['wct1_sum'] / nw).tolist(),
            'weighted_corr_top5_recall': (a['wct5r_sum'] / nw).tolist(),
            'entropy_reject_corr': to_list(ent_corr),
            'margin_reject_corr': to_list(mar_corr),
            'disagree_reject_corr': to_list(dis_corr),
        }
    return out


# ─── Main experiment ────────────────────────────────────────────────────

def run(args):
    cache_dir = args.cache_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    draft_paths = args.draft if args.draft else []
    if not draft_paths:
        print("ERROR: at least one --draft model required."); return

    print("=" * 60)
    print("Loading target model...")
    t_model, t_tok, t_norm, t_head, t_norm_dev, t_embed_dev, n_layers = \
        load_model(args.target, cache_dir=cache_dir)

    print(f"\nLoading GSM8K ({args.n_samples} samples)...")
    qa_pairs = load_gsm8k(args.n_samples, cache_dir=cache_dir)

    print("\nPhase 1: Target greedy generation...")
    target_cache = []
    for question, _ in tqdm(qa_pairs, desc="Target gen"):
        q_fmt = f"Question: {question}\nAnswer: "
        q_ids = t_tok(q_fmt, return_tensors="pt", add_special_tokens=True).input_ids
        target_toks = generate_target(t_model, t_embed_dev, q_ids, max_new=512)
        target_cache.append((q_fmt, q_ids, target_toks))

    all_draft_data = []
    for di, draft_path in enumerate(draft_paths):
        d_name = os.path.basename(draft_path.rstrip("/"))
        print(f"\n{'='*60}\nPhase 2-{di}: Draft = {d_name}")
        d_model, d_tok, _, _, _, d_embed_dev, d_nlayers = \
            load_model(draft_path, cache_dir=cache_dir)

        acc = {lbl: make_acc(n_layers) for lbl in CP_LABELS}

        for q_fmt, q_ids, target_toks in tqdm(target_cache, desc=f"[{d_name}]"):
            for cp, lbl in zip(CHECKPOINTS, CP_LABELS):
                if cp > 0 and target_toks.shape[1] < cp:
                    continue
                ctx = q_ids if cp == 0 else torch.cat([q_ids, target_toks[:, :cp]], dim=1)

                draft_ids, draft_probs = generate_draft_window(
                    d_model, d_embed_dev, ctx, WINDOW_SIZE)

                positions = verify_window_full(
                    t_model, t_norm, t_head, t_norm_dev, t_embed_dev,
                    ctx, draft_ids, draft_probs, n_layers)

                analysis = analyze_window(positions, n_layers)
                accumulate(acc[lbl], analysis, positions, n_layers)

        del d_model, d_tok; _gc_gpu()

        finalized = finalize_acc(acc, n_layers)
        all_draft_data.append({
            "name": d_name, "path": draft_path,
            "n_layers": d_nlayers, "data": finalized,
        })

        for lbl in CP_LABELS:
            fd = finalized[lbl]
            if fd['n_windows'] > 0:
                print(f"  {lbl}: n={fd['n_windows']} "
                      f"E[L_T]={fd['mean_E_len_T']:.2f} "
                      f"E[L_E]@L[-2]={fd['mean_E_len_E'][-2]:.2f} "
                      f"α_MAE@L[-2]={fd['alpha_mae'][-2]:.4f} "
                      f"aCov5@L[-2]={fd['anchor_cov_5'][-2]:.3f}")

    del t_model, t_tok, t_norm, t_head; _gc_gpu()

    t_name = os.path.basename(args.target.rstrip("/"))
    save = {
        "target": args.target, "target_name": t_name,
        "n_layers": n_layers, "n_samples": args.n_samples,
        "window_size": WINDOW_SIZE, "checkpoints": CHECKPOINTS,
        "drafts": all_draft_data,
    }
    json_path = os.path.join(out_dir, f"ee_verify_{t_name}.json")
    with open(json_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved: {json_path}")

    print_summary(save)
    return save


# ─── Summary ────────────────────────────────────────────────────────────

def print_summary(save):
    nl = save["n_layers"]
    t_name = save["target_name"]
    print(f"\n{'='*90}")
    print(f"EE-VERIFY SUMMARY  target={t_name} ({nl}L)  window={save['window_size']}")
    print(f"{'='*90}")

    for d in save["drafts"]:
        print(f"\nDraft: {d['name']} ({d['n_layers']}L)")
        for lbl in [f"cp{c}" for c in save["checkpoints"]]:
            fd = d["data"][lbl]
            nw = fd["n_windows"]
            if nw == 0: continue
            print(f"\n  [{lbl}] n_windows={nw}")
            print(f"    mean P(all accept) = {fd['mean_all_accept_prob']:.3f}")
            print(f"    mean E[accepted_len] T={fd['mean_E_len_T']:.2f}")

            # Sample layers
            for li, label in [(nl//4, f"L{nl//4}"), (nl//2, f"L{nl//2}"),
                              (3*nl//4, f"L{3*nl//4}"), (nl-1, f"L{nl-1}"),
                              (nl, f"L{nl}(final)")]:
                print(f"    {label:>12}: "
                      f"α_MAE={fd['alpha_mae'][li]:.4f} "
                      f"α_corr={fd['alpha_corr'][li]:.3f} "
                      f"h_KL={fd['h_kl'][li]:.4f} "
                      f"aCov1={fd['anchor_cov_1'][li]:.3f} "
                      f"aCov5={fd['anchor_cov_5'][li]:.3f} "
                      f"E[L_E]={fd['mean_E_len_E'][li]:.2f}")

            print(f"\n    Cheap scalar → reject correlation (sample layers):")
            for li, label in [(nl//2, f"L{nl//2}"), (3*nl//4, f"L{3*nl//4}"),
                              (nl-1, f"L{nl-1}")]:
                print(f"    {label:>12}: "
                      f"entropy={fd['entropy_reject_corr'][li]:.3f} "
                      f"margin={fd['margin_reject_corr'][li]:.3f} "
                      f"disagree={fd['disagree_reject_corr'][li]:.3f}")

            print(f"\n    Correction quality (h_T-weighted):")
            for li, label in [(nl//2, f"L{nl//2}"), (3*nl//4, f"L{3*nl//4}"),
                              (nl-1, f"L{nl-1}")]:
                wj = fd['weighted_corr_jsd'][li]
                wj_s = f"{wj:.4f}" if not math.isnan(wj) else "NaN"
                print(f"    {label:>12}: "
                      f"corr_JSD={wj_s} "
                      f"corr_top1={fd['weighted_corr_top1'][li]:.3f} "
                      f"corr_top5r={fd['weighted_corr_top5_recall'][li]:.3f}")


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Early-exit verification feasibility analysis")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", nargs="+", required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="/home/chokwans99/Parallel_SD/results")
    parser.add_argument("--cache_dir", default="/data2/shared/huggingface_cache")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None,
                        help="Override checkpoints. e.g. --checkpoints 0 1")
    args = parser.parse_args()

    if args.checkpoints is not None:
        global CHECKPOINTS, CP_LABELS
        CHECKPOINTS = args.checkpoints
        CP_LABELS = [f"cp{c}" for c in CHECKPOINTS]

    run(args)


if __name__ == "__main__":
    main()
