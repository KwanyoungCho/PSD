"""
GSM8K Early-Exit vs Draft Model Analysis
==========================================
GSM8K 데이터셋에서 30개 샘플을 뽑아 다음을 비교:
  - Target model 각 layer의 logit lens 분포 vs target final 분포 (JSD curve)
  - Draft model final layer 분포 vs target final 분포 (JSD 단일 값, 수평선)
  - 교차 layer: early-exit이 draft보다 좋아지는 시점

Checkpoint 위치 (output token 기준):
  - cp0: 첫 번째 output token (input만 넣었을 때 다음 토큰 예측)
  - cp128, cp256, cp384: output 128/256/384 토큰째 예측
  → GSM8K answer가 충분히 길면 실제 reasoning 진행 중의 분포를 볼 수 있음
"""

import argparse
import json
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# ─────────────────────────── Utilities ───────────────────────────

def get_final_norm(model):
    if hasattr(model.model, "norm"):
        return model.model.norm
    elif hasattr(model.model, "ln_f"):
        return model.model.ln_f
    raise ValueError(f"Unknown final norm for {model.__class__.__name__}")


@torch.no_grad()
def hidden_to_probs(hidden, norm, lm_head, norm_device):
    """hidden [hidden_dim] → prob [vocab_size] on CPU (float32)."""
    model_dtype = next(lm_head.parameters()).dtype
    x = hidden.to(norm_device).float()
    normed = norm(x.unsqueeze(0))
    logits = lm_head(normed.to(model_dtype)).squeeze()
    return F.softmax(logits.float(), dim=-1).cpu()


def kl_div(p, q, eps=1e-10):
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    return (q * (q / p).log()).sum().item()


def js_div(p, q):
    m = 0.5 * (p + q)
    return float(0.5 * kl_div(m, p) + 0.5 * kl_div(m, q))


def load_model(path):
    print(f"  Loading: {path}")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    norm = get_final_norm(model)
    norm_dev   = next(norm.parameters()).device
    embed_dev  = model.model.embed_tokens.weight.device
    lm_head    = model.lm_head
    n_layers   = model.config.num_hidden_layers
    print(f"  Layers={n_layers} | embed@{embed_dev} | norm/head@{norm_dev}")
    return model, tok, norm, lm_head, norm_dev, embed_dev, n_layers


# ─────────────────────────── GSM8K 로드 ───────────────────────────

def load_gsm8k(n_samples: int, seed: int = 42, cache_dir: str = None):
    """GSM8K test split에서 n_samples개 샘플링. (question, answer) 쌍 반환."""
    ds = load_dataset("openai/gsm8k", "main",
                      split="test",
                      cache_dir=cache_dir)
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n_samples)
    return [(ds[i]["question"], ds[i]["answer"]) for i in indices]


# ─────────────────────────── 분석 ───────────────────────────

@torch.no_grad()
def analyze_one_sample(
    q_ids: torch.Tensor,    # [1, q_len]  질문 토큰 (prefix)
    a_ids: torch.Tensor,    # [1, a_len]  답변 토큰 (분석 대상)
    target_model, t_norm, t_head, t_norm_dev, t_embed_dev,
    draft_model,  d_norm, d_head, d_norm_dev, d_embed_dev,
    n_layers: int,
):
    """
    답변 토큰 내 4개 위치(첫 토큰, 33%, 66%, 마지막)에서:
      - target 각 layer JSD (vs target final, logit lens)
      - draft final layer JSD (vs target final)
    반환.

    checkpoint labels: ["ans_0%", "ans_33%", "ans_66%", "ans_100%"]
    """
    a_len = a_ids.shape[1]
    if a_len == 0:
        return []

    # 답변 내 분석 위치: 0%, 33%, 66%, 마지막 토큰
    # pos는 답변 토큰 인덱스 (0-based), 실제 seq = q_ids + a_ids[:pos+1]
    ans_positions = sorted(set([
        0,
        max(0, a_len // 3),
        max(0, 2 * a_len // 3),
        a_len - 1,
    ]))
    cp_labels = [f"ans_{int(100*p/(a_len-1) if a_len>1 else 0)}%" for p in ans_positions]

    full_ids = torch.cat([q_ids, a_ids], dim=1)  # [1, q_len + a_len]

    results = []
    for ap, lbl in zip(ans_positions, cp_labels):
        # 질문 전체 + 답변 ap+1 토큰 → 마지막 토큰(답변 ap번째) 분석
        seq_len = q_ids.shape[1] + ap + 1
        seq = full_ids[:, :seq_len]

        # ── Target ──
        seq_t = seq.to(t_embed_dev)
        t_out = target_model(seq_t, output_hidden_states=True)

        ref_probs = hidden_to_probs(
            t_out.hidden_states[-1][0, -1, :], t_norm, t_head, t_norm_dev
        )
        ref_top1 = ref_probs.argmax().item()

        jsd_layers, top1_layers = [], []
        for hs in t_out.hidden_states:
            lp = hidden_to_probs(hs[0, -1, :], t_norm, t_head, t_norm_dev)
            jsd_layers.append(js_div(ref_probs, lp))
            top1_layers.append(float(lp.argmax().item() == ref_top1))

        # ── Draft (final layer only) ──
        seq_d = seq.to(d_embed_dev)
        d_out = draft_model(seq_d)
        draft_probs = F.softmax(d_out.logits[0, -1, :].float(), dim=-1).cpu()
        jsd_draft  = js_div(ref_probs, draft_probs)
        top1_draft = float(draft_probs.argmax().item() == ref_top1)

        # crossover layer (embed=0 제외, layer 1부터)
        crossover = next(
            (li for li in range(1, len(jsd_layers)) if jsd_layers[li] < jsd_draft),
            None
        )

        results.append({
            "label": lbl,
            "ans_pos": ap,
            "total_tokens": seq_len,
            "jsd_layers":      jsd_layers,
            "top1_layers":     top1_layers,
            "jsd_draft":       jsd_draft,
            "top1_draft":      top1_draft,
            "crossover_layer": crossover,
        })

    return results


# ─────────────────────────── 메인 실험 ───────────────────────────

def run(target_path, draft_path, n_samples, output_dir, cache_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── 모델 로드 ──
    t_model, t_tok, t_norm, t_head, t_norm_dev, t_embed_dev, n_layers = load_model(target_path)
    d_model, d_tok, d_norm, d_head, d_norm_dev, d_embed_dev, _        = load_model(draft_path)

    # ── GSM8K 로드 ──
    print(f"\nLoading GSM8K ({n_samples} samples)...")
    qa_pairs = load_gsm8k(n_samples, cache_dir=cache_dir)

    # 결과 누적: label별 (ans_0%, ans_33%, ans_66%, ans_100%)
    CP_LABELS = ["ans_0%", "ans_33%", "ans_66%", "ans_100%"]
    label_data = {lbl: {"jsd_layers":  np.zeros(n_layers + 1),
                         "top1_layers": np.zeros(n_layers + 1),
                         "jsd_draft":  0.0,
                         "top1_draft": 0.0,
                         "crossovers": [],
                         "count": 0}
                  for lbl in CP_LABELS}

    for question, answer in tqdm(qa_pairs, desc="Samples"):
        q_fmt = f"Question: {question}\nAnswer: "
        a_fmt = answer

        q_ids = t_tok(q_fmt, return_tensors="pt", add_special_tokens=True).input_ids
        a_ids = t_tok(a_fmt, return_tensors="pt", add_special_tokens=False).input_ids

        if a_ids.shape[1] == 0:
            continue

        sample_results = analyze_one_sample(
            q_ids, a_ids,
            t_model, t_norm, t_head, t_norm_dev, t_embed_dev,
            d_model, d_norm, d_head, d_norm_dev, d_embed_dev,
            n_layers,
        )

        for r in sample_results:
            # label 매핑: ans_X% 형태로 가장 가까운 CP_LABEL에 매핑
            lbl = _nearest_label(r["label"], CP_LABELS)
            d = label_data[lbl]
            d["jsd_layers"]  += np.array(r["jsd_layers"])
            d["top1_layers"] += np.array(r["top1_layers"])
            d["jsd_draft"]   += r["jsd_draft"]
            d["top1_draft"]  += r["top1_draft"]
            if r["crossover_layer"] is not None:
                d["crossovers"].append(r["crossover_layer"])
            d["count"] += 1

    # ── 평균 ──
    for lbl in CP_LABELS:
        d = label_data[lbl]
        c = max(d["count"], 1)
        d["jsd_layers"]  /= c
        d["top1_layers"] /= c
        d["jsd_draft"]   /= c
        d["top1_draft"]  /= c
        d["crossover_mean"]   = float(np.mean(d["crossovers"]))   if d["crossovers"] else None
        d["crossover_median"] = float(np.median(d["crossovers"])) if d["crossovers"] else None
        d["crossover_rate"]   = len(d["crossovers"]) / c

    # ── 저장 ──
    target_name = os.path.basename(target_path.rstrip("/"))
    draft_name  = os.path.basename(draft_path.rstrip("/"))
    save = {
        "target": target_path, "draft": draft_path,
        "n_samples": n_samples, "n_layers": n_layers,
        "label_data": {
            lbl: {k: v.tolist() if isinstance(v, np.ndarray) else v
                  for k, v in d.items() if k != "crossovers"}
            for lbl, d in label_data.items()
        }
    }
    json_path = os.path.join(output_dir, f"gsm8k_{target_name}.json")
    with open(json_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ── 플롯 ──
    plot(label_data, CP_LABELS, n_layers, target_name, draft_name, output_dir)

    return label_data


def _nearest_label(label: str, cp_labels: list) -> str:
    """'ans_X%' 형태의 label을 cp_labels 중 가장 가까운 것으로 매핑."""
    try:
        pct = int(label.replace("ans_", "").replace("%", ""))
    except ValueError:
        return cp_labels[0]
    targets = [int(l.replace("ans_", "").replace("%", "")) for l in cp_labels]
    return cp_labels[min(range(len(targets)), key=lambda i: abs(targets[i] - pct))]


# ─────────────────────────── 플롯 ───────────────────────────

def plot(label_data, CP_LABELS, n_layers, target_name, draft_name, output_dir):
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
    layers = np.arange(n_layers + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        f"GSM8K: Early-Exit vs Draft Model  (answer position별)\n"
        f"Target: {target_name}  |  Draft: {draft_name}",
        fontsize=13, fontweight="bold"
    )

    for idx, lbl in enumerate(CP_LABELS):
        d = label_data[lbl]
        if d["count"] == 0:
            continue
        c = colors[idx % len(colors)]

        # JSD curve (early-exit)
        axes[0].plot(layers, d["jsd_layers"], "-", color=c, lw=1.5,
                     label=f"{lbl} [early-exit]")
        # Draft JSD 수평선
        axes[0].axhline(d["jsd_draft"], color=c, lw=1.5, ls="--",
                        label=f"{lbl} [draft={d['jsd_draft']:.3f}]")
        # Crossover 수직선
        if d["crossover_mean"] is not None:
            xl = d["crossover_mean"]
            axes[0].axvline(xl, color=c, lw=0.8, ls=":", alpha=0.7)
            axes[0].annotate(f"×L{xl:.0f}", xy=(xl, d["jsd_draft"]),
                             xytext=(xl + 0.5, d["jsd_draft"] + 0.02),
                             fontsize=7, color=c)

        # Top-1 match
        axes[1].plot(layers, d["top1_layers"], "-", color=c, lw=1.5,
                     label=f"{lbl} [early-exit]")
        axes[1].axhline(d["top1_draft"], color=c, lw=1.5, ls="--",
                        label=f"{lbl} [draft={d['top1_draft']:.2f}]")

    axes[0].set_ylabel("JSD  (↓ better)")
    axes[0].set_title("JSD vs Target Final  |  실선=early-exit, 점선=draft JSD, 세로점선=교차점")
    axes[0].set_ylim(0, 0.75)
    axes[0].set_xlim(0, n_layers)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")

    axes[1].set_ylabel("Top-1 Match Rate  (↑ better)")
    axes[1].set_title("Top-1 Token Match with Target Final")
    axes[1].set_xlabel("Layer Index (0=embedding)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlim(0, n_layers)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")

    plt.tight_layout()
    p = os.path.join(output_dir, f"gsm8k_{target_name}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {p}")


# ─────────────────────────── CLI ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     required=True)
    parser.add_argument("--draft",      required=True)
    parser.add_argument("--n_samples",  type=int, default=30)
    parser.add_argument("--output_dir", default="/home/chokwans99/Parallel_SD/results")
    parser.add_argument("--cache_dir",  default="/data2/shared/huggingface_cache")
    args = parser.parse_args()

    label_data = run(
        target_path=args.target,
        draft_path=args.draft,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )

    # ── 콘솔 요약 출력 ──
    CP_LABELS = ["ans_0%", "ans_33%", "ans_66%", "ans_100%"]
    n_layers = next(iter(label_data.values()))["jsd_layers"].shape[0] - 1
    print("\n" + "="*70)
    print("SUMMARY: Crossover Layer (Early-exit beats Draft)")
    print(f"{'Position':<12} | {'Draft JSD':>10} | {'Draft top1':>10} | {'Crossover L':>12} | {'율':>6}")
    print("="*70)
    for lbl in CP_LABELS:
        d = label_data[lbl]
        if d["count"] == 0:
            print(f"{lbl:<12} | {'skipped':>10}")
            continue
        cm = d["crossover_mean"]
        cr = d["crossover_rate"]
        cm_str = f"L{cm:.1f} ({100*cm/n_layers:.0f}%)" if cm else "없음"
        print(f"{lbl:<12} | {d['jsd_draft']:>10.4f} | {d['top1_draft']:>10.3f} | {cm_str:>12} | {cr:>5.0%}")


if __name__ == "__main__":
    main()
