"""
Early Exit & Draft Model Layer Analysis
========================================
실험 목적:
  1. [Early Exit]  target model의 각 layer에서 logit lens로 뽑은 분포 vs.
                   target model 최종 출력 분포 비교
  2. [Draft Align] target model의 각 layer 분포 vs.
                   같은 vocab 계열의 소형(draft) model 최종 출력 비교

bfloat16 fix:
  final_probs를 outputs.logits가 아닌 hidden_states[-1] -> norm -> lm_head로
  계산하여 동일 코드 경로 → 마지막 layer KL = 0 보장.

Dataset 설계 (길이 × task 유형):
  short  (10-25 tok)  : 단순 사실 완성
  medium (60-120 tok) : 단락 수준 이해/추론
  long   (250-450 tok): 긴 문서 다음 토큰 예측
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────── Dataset ───────────────────────────
# short: 단순 사실 완성 (~10-25 tokens)
SHORT_PROMPTS = [
    "The capital of France is",
    "The speed of light is approximately",
    "Water boils at 100 degrees",
    "The largest planet in our solar system is",
    "In mathematics, the derivative of x^2 is",
    "Albert Einstein was born in",
    "The chemical formula for water is",
    "The Great Wall of China was built during",
]

# medium: 단락 수준 문맥 완성 (~60-120 tokens)
MEDIUM_PROMPTS = [
    (
        "The transformer architecture, introduced in 'Attention is All You Need' (2017), "
        "replaced recurrent networks for sequence modeling. Its core innovation is the "
        "self-attention mechanism, which allows every position to attend to all other "
        "positions. This enables full parallelization during training, unlike RNNs. "
        "The original paper introduced multi-head attention, positional encodings, and "
        "a feed-forward sublayer in each block. Since then, transformer-based models "
        "such as BERT, GPT, and T5 have dominated NLP benchmarks. The key advantage of "
        "self-attention over convolution is its ability to capture"
    ),
    (
        "Climate change refers to long-term shifts in global temperatures and weather "
        "patterns. While natural factors have always caused climate fluctuations, since "
        "the industrial revolution human activities—particularly the burning of fossil "
        "fuels—have become the dominant driver. Carbon dioxide and methane trap solar "
        "heat in the atmosphere, causing the greenhouse effect. Current projections "
        "suggest global average temperatures will rise between 1.5°C and 4.5°C by 2100 "
        "depending on emission trajectories. Consequences include rising sea levels, "
        "more frequent extreme weather events, and disruption of ecosystems. To limit "
        "warming to 1.5°C, global emissions must reach"
    ),
    (
        "Speculative decoding is a technique to accelerate autoregressive language model "
        "inference. A small draft model generates multiple candidate tokens cheaply, "
        "then a large target model verifies them in parallel. If the target model agrees "
        "with the draft tokens, they are accepted; otherwise generation rolls back to the "
        "first disagreement. The speedup comes from the fact that the target model can "
        "verify a batch of tokens in a single forward pass instead of generating them "
        "one by one. The acceptance rate depends on how well the draft model approximates "
        "the target model's distribution. Higher acceptance rates lead to greater speedup. "
        "A key challenge in speculative decoding is choosing a draft model that"
    ),
    (
        "The human brain contains approximately 86 billion neurons, each forming thousands "
        "of synaptic connections. These connections form the basis of learning and memory "
        "through a process called synaptic plasticity—the ability of synapses to strengthen "
        "or weaken over time in response to activity. Long-term potentiation (LTP) is "
        "considered the primary cellular mechanism underlying learning. When neurons fire "
        "together repeatedly, the synapse between them becomes stronger. This principle, "
        "sometimes summarized as 'neurons that fire together wire together,' underlies "
        "associative learning. Modern deep learning drew inspiration from this biological "
        "mechanism; artificial neural networks are trained by adjusting weights through"
    ),
]

# long: 긴 문서 문맥 다음 토큰 예측 (~250-450 tokens)
LONG_PROMPTS = [
    (
        "Large language models (LLMs) have demonstrated remarkable capabilities across a "
        "wide range of natural language tasks, from translation and summarization to "
        "complex reasoning and code generation. These models are typically based on the "
        "transformer architecture and trained on massive text corpora using self-supervised "
        "objectives, most commonly next-token prediction. The scale of these models has "
        "grown exponentially: while early transformer models had millions of parameters, "
        "current state-of-the-art models have hundreds of billions. This scaling has "
        "produced emergent capabilities—abilities that appear suddenly as model size "
        "crosses certain thresholds and that were not present in smaller versions. "
        "Examples include multi-step arithmetic, analogical reasoning, and in-context "
        "learning, where the model adapts to a task from just a few examples in the prompt "
        "without any parameter updates. Despite these impressive capabilities, LLMs face "
        "significant challenges. They can produce confident-sounding but factually "
        "incorrect outputs, a phenomenon called hallucination. They can exhibit biases "
        "present in their training data. Their inference is computationally expensive "
        "because each token must be generated sequentially, with each step requiring a "
        "full forward pass through the entire model. Various techniques have been proposed "
        "to address the inference cost. Quantization reduces the numerical precision of "
        "model weights from 32-bit or 16-bit floating point to 8-bit integers or even "
        "4-bit, reducing memory footprint and often increasing throughput with acceptable "
        "accuracy loss. Speculative decoding uses a small draft model to propose several "
        "tokens at once, which the large model then verifies in parallel—effectively "
        "amortizing the cost of the large model over multiple tokens. Early exit methods "
        "allow the model to stop at an intermediate layer when a confidence threshold is "
        "reached, avoiding computation in later layers for easy tokens. Layer pruning "
        "removes entire transformer blocks that are found to be less important. All of "
        "these methods attempt to exploit the observation that not all tokens require the "
        "same depth of computation to predict accurately. The insight behind early exit "
        "and speculative decoding is related: the"
    ),
    (
        "The history of artificial intelligence research spans more than seven decades, "
        "beginning with the foundational work of Alan Turing, who proposed the Turing "
        "Test in 1950 as a criterion for machine intelligence. The field experienced "
        "several cycles of optimism and disillusionment, known as AI winters, when "
        "progress stalled due to computational limitations and overpromised results. "
        "The first wave of AI focused on symbolic reasoning and expert systems, which "
        "encoded human knowledge as explicit rules. These systems performed well in "
        "narrow, well-defined domains but failed to generalize. The second wave brought "
        "machine learning, where systems learned patterns from data rather than relying "
        "on hand-coded rules. Neural networks, inspired by biological neurons, gained "
        "prominence in the 1980s with the backpropagation algorithm but fell out of "
        "favor due to computational constraints and the vanishing gradient problem. "
        "The deep learning revolution of the 2010s changed everything. The availability "
        "of large datasets, powerful GPUs, and algorithmic improvements such as ReLU "
        "activations and dropout enabled training of much deeper networks. AlexNet's "
        "victory in the ImageNet competition in 2012 demonstrated that deep "
        "convolutional networks could dramatically outperform traditional computer vision "
        "methods. Natural language processing followed a similar trajectory. Recurrent "
        "networks and LSTMs improved sequential modeling but struggled with long-range "
        "dependencies. The transformer architecture introduced in 2017 solved this by "
        "replacing recurrence with self-attention, enabling models to directly relate "
        "any two positions in a sequence regardless of distance. BERT, GPT, and their "
        "successors demonstrated that pretraining on large text corpora followed by "
        "fine-tuning on specific tasks—or prompting without fine-tuning—could achieve "
        "state-of-the-art results across nearly all language benchmarks. The scaling "
        "hypothesis posits that performance continues to improve predictably with more "
        "compute, data, and parameters. Following this, researchers have built "
        "increasingly large models, with GPT-4, Gemini, Claude, and Llama representing "
        "the current frontier. These models exhibit sophisticated behaviors such as "
        "chain-of-thought reasoning, tool use, and multi-step planning. The next "
        "challenge facing the field is"
    ),
]

DATASET = {
    "short": SHORT_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "long": LONG_PROMPTS,
}


# ─────────────────────────── Utilities ───────────────────────────

def get_final_norm(model):
    if hasattr(model.model, "norm"):
        return model.model.norm          # Llama / Qwen 계열
    elif hasattr(model.model, "ln_f"):
        return model.model.ln_f          # GPT 계열
    raise ValueError(f"Unknown final norm for {model.__class__.__name__}")


@torch.no_grad()
def hidden_to_probs(hidden: torch.Tensor, norm, lm_head, norm_device: torch.device) -> torch.Tensor:
    """
    hidden: [hidden_dim] tensor (last token of one layer)
    Returns: [vocab_size] float32 probability vector on CPU
    bfloat16 fix: norm을 float32로 수행 후 lm_head dtype으로 복원.
    self-consistent reference 사용 시 last layer KL=0 보장.
    """
    model_dtype = next(lm_head.parameters()).dtype
    x = hidden.to(norm_device).float()           # float32 for norm precision
    normed = norm(x.unsqueeze(0))                # [1, hidden_dim] (norm internally handles dtype)
    logits = lm_head(normed.to(model_dtype)).squeeze()  # lm_head expects model dtype
    return F.softmax(logits.float(), dim=-1).cpu()


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """KL(q || p)  (p=reference, q=target)"""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (q * (q / p).log()).sum().item()


def js_div(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * kl_div(m, p) + 0.5 * kl_div(m, q))


# ─────────────────────────── Model Loading ───────────────────────────

def load_model(path: str):
    print(f"  Loading tokenizer: {path}")
    tok = AutoTokenizer.from_pretrained(path)
    print(f"  Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    norm = get_final_norm(model)
    norm_device = next(norm.parameters()).device
    embed_device = model.model.embed_tokens.weight.device
    lm_head = model.lm_head
    num_layers = model.config.num_hidden_layers
    print(f"  Layers: {num_layers} | embed@{embed_device} | norm/head@{norm_device}")
    return model, tok, norm, lm_head, norm_device, embed_device, num_layers


# ─────────────────────────── Analysis ───────────────────────────

@torch.no_grad()
def analyze(
    target_path: str,
    draft_path: str | None,
    output_dir: str,
    length_categories: list[str],
):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load target model ──
    print(f"\n{'='*60}\nTarget: {target_path}\n{'='*60}")
    t_model, t_tok, t_norm, t_head, t_norm_dev, t_embed_dev, n_layers = load_model(target_path)

    # ── Load draft model (optional) ──
    d_model = d_tok = d_norm = d_head = d_norm_dev = d_embed_dev = None
    has_draft = draft_path is not None
    if has_draft:
        print(f"\n{'='*60}\nDraft: {draft_path}\n{'='*60}")
        d_model, d_tok, d_norm, d_head, d_norm_dev, d_embed_dev, _ = load_model(draft_path)

    # ── 결과 저장 구조 (length category별) ──
    all_results = {}
    for cat in length_categories:
        prompts = DATASET[cat]
        n = n_layers + 1  # embedding + N layers

        acc_kl_early  = np.zeros(n)   # layer vs target_final
        acc_jsd_early = np.zeros(n)
        acc_top1_early = np.zeros(n)

        acc_kl_draft  = np.zeros(n) if has_draft else None
        acc_jsd_draft = np.zeros(n) if has_draft else None
        acc_top1_draft = np.zeros(n) if has_draft else None

        token_counts = []
        count = 0

        for prompt in tqdm(prompts, desc=f"[{cat}]"):
            # ── Target forward ──
            t_ids = t_tok(prompt, return_tensors="pt").input_ids.to(t_embed_dev)
            token_counts.append(t_ids.shape[1])
            t_out = t_model(t_ids, output_hidden_states=True)

            # bfloat16 fix: final_probs from last hidden state (same code path)
            ref_probs = hidden_to_probs(
                t_out.hidden_states[-1][0, -1, :], t_norm, t_head, t_norm_dev
            )
            ref_top1 = ref_probs.argmax().item()

            # ── Draft forward ──
            if has_draft:
                d_ids = d_tok(prompt, return_tensors="pt").input_ids.to(d_embed_dev)
                d_out = d_model(d_ids)
                draft_probs = F.softmax(d_out.logits[0, -1, :].float(), dim=-1).cpu()
                draft_top1 = draft_probs.argmax().item()

            # ── Per-layer metrics ──
            for li, hs in enumerate(t_out.hidden_states):
                lp = hidden_to_probs(hs[0, -1, :], t_norm, t_head, t_norm_dev)

                # Early exit (vs target final)
                acc_kl_early[li]   += kl_div(ref_probs, lp)
                acc_jsd_early[li]  += js_div(ref_probs, lp)
                acc_top1_early[li] += float(lp.argmax().item() == ref_top1)

                # Draft alignment (vs draft final)
                if has_draft:
                    acc_kl_draft[li]   += kl_div(draft_probs, lp)
                    acc_jsd_draft[li]  += js_div(draft_probs, lp)
                    acc_top1_draft[li] += float(lp.argmax().item() == draft_top1)

            count += 1

        result = {
            "num_layers": n_layers,
            "num_prompts": count,
            "avg_token_count": float(np.mean(token_counts)),
            "kl_early": (acc_kl_early / count).tolist(),
            "jsd_early": (acc_jsd_early / count).tolist(),
            "top1_early": (acc_top1_early / count).tolist(),
        }
        if has_draft:
            result["kl_draft"]   = (acc_kl_draft / count).tolist()
            result["jsd_draft"]  = (acc_jsd_draft / count).tolist()
            result["top1_draft"] = (acc_top1_draft / count).tolist()

        all_results[cat] = result

    # ── Save JSON ──
    target_name = os.path.basename(target_path.rstrip("/"))
    draft_name  = os.path.basename(draft_path.rstrip("/")) if has_draft else None
    json_path = os.path.join(output_dir, f"{target_name}_results.json")
    payload = {
        "target": target_path,
        "draft":  draft_path,
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ── Plot ──
    plot(payload, output_dir, target_name, draft_name)

    return payload


# ─────────────────────────── Plotting ───────────────────────────

def plot(payload: dict, output_dir: str, target_name: str, draft_name: str | None):
    results = payload["results"]
    cats = list(results.keys())
    has_draft = "kl_draft" in results[cats[0]]

    cat_colors = {"short": "#e63946", "medium": "#457b9d", "long": "#2a9d8f"}
    draft_alpha = 0.5

    # ── Figure 1: Early exit (JSD + Top-1) per length category ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(f"Early Exit (Logit Lens): {target_name}", fontsize=13, fontweight="bold")

    for cat in cats:
        r = results[cat]
        layers = np.arange(len(r["jsd_early"]))
        n_tok = r["avg_token_count"]
        color = cat_colors.get(cat, "gray")
        label = f"{cat} (~{n_tok:.0f} tok)"
        axes[0].plot(layers, r["jsd_early"],  "-o", markersize=3, color=color, label=label)
        axes[1].plot(layers, r["top1_early"], "-o", markersize=3, color=color, label=label)

    axes[0].set_ylabel("JSD  (↓ better)")
    axes[0].set_title("Jensen-Shannon Divergence vs. Target Final Layer")
    axes[0].set_ylim(0, 0.75)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("Top-1 Match Rate  (↑ better)")
    axes[1].set_title("Top-1 Token Match Rate with Target Final Layer")
    axes[1].set_xlabel("Layer Index (0 = embedding)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    p = os.path.join(output_dir, f"{target_name}_early_exit.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {p}")

    # ── Figure 2: Draft alignment ──
    if has_draft:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        fig.suptitle(
            f"Draft Model Alignment\nTarget: {target_name}  |  Draft: {draft_name}",
            fontsize=13, fontweight="bold"
        )

        for cat in cats:
            r = results[cat]
            layers = np.arange(len(r["jsd_early"]))
            n_tok = r["avg_token_count"]
            color = cat_colors.get(cat, "gray")
            label = f"{cat} (~{n_tok:.0f} tok)"

            axes[0].plot(layers, r["jsd_early"],  "-",  markersize=3,
                         color=color, label=f"{label} [target-final]")
            axes[0].plot(layers, r["jsd_draft"],  "--", markersize=3,
                         color=color, alpha=0.6, label=f"{label} [draft]")

            axes[1].plot(layers, r["top1_early"],  "-",  markersize=3,
                         color=color, label=f"{label} [target-final]")
            axes[1].plot(layers, r["top1_draft"],  "--", markersize=3,
                         color=color, alpha=0.6, label=f"{label} [draft]")

        axes[0].set_ylabel("JSD  (↓ better)")
        axes[0].set_title("JSD: solid=vs target-final, dashed=vs draft-final")
        axes[0].set_ylim(0, 0.75)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=7, ncol=2)

        axes[1].set_ylabel("Top-1 Match Rate  (↑ better)")
        axes[1].set_title("Top-1 Match: solid=vs target-final, dashed=vs draft-final")
        axes[1].set_xlabel("Layer Index (0 = embedding)")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=7, ncol=2)

        plt.tight_layout()
        p = os.path.join(output_dir, f"{target_name}_draft_align.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {p}")


def plot_model_comparison(payloads: list, output_dir: str):
    """여러 target 모델을 한 그래프에 비교 (normalized layer depth)."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Model Comparison (Normalized Layer Depth)", fontsize=13, fontweight="bold")

    colors = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
    cat = "medium"   # 비교는 medium으로 고정

    for i, payload in enumerate(payloads):
        name = os.path.basename(payload["target"].rstrip("/"))
        r = payload["results"].get(cat)
        if r is None:
            continue
        x = np.linspace(0, 1, len(r["jsd_early"]))
        c = colors[i % len(colors)]
        n_tok = r["avg_token_count"]
        axes[0].plot(x, r["jsd_early"],  "-o", markersize=3, color=c,
                     label=f"{name} early-exit (~{n_tok:.0f} tok)")
        axes[1].plot(x, r["top1_early"], "-o", markersize=3, color=c,
                     label=f"{name} early-exit (~{n_tok:.0f} tok)")
        if "jsd_draft" in r:
            axes[0].plot(x, r["jsd_draft"],  "--", markersize=3, color=c, alpha=0.6,
                         label=f"{name} draft-align")
            axes[1].plot(x, r["top1_draft"], "--", markersize=3, color=c, alpha=0.6,
                         label=f"{name} draft-align")

    for ax in axes:
        ax.set_xlabel("Normalized Layer Depth (0=embed, 1=final)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("JSD")
    axes[0].set_ylim(0, 0.75)
    axes[1].set_ylabel("Top-1 Match Rate")
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    p = os.path.join(output_dir, "comparison_all.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {p}")


# ─────────────────────────── CLI ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Early Exit & Draft Alignment Analysis")
    parser.add_argument("--target", nargs="+", required=True,
                        help="Target model path(s)")
    parser.add_argument("--draft", nargs="+", default=None,
                        help="Draft model path(s), one per target")
    parser.add_argument("--output_dir", default="/home/chokwans99/Parallel_SD/results")
    parser.add_argument("--lengths", nargs="+",
                        default=["short", "medium", "long"],
                        choices=["short", "medium", "long"],
                        help="Length categories to evaluate")
    args = parser.parse_args()

    if args.draft and len(args.draft) != len(args.target):
        parser.error("--draft must have same number of entries as --target")

    os.makedirs(args.output_dir, exist_ok=True)
    payloads = []

    for i, target_path in enumerate(args.target):
        draft_path = args.draft[i] if args.draft else None
        payload = analyze(
            target_path=target_path,
            draft_path=draft_path,
            output_dir=args.output_dir,
            length_categories=args.lengths,
        )
        payloads.append(payload)

    if len(payloads) > 1:
        plot_model_comparison(payloads, args.output_dir)


if __name__ == "__main__":
    main()
