"""70B-specific compare_breakdown plot — DUET exit layers 40/47/53."""
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

BASE = sys.argv[1] if len(sys.argv) > 1 else "/home/chokwans99/PSD/ssd/tmp/final_exp2_quant_70b"
CONFIGS = [
    ("baseline_k7_geo",          "SSD (baseline)\nK=7 geo"),
    ("duet_k5_f4_dfo2_exit40",   "DUET K=5\nexit=40"),
    ("duet_k5_f4_dfo2_exit47",   "DUET K=5\nexit=47"),
    ("duet_k5_f4_dfo2_exit53",   "DUET K=5\nexit=53"),
]
COLORS = {
    "glue":"#17becf","phase1_build":"#a9cce3","phase1_prep":"#807dba","phase1_replay":"#3f007d",
    "phase2_build":"#c7e9c0","phase2_prep":"#74c476","phase2_replay":"#00441b","proxy_wait":"#ff7f00",
    "merge_cache":"#252525","tree_prep":"#fcae91","tree_replay":"#cb181d","draft_recv_cmd":"#bdbdbd",
    "hit_cache_respond":"#78c679","draft_send_response":"#006d2c","draft_glue_replay":"#1f78b4",
    "verify_setup":"#d9d9d9","graph_pre":"#e41a1c","exit_logits":"#bdbdbd","proxy_compute_send":"#ffbf00",
    "graph_post":"#a50f15","final_logits":"#737373","verify_sample_accept":"#fc9272",
    "target_spec_wait":"#fee090","target_postprocess":"#fdae61","verify_replay":"#b30000",
}

def _load(cfg):
    p = f"{BASE}/{cfg}/duet_per_step_contribution.csv"
    return pd.read_csv(p) if os.path.isfile(p) else None

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
for ax, proc in zip(axes, ["target", "draft"]):
    labels_set = set()
    per_cfg = {}
    for cfg, _ in CONFIGS:
        df = _load(cfg)
        if df is None: per_cfg[cfg] = pd.Series(dtype=float); continue
        sub = df[df["proc"] == proc].set_index("label")["ms_per_step"]
        per_cfg[cfg] = sub
        labels_set.update(sub.index)
    if "baseline_k7_geo" in per_cfg and len(per_cfg["baseline_k7_geo"]):
        order = list(per_cfg["baseline_k7_geo"].sort_values(ascending=False).index)
    else: order = sorted(labels_set)
    for lb in labels_set:
        if lb not in order: order.append(lb)
    bottoms = [0.0] * len(CONFIGS)
    for lb in order:
        vals = [per_cfg[cfg].get(lb, 0.0) for cfg, _ in CONFIGS]
        ax.bar(range(len(CONFIGS)), vals, bottom=bottoms,
               color=COLORS.get(lb, "#bdc3c7"), label=lb, edgecolor="black", linewidth=0.3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, b in enumerate(bottoms):
        ax.text(i, b + 0.5, f"{b:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([lbl for _, lbl in CONFIGS], fontsize=9)
    ax.set_ylabel("ms / spec step")
    ax.set_title(f"{proc} side — per-phase contribution per step (layerskip-llama2-70B)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
plt.tight_layout(rect=[0, 0, 0.88, 1])
out = f"{BASE}/compare_breakdown.png"
plt.savefig(out, dpi=130)
print(f"-> {out}")
