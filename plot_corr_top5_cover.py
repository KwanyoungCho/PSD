"""
Plot corr_top5 cover comparison: early-exit proxy residual vs draft baselines.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.size": 11,
})

DATA_PATH = "/home/chokwans99/Parallel_SD/results/layerskip_70B/correction_layerskip-llama2-70B.json"
OUT_PATH  = "/home/chokwans99/Parallel_SD/results/layerskip_70B/corr_top5_cover_comparison.png"

with open(DATA_PATH) as f:
    data = json.load(f)

n_layers = data["n_layers"]  # 80
checkpoints = [f"cp{c}" for c in data["checkpoints"]]

# Checkpoint → human-readable subtitle
cp_to_output_len = {
    "cp1":   "Output length = 1",
    "cp128": "Output length = 128",
    "cp256": "Output length = 256",
    "cp512": "Output length = 512",
}

drafts = data["drafts"]
draft_colors = ["#2196F3", "#FF9800"]

n_cp = len(checkpoints)
fig, axes = plt.subplots(1, n_cp, figsize=(5.8 * n_cp, 5.2), sharey=True)
if n_cp == 1:
    axes = [axes]

layers = np.arange(n_layers + 1)  # 0..80

for col, cp_key in enumerate(checkpoints):
    ax = axes[col]

    for di, draft_info in enumerate(drafts):
        fd = draft_info["data"][cp_key]
        corr_top5 = np.array(fd["corr_top5"])
        baseline  = fd["draft_baseline"]["top5"]

        # Subscript index and name for this draft
        sub = str(di + 1)
        dname = ["TinyLlama-1.1B", "Llama2-7B"][di]

        # Proxy residual curve (solid)
        ax.plot(layers, corr_top5, color=draft_colors[di], linewidth=1.8,
                label=f"Early-exit residual ({dname})  "
                      r"$[\hat{p}_E^{(l)} - p_{D_" + sub + r"}]_+$")

        # Draft baseline (dashed)
        ax.axhline(baseline, color=draft_colors[di], linestyle="--",
                   linewidth=1.3, alpha=0.7,
                   label=f"Draft: {dname}  "
                         r"($p_{D_" + sub + r"}$)"
                         f"  {baseline:.1%}")

        # Crossover marker
        crossover = np.where(corr_top5 > baseline)[0]
        if len(crossover) > 0:
            first = crossover[0]
            ax.plot(first, corr_top5[first], marker="v", markersize=10,
                    color=draft_colors[di], zorder=5)
            ax.annotate(f"L{first}", (first, corr_top5[first]),
                        textcoords="offset points", xytext=(6, -16),
                        fontsize=9, fontweight="bold", color=draft_colors[di])

    counts = [d["data"][cp_key]["count"] for d in drafts]
    if counts[0] == counts[1]:
        count_str = f"n_samples = {counts[0]}"
    else:
        count_str = f"n_samples: $D_1$={counts[0]}, $D_2$={counts[1]}"
    ax.set_title(f"{cp_to_output_len.get(cp_key, cp_key)}  ({count_str})", fontsize=11)
    ax.set_xlabel("Layer index $l$", fontsize=11)
    ax.set_xlim(-1, n_layers + 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    if col == 0:
        ax.set_ylabel("Correction Top-5 Cover", fontsize=11)

    # Legend: group solid/dashed visually
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

fig.suptitle(
    "Correction Top-5 Cover:  Early-Exit  vs.  Draft\n"
    r"(Target: LayerSkip-Llama2-70B  /  $D_1$: TinyLlama-1.1B,  $D_2$: Llama2-7B)",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# Crossover summary
print("\n=== Crossover Summary ===")
for cp_key in checkpoints:
    lbl = cp_to_output_len.get(cp_key, cp_key)
    print(f"\n{lbl}:")
    for di, draft_info in enumerate(drafts):
        fd = draft_info["data"][cp_key]
        corr_top5 = np.array(fd["corr_top5"])
        baseline  = fd["draft_baseline"]["top5"]
        crossover = np.where(corr_top5 > baseline)[0]
        first = crossover[0] if len(crossover) > 0 else -1
        val79 = corr_top5[-2]
        dname = ["TinyLlama-1.1B", "Llama2-7B"][di]
        print(f"  D{di+1} ({dname}): baseline={baseline:.1%}, "
              f"first_beat=L{first}, L[-2]={val79:.1%}")
