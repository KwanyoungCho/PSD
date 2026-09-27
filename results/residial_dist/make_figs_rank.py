"""Fig 10: how the early-exit distribution differs from the final one.

Reads the rank-profile runs (4 datasets, seed 42).  For every reject position
the tokens are lined up by their p_T rank and p^E is read on the SAME tokens,
so the two curves are directly comparable point by point.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 12, "legend.fontsize": 11,
                     "figure.facecolor": "white"})
SRC = ("/home/chokwans99/PSD/ssd/experiments/proxy_source_ablation/"
       "probe_rankprofile_20260910/out")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
D = {os.path.basename(f)[:-5].split("_")[0]: json.load(open(f))
     for f in sorted(glob.glob(f"{SRC}/*.json"))}
d0 = next(iter(D.values())); L = d0["layers"]
LAY = [40, 56, 72]
CL = {40: "#dc2626", 56: "#ea580c", 72: "#ca8a04"}
M = len(d0["rank_pT"])
x = np.arange(1, M + 1)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

# --- A: rank profile ---------------------------------------------------------
ax = axes[0]
pT = np.mean([d["rank_pT"] for d in D.values()], 0)
ax.plot(x, pT, color="#111827", lw=3, label="$p_T$  (final head)")
for l in LAY:
    pE = np.mean([d["rank_pE"][L.index(l)] for d in D.values()], 0)
    ax.plot(x, pE, color=CL[l], lw=2, label=f"$p^E$  layer {l}")
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels([1, 2, 5, 10, 20, 50])
ax.set(xlabel="token rank under $p_T$", ylabel="mean probability",
       title="A.  Same tokens, two distributions\nthe early exit deflates the head")
ax.legend(); ax.grid(alpha=.3, which="both")
ax.annotate(f"rank 1:  {pT[0]:.3f} $\\to$ "
            f"{np.mean([d['rank_pE'][L.index(56)][0] for d in D.values()]):.3f}\n"
            "at layer 56", (1.05, pT[0]), (1.6, 0.006), fontsize=10.5,
            arrowprops=dict(arrowstyle="->"))

# --- B: consistency ----------------------------------------------------------
ax = axes[1]
for l in LAY:
    lo = np.mean([d["rank_pE_below_pT"][L.index(l)] for d in D.values()], 0)
    per = np.stack([np.array(d["rank_pE_below_pT"][L.index(l)])
                    for d in D.values()])
    ax.plot(x, lo, color=CL[l], lw=2, label=f"layer {l}")
    ax.fill_between(x, per.min(0), per.max(0), color=CL[l], alpha=.15, lw=0)
ax.axhline(.5, color="k", lw=1.4, ls="--")
ax.text(1.2, .515, "coin flip", fontsize=10.5)
ax.set_xscale("log"); ax.set_ylim(0.3, 0.85)
ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels([1, 2, 5, 10, 20, 50])
ax.set(xlabel="token rank under $p_T$",
       ylabel="$P(\\,p^E < p_T\\,)$ at that rank",
       title="B.  Is it consistent?\nband = spread over the 4 datasets")
ax.legend(); ax.grid(alpha=.3, which="both")

# --- C: where p_T's top-1 lands ----------------------------------------------
ax = axes[2]
bins = d0["rank_top1_bins"]
V = np.stack([[np.mean([d["rank_of_pT_top1"][L.index(l)][j]
                        for d in D.values()]) for j in range(len(bins))]
              for l in LAY])
bot = np.zeros(len(LAY))
sh = ["#166534", "#65a30d", "#eab308", "#f97316", "#b91c1c"]
for j, lab in enumerate(bins):
    ax.bar(range(len(LAY)), V[:, j], .55, bottom=bot, color=sh[j],
           label=f"rank {lab}")
    for i in range(len(LAY)):
        if V[i, j] > .05:
            ax.text(i, bot[i] + V[i, j] / 2, f"{V[i,j]*100:.0f}%",
                    ha="center", va="center", color="white",
                    fontsize=11, weight="bold")
    bot += V[:, j]
ax.set_xticks(range(len(LAY)))
ax.set_xticklabels([f"layer {l}" for l in LAY])
ax.set(ylabel="share of reject mass", ylim=(0, 1.02),
       title="C.  Where does $p_T$'s top-1 token\nsit in $p^E$'s ordering?")
ax.legend(fontsize=9.5, loc="lower right"); ax.grid(alpha=.3, axis="y")

fig.tight_layout(); fig.savefig(f"{FIG}/10_pE_vs_pT_rank.png", dpi=140)
print("wrote 10_pE_vs_pT_rank.png")
