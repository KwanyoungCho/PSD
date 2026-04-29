"""
Combined 3-panel figure (TinyLlama-1.1B draft only):
  (a) Reject Position Distribution
  (b) Reject Position Prediction (Recall@1, Recall@3)
  (c) Correction Token Prediction (Recall@1, Recall@3)
No suptitle. Consistent style with existing layerskip_70B_v4 plots.

Outputs:
  - fig_reject_2panel.png         : (a)+(b) side-by-side, single-figure use
  - fig_correction_1panel.png     : (c) standalone, single-figure use
  - combined_3panel_compact.png   : 1x3 compact for `figure*` (2-col span)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 15, "axes.titlesize": 18,
    "axes.labelsize": 17, "legend.fontsize": 14, "figure.facecolor": "white",
    "mathtext.fontset": "cm", "xtick.labelsize": 14, "ytick.labelsize": 14,
})

DATA_PATH = "results/layerskip_70B_v4/correction_layerskip-llama2-70B.json"
OUT_PATH_1 = "results/layerskip_70B_v4/fig_reject_2panel.png"
OUT_PATH_2 = "results/layerskip_70B_v4/fig_correction_1panel.png"
OUT_PATH_3 = "results/layerskip_70B_v4/combined_3panel_compact.png"

with open(DATA_PATH) as f:
    data = json.load(f)

n_layers  = data["n_layers"]  # 80
cp_labels = [f"cp{c}" for c in data["checkpoints"]]
layers    = np.arange(n_layers + 1)  # 0..80

# TinyLlama-1.1B only (first draft)
draft = data["drafts"][0]
dd = draft["data"]
COLOR_R1 = "#2196F3"   # Recall@1 / Top-1 (blue)
COLOR_R3 = "#FF9800"   # Recall@3 / Top-3 (orange/yellow)
COLOR_BAR = "#4CAF50"  # bar chart (green)
W = 10  # window size


# ── Helpers ──
def avg_hazard_metric(dd, cp_labels, key):
    arrays, weights = [], []
    for cp in cp_labels:
        hz = dd[cp].get("hazard", {})
        nw = hz.get("n_windows", 0)
        if nw == 0:
            continue
        arrays.append(np.array(hz[key]))
        weights.append(nw)
    if not arrays:
        return None
    total = sum(weights)
    return sum(a * w for a, w in zip(arrays, weights)) / total


def wavg(dd, cp_labels, key, weight_key="count"):
    arr, wts = [], []
    for cp in cp_labels:
        fd = dd[cp]
        w = fd[weight_key]
        if w == 0:
            continue
        arr.append(np.array(fd[key], dtype=float))
        wts.append(w)
    if not arr:
        return None
    total = sum(wts)
    return sum(a * w for a, w in zip(arr, wts)) / total


def scalar_wavg(dd, cp_labels, key_path, weight_key="count"):
    vals, wts = [], []
    for cp in cp_labels:
        fd = dd[cp]
        w = fd[weight_key]
        if w == 0:
            continue
        v = fd
        for k in key_path:
            v = v[k]
        vals.append(float(v))
        wts.append(w)
    if not vals:
        return None
    total = sum(wts)
    return sum(v * w for v, w in zip(vals, wts)) / total


def find_first_crossover(vals, baseline):
    for i, v in enumerate(vals):
        if v >= baseline:
            return i
    return None


# ══════════════════════════════════════════════════════════════════════
# Figure 1: (a) Reject Position Distribution + (b) Reject Position Prediction
# ══════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

# ── (a) Reject Position Distribution ──
ax = axes1[0]
reject_hist = np.zeros(W + 1)
total_count = 0
for cp in cp_labels:
    fd = dd[cp]
    for pos_str, cnt in fd.get("reject_pos_dist", {}).items():
        reject_hist[int(pos_str)] += cnt
    reject_hist[W] += fd.get("all_accept_count", 0)
    total_count += fd["count"] + fd.get("all_accept_count", 0)
if total_count > 0:
    reject_hist /= total_count

x = np.arange(W + 1)
ax.bar(x, reject_hist, 0.6, color=COLOR_BAR, alpha=0.85,
       edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(W)] + ["accept"], fontsize=13)
ax.set_ylabel("Fraction", fontsize=17)
ax.grid(True, alpha=0.3, axis="y")
ax.text(0.5, -0.18, "(a) Reject Position Distribution", transform=ax.transAxes,
        fontsize=16, fontweight="bold", ha="center", va="top")

# ── (b) Reject Position Prediction ──
ax = axes1[1]
top1 = avg_hazard_metric(dd, cp_labels, "top1_hit")
top3 = avg_hazard_metric(dd, cp_labels, "top3_hit")
if top1 is not None:
    ax.plot(layers, top1, color=COLOR_R1, linewidth=1.8, linestyle="-",
            label="Recall@1")
if top3 is not None:
    ax.plot(layers, top3, color=COLOR_R3, linewidth=1.8, linestyle="-",
            label="Recall@3")
ax.set_xlabel("Layer Index", fontsize=17)
ax.set_ylabel("Recall", fontsize=17)
ax.set_xlim(0, n_layers)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=14, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.text(0.5, -0.18, "(b) Reject Position Prediction", transform=ax.transAxes,
        fontsize=16, fontweight="bold", ha="center", va="top")

fig1.tight_layout()
fig1.subplots_adjust(bottom=0.15)
fig1.savefig(OUT_PATH_1, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH_1}")

# ══════════════════════════════════════════════════════════════════════
# Figure 2: (a) Correction Token Prediction (original style)
# ══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 13, "legend.fontsize": 10,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})
fig2, ax = plt.subplots(1, 1, figsize=(9, 5))

corr_top1 = wavg(dd, cp_labels, "corr_top1")
corr_top3 = wavg(dd, cp_labels, "corr_top3")
bl_top1   = scalar_wavg(dd, cp_labels, ["draft_baseline", "top1"])
bl_top3   = scalar_wavg(dd, cp_labels, ["draft_baseline", "top3"])

if corr_top1 is not None:
    ax.plot(layers, corr_top1, color=COLOR_R1, linewidth=1.8, linestyle="-",
            label="Early-exit Recall@1")
if corr_top3 is not None:
    ax.plot(layers, corr_top3, color=COLOR_R3, linewidth=1.8, linestyle="-",
            label="Early-exit Recall@3")

# Draft baselines
if bl_top1 is not None:
    ax.axhline(bl_top1, color=COLOR_R1, linestyle="--", linewidth=1.3, alpha=0.6,
               label=f"Draft-only @1 ({bl_top1:.3f})")
if bl_top3 is not None:
    ax.axhline(bl_top3, color=COLOR_R3, linestyle="--", linewidth=1.3, alpha=0.6,
               label=f"Draft-only @3 ({bl_top3:.3f})")

# Crossover annotations
for vals, bl, ls_color, y_off_base in [
    (corr_top1, bl_top1, COLOR_R1, 14),
    (corr_top3, bl_top3, COLOR_R3, 26),
]:
    if vals is not None and bl is not None:
        cross_l = find_first_crossover(vals, bl)
        if cross_l is not None:
            cross_val = vals[cross_l]
            ax.plot(cross_l, cross_val, marker="^", markersize=11,
                    color=ls_color, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.0)
            ax.axvline(cross_l, color=ls_color, linewidth=0.9,
                       linestyle=":", alpha=0.5)
            ax.annotate(
                f"Layer {cross_l}",
                xy=(cross_l, cross_val),
                xytext=(5, y_off_base),
                textcoords="offset points",
                fontsize=9.5, fontweight="bold", color=ls_color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=ls_color, alpha=0.85, lw=0.8),
            )

ax.set_xlabel("Layer Index", fontsize=13)
ax.set_ylabel("Recall", fontsize=13)
ax.set_xlim(0, n_layers)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT_PATH_2, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH_2}")

# ══════════════════════════════════════════════════════════════════════
# Figure 3: Compact 1x3 combined (for dual-column paper, `figure*` span)
# ══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10.5, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})
fig3, axes3 = plt.subplots(1, 3, figsize=(7.0, 2.9))

# ── (a) Reject Position Distribution (truncated 5..8 + standard break mark) ──
ax = axes3[0]
# Show 0..4, break, 9, accept (drop positions 5..8)
KEEP_LO = 5            # positions 0..4
BREAK_X = KEEP_LO      # x=5 is the break gap (no bar)
NINE_X  = KEEP_LO + 1  # x=6 for position 9
ACC_X   = KEEP_LO + 3  # x=8 for accept (extra space so "9" / "accept" labels don't collide)

ax.bar(np.arange(KEEP_LO), reject_hist[:KEEP_LO], 0.7,
       color=COLOR_BAR, alpha=0.85, edgecolor="white", linewidth=0.4)
ax.bar([NINE_X], [reject_hist[9]], 0.7,
       color=COLOR_BAR, alpha=0.85, edgecolor="white", linewidth=0.4)
ax.bar([ACC_X], [reject_hist[W]], 0.7,
       color=COLOR_BAR, alpha=0.85, edgecolor="white", linewidth=0.4)

ax.set_xticks(list(range(KEEP_LO)) + [NINE_X, ACC_X])
ax.set_xticklabels([str(i) for i in range(KEEP_LO)] + ["9", "accept"], fontsize=9)
ax.set_xlim(-0.6, ACC_X + 0.6)

# Standard broken-axis mark: two parallel diagonal slashes at x=BREAK_X
slash_h = 0.10   # half-width of each slash (data x)
slash_v = 0.030  # half-height (axes-fraction y)
sep     = 0.18   # spacing between the two parallel slashes (data x)
for xc in (BREAK_X - sep / 2, BREAK_X + sep / 2):
    ax.plot([xc - slash_h, xc + slash_h], [-slash_v, slash_v],
            transform=ax.get_xaxis_transform(),
            color="0.15", linewidth=1.1, clip_on=False, zorder=10)

ax.set_xlabel("Reject Position")
ax.set_ylabel("Fraction")
ax.grid(True, alpha=0.3, axis="y")

# ── (b) Reject Position Prediction ──
ax = axes3[1]
if top1 is not None:
    ax.plot(layers, top1, color=COLOR_R1, linewidth=1.4, label="Recall@1")
if top3 is not None:
    ax.plot(layers, top3, color=COLOR_R3, linewidth=1.4, label="Recall@3")
ax.set_xlabel("Layer Index")
ax.set_ylabel("Recall")
ax.set_xlim(0, n_layers)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([0, 20, 40, 60, 80])
ax.legend(loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.3)

# ── (c) Correction Token Prediction ──
ax = axes3[2]
if corr_top1 is not None:
    ax.plot(layers, corr_top1, color=COLOR_R1, linewidth=1.4, label="Recall@1")
if corr_top3 is not None:
    ax.plot(layers, corr_top3, color=COLOR_R3, linewidth=1.4, label="Recall@3")
if bl_top1 is not None:
    ax.axhline(bl_top1, color=COLOR_R1, linestyle="--", linewidth=1.0, alpha=0.6,
               label="Draft @1")
    ax.text(2, bl_top1 + 0.025, f"{bl_top1:.2f}",
            fontsize=7.5, color=COLOR_R1, fontweight="bold",
            ha="left", va="bottom")
if bl_top3 is not None:
    ax.axhline(bl_top3, color=COLOR_R3, linestyle="--", linewidth=1.0, alpha=0.6,
               label="Draft @3")
    ax.text(2, bl_top3 + 0.025, f"{bl_top3:.2f}",
            fontsize=7.5, color=COLOR_R3, fontweight="bold",
            ha="left", va="bottom")

for vals, bl, ls_color, y_off in [
    (corr_top1, bl_top1, COLOR_R1, 8),
    (corr_top3, bl_top3, COLOR_R3, 12),
]:
    if vals is not None and bl is not None:
        cl = find_first_crossover(vals, bl)
        if cl is not None:
            cv = vals[cl]
            ax.plot(cl, cv, marker="^", markersize=7,
                    color=ls_color, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.7)
            ax.axvline(cl, color=ls_color, linewidth=0.7,
                       linestyle=":", alpha=0.5)
            ax.annotate(
                f"L{cl}", xy=(cl, cv), xytext=(4, y_off),
                textcoords="offset points",
                fontsize=7.5, fontweight="bold", color=ls_color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec=ls_color, alpha=0.9, lw=0.5),
            )

ax.set_xlabel("Layer Index")
ax.set_ylabel("Recall")
ax.set_xlim(0, n_layers)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([0, 20, 40, 60, 80])
ax.legend(loc="lower right", framealpha=0.9, fontsize=7, ncol=2,
          columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
ax.grid(True, alpha=0.3)

# Panel labels below each subplot (no box). Use figure-fraction so they line up.
fig3.tight_layout(pad=0.6, w_pad=0.9)
fig3.subplots_adjust(bottom=0.30)
panel_labels = [
    "(a) Reject Position Dist.",
    "(b) Reject Position Pred.",
    "(c) Correction Token Pred.",
]
for ax_, label in zip(axes3, panel_labels):
    pos = ax_.get_position()
    cx = (pos.x0 + pos.x1) / 2
    fig3.text(cx, 0.03, label, ha="center", va="bottom",
              fontsize=9.5, fontweight="bold")

fig3.savefig(OUT_PATH_3, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PATH_3}")
