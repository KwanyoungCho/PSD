#!/usr/bin/env python3
"""bscale32 figures — B=16/32 extension + C-fairness story (fig1..fig5).

Parses run.log files directly (bscale32 scan/confirm32 + bscale + pb_sweep),
no manual data entry. Run with the ssd env python, MPLCONFIGDIR=/tmp/matplotlib.

Series color = entity, fixed categorical order (dataviz reference palette,
validated): slot1 blue = DUET-opt, slot2 green = C-opt, slot3 magenta =
C-fixed (k7f6). Magenta is sub-3:1 on the light surface -> relief rule:
dashed linestyle + direct labels everywhere it appears.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BASE = Path(__file__).resolve().parent            # .../b_gt1/bscale32
BSC = BASE.parent / "bscale"                       # .../b_gt1/bscale
PB = BASE.parent / "pb_sweep"                      # .../b_gt1/pb_sweep
CONF = BASE / "confirm32"
FIGS = BASE / "figs"
FIGS.mkdir(exist_ok=True)

# ---- palette (dataviz reference instance, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DUET = "#2a78d6"      # categorical slot 1 (blue)
COPT = "#008300"      # categorical slot 2 (green)
CFIX = "#e87ba4"      # categorical slot 3 (magenta) — always direct-labeled
# sequential blue ramp (magnitude job, fig4)
BLUES = ["#b7d3f6", "#9cc2f2", "#6da7ec", "#4a90e0", "#2a78d6", "#1c5cab", "#184f95"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def tps_of(log: Path):
    if not log.exists():
        return None
    m = re.findall(r"Final Decode Throughput: ([\d.]+)tok/s",
                   log.read_text(errors="replace"))
    return float(m[-1]) if m else None


def reps(dirpat, root):
    out = []
    for r in (1, 2, 3):
        v = tps_of(root / f"{dirpat}_r{r}" / "run.log")
        if v is not None:
            out.append(v)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def style_axes(ax):
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def logb_axis(ax, bs):
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(bs))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("batch size B (log2)")


# ------------------------------------------------------------------
# data assembly
# ------------------------------------------------------------------
BS = [1, 2, 4, 8, 16, 32]

# DUET-opt / C-opt: B=1 same-regime anchors (bscale, 1 run), else confirm32
DUET_R = {1: [v for v in [tps_of(BSC / "b1_e9k24_jit" / "run.log")] if v]}
COPT_R = {1: [v for v in [tps_of(BSC / "b1_c" / "run.log")] if v]}
for b in (2, 4, 8, 16, 32):
    DUET_R[b] = reps(f"b{b}_duet", CONF)
    COPT_R[b] = reps(f"b{b}_c", CONF)

# C-fixed (k7f6): B=1 = C-opt anchor; B=2/4 pb_sweep confirm, B=8 bscale
# confirm (all 3-rep interleaved ns=20); B=16 bscale32 scan (1 run ns=16);
# B=32 DNF (draft CG capture OOM).
CFIX_R = {
    1: COPT_R[1],
    2: reps("b2_c", PB / "confirm"),
    4: reps("b4_c", PB / "confirm"),
    8: reps("b8_c", BSC / "confirm"),
    16: [v for v in [tps_of(BASE / "cb16_k7f6" / "run.log")] if v],
    32: [],  # DNF
}

DUET_LBL = {1: "E9K24_jit", 2: "k6x5_d3p1", 4: "k3x3_d4p1", 8: "k2x2_d5p1",
            16: "k1x1_d5p1", 32: "k1x1_d4p1"}
COPT_LBL = {1: "k7f6", 2: "k5f6", 4: "k3f6", 8: "k3f6", 16: "k2f3", 32: "k2f2"}

DUET_K1 = {1: 9, 2: 6, 4: 3, 8: 2, 16: 1, 32: 1}
COPT_K = {1: 7, 2: 5, 4: 3, 8: 3, 16: 2, 32: 2}


def series(R, bs):
    m = [mean(R[b]) for b in bs]
    lo = [mean(R[b]) - min(R[b]) for b in bs]
    hi = [max(R[b]) - mean(R[b]) for b in bs]
    return m, [lo, hi]


# ------------------------------------------------------------------
# fig1 — aggregate decode TPS vs B, three series + DNF marker
# ------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    bs_d = [b for b in BS if DUET_R[b]]
    bs_c = [b for b in BS if COPT_R[b]]
    bs_f = [b for b in BS if CFIX_R[b]]
    dm, de = series(DUET_R, bs_d)
    cm, ce = series(COPT_R, bs_c)
    fm, fe = series(CFIX_R, bs_f)
    ax.errorbar(bs_f, fm, yerr=fe, color=CFIX, lw=1.8, ls=(0, (4, 2)),
                marker="^", ms=6, capsize=3, label="C fixed k7f6 (old baseline)",
                zorder=2)
    ax.errorbar(bs_c, cm, yerr=ce, color=COPT, lw=2, marker="s", ms=6.5,
                capsize=3, label="C per-B optimized (fair)", zorder=3)
    ax.errorbar(bs_d, dm, yerr=de, color=DUET, lw=2, marker="o", ms=7,
                capsize=3, label="DUET per-B best shape", zorder=3)
    for b, v in zip(bs_d, dm):
        dy = -17 if b >= 4 else 9
        ax.annotate(f"{v:.0f}", (b, v), textcoords="offset points",
                    xytext=(0, dy), ha="center", color=INK2, fontsize=8.5)
    for b, v in zip(bs_c, cm):
        if b >= 8:
            ax.annotate(f"{v:.0f}", (b, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", color=INK2, fontsize=8.5)
    # direct label for the sub-contrast magenta series + DNF marker at B=32
    ax.annotate(f"C k7f6: {fm[-1]:.0f}", (bs_f[-1], fm[-1]),
                textcoords="offset points", xytext=(-10, -6), ha="right",
                color=INK2, fontsize=8.5)
    ax.scatter([32], [fm[-1]], marker="x", s=90, color=INK, zorder=4)
    ax.annotate("k7f6 @ B=32: DNF\n(draft CG capture OOM)", (32, fm[-1]),
                textcoords="offset points", xytext=(-6, -30), ha="right",
                color=INK, fontsize=8.5)
    logb_axis(ax, BS)
    style_axes(ax)
    ax.set_ylabel("aggregate decode TPS (tok/s)")
    ax.set_title("Decode throughput vs batch size — optimum-vs-optimum rebuild\n"
                 "(B≥2 = 3-rep interleaved confirms; B=1 anchors and "
                 "C-k7f6 B=16 are single runs)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_tps_vs_B.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig2 — DUET advantage vs C-opt AND vs C-fixed
# ------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    bs = [b for b in BS if DUET_R[b] and COPT_R[b]]
    adv_opt, band = [], []
    for b in bs:
        d, c = DUET_R[b], COPT_R[b]
        adv_opt.append(100.0 * (mean(d) / mean(c) - 1.0))
        if len(d) >= 3 and len(c) >= 3:
            band.append("D" if min(d) > max(c) else ("C" if min(c) > max(d) else ""))
        else:
            band.append("")
    bs_f = [b for b in BS if DUET_R[b] and CFIX_R[b]]
    adv_fix = [100.0 * (mean(DUET_R[b]) / mean(CFIX_R[b]) - 1.0) for b in bs_f]
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(bs_f, adv_fix, color=CFIX, lw=1.8, ls=(0, (4, 2)), marker="^",
            ms=6, label="vs C fixed k7f6 (old story)", zorder=2)
    ax.plot(bs, adv_opt, color=DUET, lw=2, marker="o", ms=7,
            label="vs C per-B optimized (fair)", zorder=3)
    for b, a, bc in zip(bs, adv_opt, band):
        lbl = f"{a:+.1f}%"
        if bc == "D":
            lbl += "\nband-clear D"
        elif bc == "C":
            lbl += "\nband-clear C"
        ax.annotate(lbl, (b, a), textcoords="offset points", xytext=(0, -24),
                    ha="center", color=INK if bc else INK2, fontsize=8.5,
                    fontweight="bold" if bc else "normal")
    for b, a in zip(bs_f, adv_fix):
        ax.annotate(f"{a:+.1f}%", (b, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=INK2, fontsize=8)
    ax.annotate("vs k7f6 @ B=32:\nundefined (DNF)", (32, adv_fix[-1]),
                textcoords="offset points", xytext=(0, 16), ha="right",
                color=INK2, fontsize=8.5)
    logb_axis(ax, BS)
    style_axes(ax)
    ax.set_ylabel("DUET advantage (%)")
    ax.set_title("The amplification curve, refitted — the old story was the "
                 "baseline's shape\n(band-clear = no overlap between 3-rep "
                 "min/max bands)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.12 * (ymax - ymin), ymax + 0.1 * (ymax - ymin))
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_advantage_vs_B.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig3 — shape law: DUET K1* and C k* vs B
# ------------------------------------------------------------------
def fig3():
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
    ax.plot(BS, [DUET_K1[b] for b in BS], color=DUET, lw=2.6, marker="o",
            ms=8, label="DUET optimal K1 (phase-1 depth)")
    ax.plot(BS, [COPT_K[b] for b in BS], color=COPT, lw=2, marker="s",
            ms=6.5, label="C optimal k (chain depth)")
    ax.annotate("K1 = 9", (1, 9), textcoords="offset points", xytext=(0, -16),
                ha="center", color=INK2, fontsize=8.5)
    ax.annotate("k = 7", (1, 7), textcoords="offset points", xytext=(0, 9),
                ha="center", color=INK2, fontsize=8.5)
    ax.annotate("K1* = 1", (32, 1), textcoords="offset points", xytext=(10, 0),
                va="center", color=INK2, fontsize=8.5)
    ax.annotate("k* = 2", (32, 2), textcoords="offset points", xytext=(10, 0),
                va="center", color=INK2, fontsize=8.5)
    # the one transition worth calling out: K1 2->1 happens at B=16
    ax.annotate("2→1 transition at B=16\n(K1=1 loses at B=8: probe 209.1 < 213.5)",
                (16, 1), textcoords="offset points", xytext=(-12, -26),
                ha="center", color=INK2, fontsize=8)
    logb_axis(ax, BS)
    style_axes(ax)
    ax.set_xlim(right=32 * 1.5)
    ax.set_ylim(0, 10.2)
    ax.set_ylabel("optimal speculation depth")
    ax.set_title("Both systems obey the same shape law — depth collapses as B "
                 "grows\n(DUET K1 9→6→3→2→1→1, "
                 "C k 7→5→3→3→2→2)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_shape_law.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig4 — C feasibility map: draft CG capture rows vs the memory wall
# ------------------------------------------------------------------
def fig4():
    # rows/seq for async C = MQ_LEN = (k+1)*f; total capture rows = MQ_LEN*B
    configs = [("k7f6", 48), ("k5f6", 36), ("k3f6", 24), ("k5f3", 18),
               ("k3f3", 12), ("k2f3", 9), ("k2f2", 6)]
    bs = [1, 2, 4, 8, 16, 32]
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    # the measured wall: 1152 rows fits (cb32_k5f6), 1536 rows OOM (k7f6 smoke)
    ax.axhspan(1152, 1536, color=GRID, alpha=0.7, zorder=1)
    ax.axhline(1536, color=INK2, lw=1.2, ls=(0, (2, 2)))
    ax.annotate("OOM (measured): 1536 rows — k7f6 × B=32", (1.05, 1536),
                xytext=(0, 5), textcoords="offset points", color=INK,
                fontsize=8.5)
    ax.annotate("fits (measured): 1152 rows — k5f6 × B=32", (1.05, 1152),
                xytext=(0, -13), textcoords="offset points", color=INK2,
                fontsize=8.5)
    for (name, rps), color in zip(configs, BLUES[::-1]):
        ys = [rps * b for b in bs]
        ax.plot(bs, ys, color=color, lw=2, marker="o", ms=5, zorder=3)
        dy = 7 if name == "k7f6" else 0
        ax.annotate(f"{name} ({rps}/seq)", (bs[-1], ys[-1]),
                    textcoords="offset points", xytext=(12, dy), va="center",
                    color=INK2, fontsize=8.5)
    # measured endpoint states at B=32 and the k7f6 B=16 fit
    ax.scatter([32], [48 * 32], marker="x", s=110, color=INK, zorder=4)
    ax.scatter([16], [48 * 16], marker="o", s=46, facecolors=SURFACE,
               edgecolors=INK, zorder=4)
    ax.annotate("768: fits", (16, 768), textcoords="offset points",
                xytext=(0, 8), ha="center", color=INK2, fontsize=8)
    ax.set_yscale("log", base=2)
    ax.set_yticks([8, 32, 128, 512, 1152, 1536, 2048])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    logb_axis(ax, bs)
    ax.set_xlim(right=32 * 2.6)
    style_axes(ax)
    ax.grid(axis="y", visible=True)
    ax.set_ylabel("draft CUDA-graph capture rows = (k+1)×f×B (log2)")
    ax.set_title("C feasibility map — the draft-GPU CG capture memory wall "
                 "(24 GB)\nfixed k7f6 hits the wall at B=32; per-B shapes stay "
                 "far below it", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_feasibility_map.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig5 — aggregate + per-seq TPS vs B (two panels, one axis each)
# ------------------------------------------------------------------
def fig5():
    fig, (axa, axp) = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=150)
    bs_d = [b for b in BS if DUET_R[b]]
    bs_c = [b for b in BS if COPT_R[b]]
    bs_f = [b for b in BS if CFIX_R[b]]
    dm, _ = series(DUET_R, bs_d)
    cm, _ = series(COPT_R, bs_c)
    fm, _ = series(CFIX_R, bs_f)
    for ax, transform, ylab, title in (
        (axa, lambda v, b: v, "aggregate decode TPS (tok/s)",
         "aggregate throughput"),
        (axp, lambda v, b: v / b, "per-seq decode TPS (aggregate / B)",
         "per-seq token rate (latency proxy)"),
    ):
        ax.plot(bs_f, [transform(v, b) for v, b in zip(fm, bs_f)], color=CFIX,
                lw=1.8, ls=(0, (4, 2)), marker="^", ms=5.5,
                label="C fixed k7f6")
        ax.plot(bs_c, [transform(v, b) for v, b in zip(cm, bs_c)], color=COPT,
                lw=2, marker="s", ms=6, label="C per-B optimized")
        ax.plot(bs_d, [transform(v, b) for v, b in zip(dm, bs_d)], color=DUET,
                lw=2, marker="o", ms=6.5, label="DUET per-B best")
        logb_axis(ax, BS)
        style_axes(ax)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
    for b, v in zip(bs_d, dm):
        axp.annotate(f"{v/b:.0f}", (b, v / b), textcoords="offset points",
                     xytext=(0, 8), ha="center", color=INK2, fontsize=8)
    axa.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Serving frontier — batching buys aggregate throughput at a "
                 "per-seq latency cost (same runs as fig1)", fontsize=10.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "fig5_per_seq.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    for b in BS:
        d = DUET_R[b]; c = COPT_R[b]; f = CFIX_R[b]
        print(f"B={b}: DUET {[round(x,2) for x in d]} mean "
              f"{mean(d):.2f} | C-opt {[round(x,2) for x in c]} mean "
              f"{mean(c):.2f}" if d and c else f"B={b}: incomplete")
        print(f"   C-fixed: {[round(x,2) for x in f] if f else 'DNF'}")
    print("figs written to", FIGS)
