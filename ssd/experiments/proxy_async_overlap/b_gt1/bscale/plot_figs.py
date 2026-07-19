#!/usr/bin/env python3
"""bscale figures — B-scaling story (fig1..fig5) into figs/.

Parses run.log files directly (bscale scan/confirm + pb_sweep scan/confirm),
so it needs no manual data entry. Run with the ssd env python and
MPLCONFIGDIR=/tmp/matplotlib.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BASE = Path(__file__).resolve().parent          # .../b_gt1/bscale
PB = BASE.parent / "pb_sweep"                    # .../b_gt1/pb_sweep
FIGS = BASE / "figs"
FIGS.mkdir(exist_ok=True)

# ---- palette (dataviz reference instance, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DUET = "#2a78d6"     # categorical slot 1 (blue)
CBASE = "#008300"    # categorical slot 2 (green)
# ordinal blue ramp steps (250/400/550/700) for ordered dfo classes
ORDINAL = {2: "#86b6ef", 3: "#3987e5", 4: "#1c5cab", 5: "#0d366b"}

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
    """[tps...] for confirm reps r1..r3 under root."""
    out = []
    for r in (1, 2, 3):
        v = tps_of(root / f"{dirpat}_r{r}" / "run.log")
        if v is not None:
            out.append(v)
    return out


def style_axes(ax):
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def logb_axis(ax, bs=(1, 2, 4, 8)):
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(bs))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("batch size B (log2)")


# ------------------------------------------------------------------
# data assembly
# ------------------------------------------------------------------
# per-B best DUET shape + C. Confirm reps where available, else scan.
B8_WINNER = None  # filled from scan below
b8_scan = {}
for cell in ("b8_k2x2_d4p1", "b8_k2x2_d5p1", "b8_k3x3_d4p1",
             "b8_k3x3_d4p2", "b8_k4x4_d3p1"):
    v = tps_of(BASE / cell / "run.log")
    if v is not None:
        b8_scan[cell] = v
if b8_scan:
    B8_WINNER = max(b8_scan, key=b8_scan.get)

DATA = {}  # B -> dict(duet=[reps], c=[reps], duet_label=str, source=str)
# B=1: bscale same-regime scan anchors (single run, ns=12)
DATA[1] = {
    "duet": [v for v in [tps_of(BASE / "b1_e9k24_jit" / "run.log")] if v],
    "c": [v for v in [tps_of(BASE / "b1_c" / "run.log")] if v],
    "duet_label": "E9K24_jit (K1=9 K2=4)",
    "source": "scan (1 run, ns=12)",
}
# B=2, B=4: pb_sweep confirm (3-rep interleaved, ns=20)
DATA[2] = {"duet": reps("b2_duet", PB / "confirm"),
           "c": reps("b2_c", PB / "confirm"),
           "duet_label": "k6x5_d3p1 (K1=6 K2=5)",
           "source": "confirm (3-rep, ns=20)"}
DATA[4] = {"duet": reps("b4_duet", PB / "confirm"),
           "c": reps("b4_c", PB / "confirm"),
           "duet_label": "k3x3_d4p1 (K1=3 K2=3)",
           "source": "confirm (3-rep, ns=20)"}
# B=8: bscale confirm (3-rep interleaved, ns=20); scan fallback
DATA[8] = {"duet": reps("b8_duet", BASE / "confirm"),
           "c": reps("b8_c", BASE / "confirm"),
           "duet_label": B8_WINNER.replace("b8_", "") if B8_WINNER else "?",
           "source": "confirm (3-rep, ns=20)"}
if not DATA[8]["duet"] and B8_WINNER:
    DATA[8]["duet"] = [b8_scan[B8_WINNER]]
    DATA[8]["c"] = [v for v in [tps_of(BASE / "b8_c" / "run.log")] if v]
    DATA[8]["source"] = "scan (1 run, ns=12)"

BS = [b for b in (1, 2, 4, 8) if DATA[b]["duet"] and DATA[b]["c"]]


def mean(xs):
    return sum(xs) / len(xs)


def series(which):
    m = [mean(DATA[b][which]) for b in BS]
    lo = [mean(DATA[b][which]) - min(DATA[b][which]) for b in BS]
    hi = [max(DATA[b][which]) - mean(DATA[b][which]) for b in BS]
    return m, [lo, hi]


# ------------------------------------------------------------------
# fig1 — aggregate decode TPS vs B
# ------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    dm, derr = series("duet")
    cm, cerr = series("c")
    ax.errorbar(BS, dm, yerr=derr, color=DUET, lw=2, marker="o", ms=7,
                capsize=3, label="DUET (per-B best shape)", zorder=3)
    ax.errorbar(BS, cm, yerr=cerr, color=CBASE, lw=2, marker="s", ms=6.5,
                capsize=3, label="async-SD best C (k7 f6)", zorder=3)
    for b, v in zip(BS, dm):
        ax.annotate(f"{v:.0f}", (b, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=8.5)
    logb_axis(ax, BS)
    style_axes(ax)
    ax.set_ylabel("aggregate decode TPS (tok/s)")
    ax.set_title("Decode throughput vs batch size — DUET per-B best vs SD-best C\n"
                 "(error bars = min/max over 3-rep interleaved confirm; "
                 "B=1 single-run same-regime anchors)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_tps_vs_B.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig2 — DUET advantage % vs B
# ------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    adv, band_clear = [], []
    for b in BS:
        d, c = DATA[b]["duet"], DATA[b]["c"]
        adv.append(100.0 * (mean(d) / mean(c) - 1.0))
        band_clear.append(len(d) >= 3 and min(d) > max(c))
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(BS, adv, color=DUET, lw=2, marker="o", ms=7, zorder=3)
    for b, a, bc in zip(BS, adv, band_clear):
        lbl = f"{a:+.1f}%" + ("\nband-clear" if bc else "")
        ax.annotate(lbl, (b, a), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=INK if bc else INK2, fontsize=9,
                    fontweight="bold" if bc else "normal")
    logb_axis(ax, BS)
    style_axes(ax)
    ax.set_ylabel("DUET advantage over C (%)")
    ax.set_title("The amplification curve — DUET-over-C advantage vs batch size\n"
                 "(band-clear = worst DUET rep > best C rep, 3-rep interleaved)",
                 fontsize=10)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.15 * (ymax - ymin))
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_advantage_vs_B.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig3 — optimal shape vs B
# ------------------------------------------------------------------
def parse_shape(label):
    m = re.match(r"k(\d+)x(\d+)_d(\d+)p(\d+)", label)
    if not m:
        return None
    k1, k2, dfo, pfo = map(int, m.groups())
    return dict(K1=k1, K2=k2, f=dfo + pfo, rows=k1 + 1)


def fig3():
    shapes = {1: dict(K1=9, K2=4, f=3, rows=10)}  # champion E9K24_jit
    shapes[2] = parse_shape("k6x5_d3p1")
    shapes[4] = parse_shape("k3x3_d4p1")
    if B8_WINNER:
        shapes[8] = parse_shape(B8_WINNER.replace("b8_", ""))
    bs = sorted(b for b in shapes if shapes[b])
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    # categorical slots 1-4; magenta/yellow are sub-3:1 on light surface ->
    # relief rule: every series is direct-labeled at its endpoint.
    # K1 is drawn wider/larger than K2 so both stay visible where they
    # coincide (K1=K2 at B>=4).
    series_defs = [
        ("K1 (phase-1 depth)", "K1", "#2a78d6", "o", 3.4, 11),
        ("K2 (phase-2 depth)", "K2", "#008300", "s", 1.8, 5.5),
        ("f = dfo+pfo (fan-out)", "f", "#e87ba4", "^", 2, 7),
        ("verify rows/seq (K1+1)", "rows", "#eda100", "D", 2, 7),
    ]
    for name, key, color, mk, lw, ms in series_defs:
        ys = [shapes[b][key] for b in bs]
        ax.plot(bs, ys, color=color, lw=lw, marker=mk, ms=ms, label=name)
    # endpoint direct labels, merging series that coincide at the endpoint
    short = {"K1": "K1", "K2": "K2", "f": "f", "rows": "rows/seq"}
    end_groups = {}
    for _, key, *_ in series_defs:
        end_groups.setdefault(shapes[bs[-1]][key], []).append(short[key])
    for val, names in end_groups.items():
        ax.annotate(f"{' = '.join(names)} = {val}", (bs[-1], val),
                    textcoords="offset points", xytext=(9, 0), va="center",
                    color=INK2, fontsize=8.5)
    logb_axis(ax, bs)
    style_axes(ax)
    ax.set_xlim(right=bs[-1] * 1.55)
    ax.set_ylabel("count (depth / fan-out / rows)")
    ax.set_title("Optimal speculation shape vs batch size — shallower and fatter as B grows\n"
                 "(per-B sweep winners; B=1 = campaign champion E9K24_jit)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_optimal_shape_vs_B.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig4 — B=4 response surface (scan TPS vs K1)
# ------------------------------------------------------------------
def fig4():
    cells = {}
    for d in list(PB.glob("b4_k*")) + list(BASE.glob("b4_k*")):
        v = tps_of(d / "run.log")
        if v is not None:
            cells[d.name] = v
    c_tps = tps_of(PB / "b4_c" / "run.log")
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=150)
    combos = []
    for name, v in sorted(cells.items()):
        s = parse_shape(name.replace("b4_", ""))
        dfo = int(re.search(r"_d(\d+)p", name).group(1))
        pfo = int(re.search(r"p(\d+)$", name).group(1))
        mk = "o" if pfo == 1 else "^"
        if (dfo, pfo) not in combos:
            combos.append((dfo, pfo))
        ax.scatter(s["K1"], v, s=70, color=ORDINAL[dfo], marker=mk,
                   edgecolors=SURFACE, linewidths=1.2, zorder=3)
    # legend handles in (dfo, pfo) order, drawn as proxy artists
    for dfo, pfo in sorted(combos):
        ax.scatter([], [], s=70, color=ORDINAL[dfo],
                   marker="o" if pfo == 1 else "^",
                   label=f"dfo={dfo}, pfo={pfo}")
    if c_tps is not None:
        ax.axhline(c_tps, color=CBASE, lw=1.5, ls=(0, (4, 3)))
        ax.annotate(f"C anchor (k7 f6): {c_tps:.1f}", (ax.get_xlim()[1], c_tps),
                    xytext=(-4, 5), textcoords="offset points", ha="right",
                    color=CBASE, fontsize=8.5)
    # jitter-free direct labels for the extremes
    best = max(cells, key=cells.get)
    bs_ = parse_shape(best.replace("b4_", ""))
    ax.annotate(best.replace("b4_", ""), (bs_["K1"], cells[best]),
                textcoords="offset points", xytext=(0, 8), ha="center",
                color=INK, fontsize=8.5, fontweight="bold")
    style_axes(ax)
    ax.set_xticks(sorted({parse_shape(n.replace("b4_", ""))["K1"] for n in cells}))
    ax.set_xlabel("K1 (phase-1 depth = verify width - 1)")
    ax.set_ylabel("decode TPS at B=4 (tok/s, ns=12 single run)")
    ax.set_title("B=4 response surface — TPS vs K1 (color = dfo, marker = pfo)\n"
                 "pb_sweep grid + bscale K1=2 edge cells", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, title="fan-out", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_b4_response_surface.png")
    plt.close(fig)


# ------------------------------------------------------------------
# fig5 — per-seq tok/s vs B
# ------------------------------------------------------------------
def fig5():
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    dm, derr = series("duet")
    cm, cerr = series("c")
    dps = [v / b for v, b in zip(dm, BS)]
    cps = [v / b for v, b in zip(cm, BS)]
    dpe = [[e / b for e, b in zip(derr[0], BS)], [e / b for e, b in zip(derr[1], BS)]]
    cpe = [[e / b for e, b in zip(cerr[0], BS)], [e / b for e, b in zip(cerr[1], BS)]]
    ax.errorbar(BS, dps, yerr=dpe, color=DUET, lw=2, marker="o", ms=7,
                capsize=3, label="DUET (per-B best shape)", zorder=3)
    ax.errorbar(BS, cps, yerr=cpe, color=CBASE, lw=2, marker="s", ms=6.5,
                capsize=3, label="async-SD best C (k7 f6)", zorder=3)
    for b, v in zip(BS, dps):
        ax.annotate(f"{v:.1f}", (b, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", color=INK2, fontsize=8.5)
    logb_axis(ax, BS)
    style_axes(ax)
    ax.set_ylabel("per-seq decode TPS (aggregate / B, tok/s)")
    ax.set_title("Throughput/latency tradeoff — per-seq token rate vs batch size\n"
                 "(same runs as fig1; per-seq rate = aggregate TPS / B)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_per_seq_latency.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("B8 scan:", b8_scan, "winner:", B8_WINNER)
    for b in BS:
        d, c = DATA[b]["duet"], DATA[b]["c"]
        print(f"B={b}: DUET {mean(d):.2f} {d} vs C {mean(c):.2f} {c} "
              f"[{DATA[b]['source']}]")
    print("figs written to", FIGS)
