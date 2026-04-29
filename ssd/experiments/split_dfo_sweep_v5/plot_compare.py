"""dfo sweep comparison: stack 4 timelines + bar chart of TPS/accept/draft_ms."""
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
DFOS = [1, 2, 3, 4]


def parse(d):
    log = os.path.join(d, "run.log")
    if not os.path.exists(log):
        return {}
    text = open(log).read()
    pat = lambda r: float(m.group(1)) if (m := re.search(r, text)) else None
    return {
        "tps": pat(r"Total Throughput:\s*([\d.]+)"),
        "accept": pat(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)"),
        "p1": pat(r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)"),
        "p2": pat(r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)"),
        "draft_ms": pat(r"Avg draft step time \(ms\):\s*([\d.]+)"),
        "verify_ms": pat(r"Avg target verify time \(ms\):\s*([\d.]+)"),
    }


def main():
    rows = []
    for dfo in DFOS:
        sub = os.path.join(RES, f"dfo{dfo}_pfo1_K1_7_K2_7")
        rows.append((dfo, parse(sub), sub))

    # ---- Stacked timelines ----
    fig, axes = plt.subplots(len(DFOS), 1, figsize=(16, 3.4 * len(DFOS)))
    for i, (dfo, m, sub) in enumerate(rows):
        png = os.path.join(sub, "mesa_timeline_step50.png")
        ax = axes[i]
        if os.path.exists(png):
            ax.imshow(mpimg.imread(png))
            ax.set_title(
                f"dfo={dfo}, pfo=1, K1=7, K2=7   |   "
                f"TPS={m['tps']:.1f}  accept={m['accept']:.2f}  "
                f"P1={m['p1']:.2f}  P2={m['p2']:.2f}  "
                f"draft={m['draft_ms']:.1f}ms  verify={m['verify_ms']:.1f}ms",
                fontsize=11, loc="left",
            )
        ax.axis("off")
    plt.suptitle(
        "Split (NEW) — dfo sweep at K1=7, K2=7, pfo=1 (70B both-AWQ)",
        fontsize=13,
    )
    plt.tight_layout()
    out_tl = os.path.join(RES, "compare_timelines.png")
    plt.savefig(out_tl, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_tl}")

    # ---- Bar charts: TPS / accept / draft_ms / verify_ms ----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    x = np.arange(len(DFOS))
    labels = [f"dfo={d}" for d in DFOS]

    metrics = [
        ("tps", "Throughput (tok/s)", "#3b82f6"),
        ("accept", "Acceptance rate", "#16a34a"),
        ("draft_ms", "Draft step (ms)", "#f97316"),
        ("verify_ms", "Target verify (ms)", "#a855f7"),
    ]
    for ax, (key, title, color) in zip(axes, metrics):
        vals = [r[1].get(key) or 0 for r in rows]
        bars = ax.bar(x, vals, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(
        "Split (NEW) dfo sweep — K1=7, K2=7, pfo=1",
        fontsize=12,
    )
    plt.tight_layout()
    out_bar = os.path.join(RES, "compare_bars.png")
    plt.savefig(out_bar, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_bar}")


if __name__ == "__main__":
    main()
