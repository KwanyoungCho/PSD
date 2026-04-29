"""Per-fanout grid: stack the K1 candidates' timelines vertically so the user
can visually compare how Phase 1 / Phase 2 fill the step at different K1.
"""
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")

FANOUTS = [
    ("dfo2_pfo3", [6, 7, 8, 9]),
    ("dfo3_pfo3", [6, 7, 8, 9]),
    ("dfo3_pfo5", [6, 7, 8, 9]),
    ("dfo4_pfo4", [5, 6, 7, 8]),
    ("dfo4_pfo6", [5, 6, 7, 8]),
]


def parse_tps(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.search(r"Total Throughput:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def parse_accept(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.search(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def main():
    for fanout, k1_list in FANOUTS:
        n = len(k1_list)
        fig, axes = plt.subplots(n, 1, figsize=(16, 3.2 * n))
        if n == 1:
            axes = [axes]
        for i, k1 in enumerate(k1_list):
            k2 = 10 - k1
            tag = f"{fanout}_K1_{k1}_K2_{k2}"
            sub = os.path.join(RES, tag)
            png = os.path.join(sub, "mesa_timeline_step30.png")
            tps = parse_tps(os.path.join(sub, "run.log"))
            accept = parse_accept(os.path.join(sub, "run.log"))
            ax = axes[i]
            if os.path.exists(png):
                img = mpimg.imread(png)
                ax.imshow(img)
                ax.set_title(
                    f"{fanout}  K1={k1}, K2={k2}   |   TPS={tps:.1f}  accept={accept:.2f}",
                    fontsize=11, loc="left",
                )
            else:
                ax.text(0.5, 0.5, f"missing: {tag}", ha="center", va="center")
            ax.axis("off")
        plt.suptitle(f"K1 sweep timelines — {fanout} (70B both-AWQ, K_total=10)", fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(RES, f"grid_{fanout}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
