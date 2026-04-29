"""Stack timeline + breakdown for all exit_layer values for visual comparison."""
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.abspath(__file__))
LAYERS = [30, 40, 50, 60, 70, 78]


def parse(d):
    log = os.path.join(d, "run.log")
    text = open(log).read()
    return {
        "tps": float(re.search(r"Total Throughput:\s*([\d.]+)", text).group(1)),
        "accept": float(re.search(r"Avg Fraction.*Accepted:\s*([\d.]+)", text).group(1)),
        "draft_ms": float(re.search(r"Avg draft step time \(ms\):\s*([\d.]+)", text).group(1)),
        "verify_ms": float(re.search(r"Avg target verify time \(ms\):\s*([\d.]+)", text).group(1)),
    }


def parse_proxy_wait(d):
    """proxy_wait mean ms from JSON profile."""
    import json, glob
    paths = sorted(glob.glob(os.path.join(d, "mesa_profile_draft_*.json")))
    if not paths:
        return None
    rows = json.load(open(paths[-1]))
    pw = [r["ms"] for r in rows if r["label"] == "proxy_wait"]
    return sum(pw) / max(1, len(pw))


def make_timeline_grid():
    fig, axes = plt.subplots(len(LAYERS), 1, figsize=(16, 3.4 * len(LAYERS)))
    for i, L in enumerate(LAYERS):
        d = os.path.join(ROOT, f"exit_{L}")
        m = parse(d)
        pw = parse_proxy_wait(d)
        png = os.path.join(d, "mesa_timeline_step50.png")
        ax = axes[i]
        if os.path.exists(png):
            ax.imshow(mpimg.imread(png))
            wait_str = f"proxy_wait={pw:.2f}ms" if pw is not None else "proxy_wait=?"
            ax.set_title(
                f"exit_layer={L}  |  TPS={m['tps']:.1f}  accept={m['accept']:.2f}  "
                f"{wait_str}  draft={m['draft_ms']:.1f}ms  verify={m['verify_ms']:.1f}ms",
                fontsize=11, loc="left",
            )
        ax.axis("off")
    plt.suptitle(
        "Split-K1/K2 exit_layer sweep — K1=K2=8, dfo=2, pfo=1 (70B both-AWQ)",
        fontsize=13,
    )
    plt.tight_layout()
    out = os.path.join(ROOT, "compare_timelines.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved: {out}")


def make_breakdown_grid():
    """Stack breakdown bars for each exit value."""
    fig, axes = plt.subplots(len(LAYERS), 1, figsize=(16, 3.0 * len(LAYERS)))
    for i, L in enumerate(LAYERS):
        d = os.path.join(ROOT, f"exit_{L}")
        png = os.path.join(d, "mesa_breakdown.png")
        ax = axes[i]
        if os.path.exists(png):
            ax.imshow(mpimg.imread(png))
            ax.set_title(f"exit_layer={L}", fontsize=11, loc="left")
        ax.axis("off")
    plt.suptitle("Per-phase breakdown — exit_layer sweep", fontsize=13)
    plt.tight_layout()
    out = os.path.join(ROOT, "compare_breakdowns.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    make_timeline_grid()
    make_breakdown_grid()
