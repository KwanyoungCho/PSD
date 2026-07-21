#!/usr/bin/env python
"""Dual-process wall-clock timeline (Gantt) from SSD_PROFILE_DUET dumps.

Draws target_rank0 and draft lanes on ONE wall-clock axis (host-monotonic
wall_ns is shared across processes on the same box) for a window of
consecutive steady-state steps, so draft/target overlap is visually
checkable instead of inferred from averages. Also prints an overlap
metric: fraction of target-busy time during which the draft is busy.

Usage: plot_timeline.py <profile_dir> <out_png> <title> [n_steps]
"""
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Okabe-Ito (CVD-safe) + grays; waiting states share the same gray on
# both lanes so "wait" reads as one semantic across the figure.
CAT = {
    "t_wait":   ("#999999", "target: spec_wait (waiting for draft)"),
    "t_layers": ("#0072B2", "target: verify layers (GEMM)"),
    "t_proxy":  ("#D55E00", "target: proxy block (exit+policy+send)"),
    "t_misc":   ("#56B4E9", "target: setup/sample/post"),
    "d_wait":   ("#BBBBBB", "draft: wait (recv/proxy_wait)"),
    "d_glue":   ("#009E73", "draft: glue decode"),
    "d_p1":     ("#CC79A7", "draft: phase-1 fwd/build"),
    "d_p2":     ("#F0E442", "draft: phase-2 fwd/build"),
    "d_cache":  ("#E69F00", "draft: cache fill/merge/wire"),
    "other":    ("#DDDDDD", "other"),
}

T_MAP = {
    "target_spec_wait": "t_wait",
    "graph_pre": "t_layers", "graph_post": "t_layers",
    "exit_logits": "t_layers", "final_logits": "t_layers",
    "proxy_compute_send": "t_proxy",
    "verify_replay": "t_layers",
    "verify_setup": "t_misc", "verify_sample_accept": "t_misc",
    "target_postprocess": "t_misc",
}
D_PREFIX = [  # (prefix, category) — first match wins
    ("draft_recv", "d_wait"), ("proxy_wait", "d_wait"),
    ("glue", "d_glue"), ("draft_glue", "d_glue"),
    ("phase1", "d_p1"), ("phase2", "d_p2"),
    ("hit_cache_respond", "d_cache"), ("merge_cache", "d_cache"),
    ("draft_send", "d_cache"),
    # non-DUET async-SD draft labels (best effort; unknown -> other)
    ("tree", "d_p1"), ("jit", "d_cache"), ("spec", "d_p1"),
]
D_WAIT = {"d_wait"}


def load(path):
    ev = json.load(open(path))
    return [e for e in ev[1:] if e.get("wall_start_ns") is not None]


def dcat(label):
    for p, c in D_PREFIX:
        if label.startswith(p):
            return c
    return "other"


def union(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def inter_len(a, b):
    i = j = tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def main():
    pdir, out_png, title = sys.argv[1], sys.argv[2], sys.argv[3]
    n_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    tj = sorted(glob.glob(f"{pdir}/duet_profile_target_rank0_*.json"))[-1]
    dj = sorted(glob.glob(f"{pdir}/duet_profile_draft_*.json"))[-1]
    tev, dev = load(tj), load(dj)

    steps = sorted({e["step_id"] for e in tev if e.get("step_id") is not None})
    mid = steps[len(steps) // 2]
    win_steps = [s for s in steps if mid <= s < mid + n_steps]
    wt = [e for e in tev if e.get("step_id") in win_steps
          and e.get("parent_label") is None]
    t0 = min(e["wall_start_ns"] for e in wt)
    t1 = max(e["wall_end_ns"] for e in wt)
    wd = [e for e in dev if e.get("parent_label") is None
          and e["wall_end_ns"] > t0 and e["wall_start_ns"] < t1]

    ms = lambda ns: (ns - t0) / 1e6
    fig, ax = plt.subplots(figsize=(15, 4.2), dpi=150)
    used = set()

    for e in wt:
        c = T_MAP.get(e["label"], "other")
        used.add(c)
        ax.broken_barh([(ms(e["wall_start_ns"]),
                         ms(e["wall_end_ns"]) - ms(e["wall_start_ns"]))],
                       (1.12, 0.76), facecolors=CAT[c][0],
                       edgecolors="#666666", linewidth=0.3)
    for e in wd:
        c = dcat(e["label"])
        used.add(c)
        ax.broken_barh([(ms(e["wall_start_ns"]),
                         ms(e["wall_end_ns"]) - ms(e["wall_start_ns"]))],
                       (0.12, 0.76), facecolors=CAT[c][0],
                       edgecolors="#666666", linewidth=0.3)

    # step boundaries = start of each step's first top-level target event
    for s in win_steps:
        xs = min(ms(e["wall_start_ns"]) for e in wt if e["step_id"] == s)
        ax.axvline(xs, color="#444444", lw=0.8, ls="--", zorder=0)
        ax.text(xs, 2.02, f"step {s}", fontsize=8, color="#444444")

    # overlap metric over the window
    tbusy = union([[e["wall_start_ns"], e["wall_end_ns"]] for e in wt
                   if T_MAP.get(e["label"], "other") != "t_wait"])
    dbusy = union([[e["wall_start_ns"], e["wall_end_ns"]] for e in wd
                   if dcat(e["label"]) not in D_WAIT])
    tb = sum(e - s for s, e in tbusy)
    db = sum(e - s for s, e in dbusy)
    ov = inter_len(tbusy, dbusy)
    hid = ov / db if db else 0.0
    ax.text(0.995, 0.02,
            f"window {n_steps} steps: target-busy {tb/1e6:.0f} ms, draft-work {db/1e6:.0f} ms, "
            f"draft-work hidden under target-busy: {hid*100:.0f}%",
            transform=ax.transAxes, ha="right", fontsize=9, color="#222222")

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["draft (GPU4)", "target rank0"])
    ax.set_xlabel("wall-clock ms (window-relative)")
    ax.set_ylim(0, 2.2)
    ax.set_title(title)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=CAT[c][0], edgecolor="#666666", label=CAT[c][1])
               for c in CAT if c in used]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"saved {out_png}; target-busy {tb/1e6:.0f}ms draft-work {db/1e6:.0f}ms hidden {hid*100:.1f}%")


if __name__ == "__main__":
    main()
