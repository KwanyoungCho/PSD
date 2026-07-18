#!/usr/bin/env python3
"""Verdict Exp1 forensics — B=4 champion profile vs the structural ROW MODEL.

Reads duet_profile_*.json from prof_b4/ (B=4) and the B=1 champion profile
(champion_profile/e9k24_jit_profile/) as the reference, and prints:
  1. per-status per-label ms/step tables (B=4 vs B=1 hit_k1 reference)
  2. phase1/phase2 per-forward time distributions (tile-cliff check)
  3. verify width (vk_max) distribution via target graph_pre bimodality
     + draft phase1 per-forward clustering
  4. step-wall accounting (top-level label sum vs wall) per proc/status

Usage: python analyze_prof.py [--warmup 100]
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
B1_DIR = os.path.join(BASE, "..", "..", "champion_profile", "e9k24_jit_profile")
B4_DIR = os.path.join(BASE, "prof_b4")


def load(dirpath):
    """{proc: {step_id: [row,...]}} top-level rows; children kept flagged."""
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(dirpath, "duet_profile_*.json"))):
        with open(p) as f:
            payload = json.load(f)
        for r in payload:
            if r.get("label") == "_anchor" or r.get("step_id") is None:
                continue
            proc = r.get("proc") or (
                "target" if "target" in os.path.basename(p) else "draft")
            if proc.startswith("target"):
                proc = "target"
            out[proc][r["step_id"]].append(r)
    return out


def trim(steps, warmup):
    sids = sorted(steps)
    return [s for s in sids if s >= sids[0] + warmup]


def step_status(rows):
    stats = [r.get("status") for r in rows if r.get("status")]
    return max(set(stats), key=stats.count) if stats else "unknown"


def label_table(steps, sids, title, ref=None):
    """Per-status per-label mean ms/step (+count). ref: {label: ms} to show."""
    agg = defaultdict(lambda: defaultdict(list))
    cnt = defaultdict(lambda: defaultdict(list))
    wall = defaultdict(list)
    stat_n = defaultdict(int)
    for s in sids:
        rs = steps[s]
        st = step_status(rs)
        stat_n[st] += 1
        by = defaultdict(float)
        n = defaultdict(int)
        for r in rs:
            key = ("child:" if r.get("parent_label") else "") + r["label"]
            by[key] += r["cuda_ms"]
            n[key] += 1
        for lb, ms in by.items():
            agg[st][lb].append(ms)
            cnt[st][lb].append(n[lb])
        w0 = min(r["wall_start_ns"] for r in rs)
        w1 = max(r["wall_end_ns"] for r in rs)
        wall[st].append((w1 - w0) / 1e6)
    tot = sum(stat_n.values())
    print(f"\n## {title} — {tot} steps, status shares: "
          + ", ".join(f"{k}={v} ({v/tot:.1%})" for k, v in sorted(stat_n.items())))
    for st in sorted(agg, key=lambda k: -stat_n[k]):
        print(f"\n### status={st} (n={stat_n[st]}, wall {statistics.mean(wall[st]):.2f} ms)")
        print("| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |")
        print("|---|---|---|---|---|")
        toplevel_sum = 0.0
        for lb in sorted(agg[st], key=lambda l: -statistics.mean(agg[st][l])):
            ms = statistics.mean(agg[st][lb])
            c = statistics.mean(cnt[st][lb])
            if not lb.startswith("child:"):
                toplevel_sum += ms
            rv = f"{ref[lb]:.2f}" if ref and lb in ref else "-"
            print(f"| {lb} | {ms:.3f} | {c:.2f} | {ms/max(c,1e-9):.3f} | {rv} |")
        print(f"| _top-level sum_ | {toplevel_sum:.2f} | | | wall {statistics.mean(wall[st]):.2f} |")
    return stat_n


def ref_table(steps, sids, status="hit_k1"):
    agg = defaultdict(list)
    for s in sids:
        rs = steps[s]
        if step_status(rs) != status:
            continue
        by = defaultdict(float)
        for r in rs:
            key = ("child:" if r.get("parent_label") else "") + r["label"]
            by[key] += r["cuda_ms"]
        for lb, ms in by.items():
            agg[lb].append(ms)
    return {lb: statistics.mean(v) for lb, v in agg.items()}


def per_forward(steps, sids, label):
    xs = []
    per_step_n = []
    for s in sids:
        rs = [r for r in steps[s] if r["label"] == label]
        if rs:
            xs.extend(r["cuda_ms"] for r in rs)
            per_step_n.append(len(rs))
    return xs, per_step_n


def dist(xs, name):
    if not xs:
        print(f"{name}: EMPTY")
        return
    xs_s = sorted(xs)
    q = lambda p: xs_s[int(p * (len(xs_s) - 1))]
    print(f"{name}: n={len(xs)} mean={statistics.mean(xs):.3f} "
          f"p10={q(.1):.3f} p25={q(.25):.3f} p50={q(.5):.3f} "
          f"p75={q(.75):.3f} p90={q(.9):.3f} p99={q(.99):.3f}")


def width_split(steps, sids, label, cut):
    """Classify steps by per-step mean cuda_ms of `label` vs threshold."""
    lo, hi = [], []
    for s in sids:
        rs = [r["cuda_ms"] for r in steps[s] if r["label"] == label]
        if not rs:
            continue
        (hi if statistics.mean(rs) > cut else lo).append(s)
    n = len(lo) + len(hi)
    print(f"width split on {label} (cut {cut:.2f} ms): "
          f"short(K2)={len(lo)} ({len(lo)/n:.1%})  long(K1)={len(hi)} ({len(hi)/n:.1%})")
    return set(lo), set(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    b1 = load(B1_DIR)
    b4 = load(B4_DIR)

    # B=1 references (hit_k1 steps)
    refs = {}
    for proc in b1:
        sids = trim(b1[proc], args.warmup)
        refs[proc] = ref_table(b1[proc], sids)

    for proc in ["draft", "target"]:
        if proc not in b4:
            continue
        sids = trim(b4[proc], args.warmup)
        label_table(b4[proc], sids, f"B=4 {proc}", refs.get(proc))

    # per-forward distributions, draft
    print("\n## per-forward distributions (draft)")
    dsids = trim(b4["draft"], args.warmup)
    b1sids = trim(b1["draft"], args.warmup)
    for lb in ["phase1_replay", "phase2_replay", "draft_glue_replay"]:
        xs4, n4 = per_forward(b4["draft"], dsids, lb)
        xs1, n1 = per_forward(b1["draft"], b1sids, lb)
        dist(xs1, f"B=1 {lb}/fwd")
        dist(xs4, f"B=4 {lb}/fwd")
        if n4:
            print(f"  B=4 forwards/step: mean {statistics.mean(n4):.2f}")

    # verify width distribution via target graph_pre bimodality
    print("\n## verify width (vk_max) distribution")
    tsids = trim(b4["target"], args.warmup)
    xs, _ = per_forward(b4["target"], tsids, "graph_pre")
    dist(xs, "B=4 graph_pre")
    xs1, _ = per_forward(b1["target"], trim(b1["target"], args.warmup), "graph_pre")
    dist(xs1, "B=1 graph_pre")
    # cut: midpoint between B=1 k2 (~25.3) and k1 (~31.5) modes scaled — pick
    # from the B=4 histogram: print a coarse histogram to choose visually
    xs_s = sorted(xs)
    lo, hi = xs_s[0], xs_s[-1]
    nb = 20
    binw = (hi - lo) / nb
    hist = [0] * nb
    for x in xs:
        hist[min(int((x - lo) / binw), nb - 1)] += 1
    print("graph_pre histogram (bin_start:count):")
    for i, c in enumerate(hist):
        print(f"  {lo + i*binw:7.2f}: {'#' * max(1, c * 60 // max(hist))} {c}")


if __name__ == "__main__":
    main()
