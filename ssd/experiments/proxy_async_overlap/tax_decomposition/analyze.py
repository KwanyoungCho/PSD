#!/usr/bin/env python3
"""Tax decomposition analyzer — per-label slope vs K1.

Reads duet_profile_*.json dumps from N cell dirs (one per K1 point),
aggregates per-step per-label cuda_ms sums (split by status), and fits a
per-K1 slope for every label. 3 points let us flag step jumps (CG bucket
crossings) via the middle-point residual.

Usage:
  python analyze.py --cells k1_7:7 k1_8:8 k1_9:9 [--warmup 100]
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load_cell(cell_dir):
    """Return {proc: [row, ...]} for all duet_profile jsons in cell_dir."""
    rows_by_proc = defaultdict(list)
    paths = sorted(glob.glob(os.path.join(cell_dir, "duet_profile_*.json")))
    if not paths:
        raise FileNotFoundError(f"no duet_profile_*.json in {cell_dir}")
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        for r in payload:
            if r.get("label") == "_anchor":
                continue
            proc = r.get("proc") or ("target" if "target" in os.path.basename(p) else "draft")
            rows_by_proc[proc].append(r)
    return rows_by_proc


def per_step_label_sums(rows, warmup):
    """{(status, label): [per-step summed cuda_ms ...]}, child labels kept
    separate under key (status, 'child:'+label). Also returns
    {status: [step_wall_ms ...]} and step counts."""
    steps = defaultdict(list)
    for r in rows:
        sid = r.get("step_id")
        if sid is None:
            continue
        steps[sid].append(r)
    if not steps:
        return {}, {}, 0
    sids = sorted(steps)
    sids = [s for s in sids if s >= sids[0] + warmup]
    label_sums = defaultdict(list)
    step_wall = defaultdict(list)
    for sid in sids:
        rs = steps[sid]
        # status: majority label on the step's rows (late-bound; identical
        # within a step in practice)
        stats = [r.get("status") for r in rs if r.get("status")]
        status = max(set(stats), key=stats.count) if stats else "unknown"
        by_label = defaultdict(float)
        for r in rs:
            key = r["label"] if not r.get("parent_label") else "child:" + r["label"]
            by_label[key] += r["cuda_ms"]
        for lb, ms in by_label.items():
            label_sums[(status, lb)].append(ms)
        w0 = min(r["wall_start_ns"] for r in rs)
        w1 = max(r["wall_end_ns"] for r in rs)
        step_wall[status].append((w1 - w0) / 1e6)
    return label_sums, step_wall, len(sids)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fit_slope(k1s, ys):
    """Least-squares slope + middle-point residual (linear vs step check)."""
    n = len(k1s)
    mx, my = sum(k1s) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(k1s, ys))
    var = sum((x - mx) ** 2 for x in k1s)
    slope = cov / var if var else float("nan")
    resid = None
    if n == 3:
        pred_mid = ys[0] + slope * (k1s[1] - k1s[0])
        resid = ys[1] - pred_mid
    return slope, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True,
                    help="dir:k1 pairs, e.g. k1_7:7 k1_8:8 k1_9:9")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--base", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    cells = []
    for spec in args.cells:
        d, k1 = spec.rsplit(":", 1)
        cells.append((os.path.join(args.base, d), int(k1)))

    # {proc: {(status,label): {k1: mean_ms}}}, {proc: {status: {k1: (mean_wall, n)}}}
    agg = defaultdict(lambda: defaultdict(dict))
    walls = defaultdict(lambda: defaultdict(dict))
    status_share = defaultdict(dict)
    for cell_dir, k1 in cells:
        rows_by_proc = load_cell(cell_dir)
        for proc, rows in rows_by_proc.items():
            label_sums, step_wall, nsteps = per_step_label_sums(rows, args.warmup)
            for key, xs in label_sums.items():
                agg[proc][key][k1] = (mean(xs), len(xs))
            tot = sum(len(v) for v in step_wall.values())
            for st, xs in step_wall.items():
                walls[proc][st][k1] = (mean(xs), len(xs))
                status_share[(proc, st)][k1] = len(xs) / tot if tot else float("nan")

    k1s = [k1 for _, k1 in cells]

    for proc in sorted(agg):
        print(f"\n## proc = {proc}")
        print(f"\n### status shares")
        hdr = " | ".join(f"K1={k}" for k in k1s)
        print(f"| status | {hdr} |")
        print("|---|" + "---|" * len(k1s))
        for (p, st), d in sorted(status_share.items()):
            if p != proc:
                continue
            vals = " | ".join(f"{d.get(k, float('nan')):.3f}" for k in k1s)
            print(f"| {st} | {vals} |")

        print(f"\n### step wall (ms, per status)")
        print(f"| status | {hdr} | slope | mid-resid |")
        print("|---|" + "---|" * (len(k1s) + 2))
        for st, d in sorted(walls[proc].items()):
            if not all(k in d for k in k1s):
                continue
            ys = [d[k][0] for k in k1s]
            slope, resid = fit_slope(k1s, ys)
            vals = " | ".join(f"{y:.2f}" for y in ys)
            rs = f"{resid:+.2f}" if resid is not None else "-"
            print(f"| {st} | {vals} | {slope:+.2f} | {rs} |")

        # per-label tables, split by status; sorted by |slope| on the
        # dominant status (hit_k1) to surface the tax carriers
        for st in ["hit_k1", "hit_k2", "miss", "unknown"]:
            keys = [(s, lb) for (s, lb) in agg[proc] if s == st]
            if not keys:
                continue
            rows_out = []
            for key in keys:
                d = agg[proc][key]
                if not all(k in d for k in k1s):
                    continue
                ys = [d[k][0] for k in k1s]
                slope, resid = fit_slope(k1s, ys)
                rows_out.append((abs(slope), key[1], ys, slope, resid))
            rows_out.sort(reverse=True)
            print(f"\n### per-label mean per-step cuda_ms — status={st}")
            print(f"| label | {hdr} | slope ms/pos | mid-resid |")
            print("|---|" + "---|" * (len(k1s) + 2))
            for _, lb, ys, slope, resid in rows_out:
                vals = " | ".join(f"{y:.3f}" for y in ys)
                rs = f"{resid:+.3f}" if resid is not None else "-"
                print(f"| {lb} | {vals} | {slope:+.3f} | {rs} |")


if __name__ == "__main__":
    main()
