"""Side-by-side per-phase breakdown for split vs hybrid MESA.

Reads mesa_profile_{draft,target_rank0}_*.json from two run dirs and emits:
  - per-phase mean/median/p95/n table for each run
  - delta column: hybrid - split (negative = hybrid faster)
  - per-spec-step roll-up that uses the EMPIRICAL replay multiplier (replay
    label count / merge_cache count), not a hardcoded K.

Usage:
    python compare_breakdown.py <results_dir>
where results_dir contains subdirs split_k5_dfo2_exit40/ and
hybrid_k5_K1_3_K2_2_exit40/ (the standard layout from run.sh).
"""
import glob
import json
import os
import sys
from collections import defaultdict

import pandas as pd

WARMUP = 5  # drop first N events per (proc, label) — CG capture / compile spikes


def _load_run(run_dir):
    rows = []
    for tag in ("draft", "target_rank0"):
        paths = sorted(glob.glob(f"{run_dir}/mesa_profile_{tag}_*.json"))
        if not paths:
            continue
        for r in json.load(open(paths[-1])):
            r["proc"] = tag.replace("_rank0", "")
            rows.append(r)
    if not rows:
        raise SystemExit(f"no mesa_profile_*.json in {run_dir}")
    df = pd.DataFrame(rows)
    df = df.sort_values(["proc", "label", "idx"])
    df["_rank_in_grp"] = df.groupby(["proc", "label"]).cumcount()
    df = df[df["_rank_in_grp"] >= WARMUP].drop(columns="_rank_in_grp").reset_index(drop=True)
    return df


def _phase_summary(df, run_name):
    g = df.groupby(["proc", "label"])["ms"]
    summ = g.agg(
        mean="mean",
        median="median",
        p95=lambda s: s.quantile(0.95),
        n="count",
    ).round(3)
    summ.columns = pd.MultiIndex.from_product([[run_name], summ.columns])
    return summ


def _per_step_rollup(df):
    """Empirical replay multiplier from event counts per (proc).

    Multiplier(label) = count(label) / count(merge_cache on draft).
    For draft proc this collapses K1 phase1 replays + K2 phase2 replays
    + 1 merge into 1 spec step's per-phase contribution.
    For target proc the equivalent base is target_postprocess (1/step).
    """
    out = []
    for proc, sub in df.groupby("proc"):
        base_label = "merge_cache" if proc == "draft" else "target_postprocess"
        n_steps = (sub["label"] == base_label).sum()
        if n_steps == 0:
            base_label = "phase2_build" if proc == "draft" else "verify_sample_accept"
            n_steps = (sub["label"] == base_label).sum()
        if n_steps == 0:
            continue
        for label, grp in sub.groupby("label"):
            mean_ms = grp["ms"].mean()
            mult = grp.shape[0] / n_steps
            out.append({
                "proc": proc, "label": label,
                "mean_ms": round(mean_ms, 3),
                "events_per_step": round(mult, 2),
                "ms_per_step": round(mean_ms * mult, 3),
                "n_events": grp.shape[0],
            })
    return pd.DataFrame(out)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: compare_breakdown.py <results_dir>")
    root = sys.argv[1]
    runs = {}
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        if not glob.glob(f"{d}/mesa_profile_*.json"):
            continue
        runs[sub] = _load_run(d)

    if len(runs) < 2:
        raise SystemExit(f"need at least 2 runs in {root}; found {list(runs)}")

    # Per-phase mean table per run
    per_phase = pd.concat(
        [_phase_summary(df, name) for name, df in runs.items()], axis=1,
    ).fillna("-")
    print("=" * 88)
    print(f"Per-phase summary (post-warmup, {WARMUP} events dropped per label):")
    print("=" * 88)
    print(per_phase.to_string())

    # Per-spec-step contribution per run
    print()
    print("=" * 88)
    print("Per-spec-step contribution (mean_ms × events_per_step):")
    print("=" * 88)
    per_step = []
    for name, df in runs.items():
        ps = _per_step_rollup(df)
        ps.insert(0, "run", name)
        per_step.append(ps)
    ps_all = pd.concat(per_step, ignore_index=True)
    ps_all = ps_all.sort_values(["proc", "run", "ms_per_step"], ascending=[True, True, False])
    print(ps_all.to_string(index=False))

    # Delta: same labels across runs
    print()
    print("=" * 88)
    print("Per-phase delta (hybrid run - split run, negative = hybrid faster):")
    print("=" * 88)
    if len(runs) == 2:
        names = list(runs.keys())
        a, b = sorted(names, key=lambda n: 0 if n.startswith("split") else 1)  # split first
        means = pd.DataFrame({
            a: runs[a].groupby(["proc", "label"])["ms"].mean(),
            b: runs[b].groupby(["proc", "label"])["ms"].mean(),
        }).round(3)
        means["delta_ms"] = (means[b] - means[a]).round(3)
        means["delta_pct"] = (
            (means[b] - means[a]) / means[a].replace(0, float("nan")) * 100
        ).round(1)
        print(means.to_string())

    # Save
    out_csv = os.path.join(root, "compare_per_phase.csv")
    per_phase.to_csv(out_csv)
    print(f"\n-> {out_csv}")
    out_csv2 = os.path.join(root, "compare_per_step.csv")
    ps_all.to_csv(out_csv2, index=False)
    print(f"-> {out_csv2}")


if __name__ == "__main__":
    main()
