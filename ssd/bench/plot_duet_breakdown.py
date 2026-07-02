"""Plot DUET per-phase breakdown. Reads duet_profile_{draft,target_rank0}.json from OUTDIR.

OUTDIR resolution: argv[1] > env SSD_PROFILE_DIR > /tmp.
Writes PNG/CSV into the same OUTDIR.
"""
import json, sys, os
import pandas as pd
import matplotlib.pyplot as plt

WARMUP = 5   # drop first N events per label (capture/compile spikes)
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SSD_PROFILE_DIR", "/tmp")

import glob
dfs = []
for tag in ("draft", "target_rank0"):
    paths = sorted(glob.glob(f"{OUTDIR}/duet_profile_{tag}_*.json")) or \
            sorted(glob.glob(f"{OUTDIR}/duet_profile_{tag}.json")) or sorted(glob.glob(f"{OUTDIR}/mesa_profile_{tag}_*.json")) or sorted(glob.glob(f"{OUTDIR}/mesa_profile_{tag}.json"))
    if not paths:
        print(f"[warn] missing duet_profile_{tag}_*.json in {OUTDIR}", file=sys.stderr); continue
    path = paths[-1]
    rows = json.load(open(path))
    for r in rows:
        r["proc"] = tag.replace("_rank0", "")
    dfs.append(pd.DataFrame(rows))

if not dfs:
    print("No profile files found."); sys.exit(1)

df = pd.concat(dfs, ignore_index=True)

# Drop warmup per (proc, label)
df = df.sort_values(["proc", "label", "idx"])
df["_rank_in_grp"] = df.groupby(["proc", "label"]).cumcount()
df = df[df["_rank_in_grp"] >= WARMUP].drop(columns="_rank_in_grp").reset_index(drop=True)

summary = df.groupby(["proc", "label"])["ms"].agg(
    mean="mean", median="median", p95=lambda s: s.quantile(0.95), n="count"
).round(3)
print("=== DUET phase breakdown (post-warmup) ===")
print(summary)

# Grouped bar: mean ms per phase by proc
pivot = df.groupby(["proc", "label"])["ms"].mean().unstack("proc").fillna(0)
ax = pivot.plot.bar(figsize=(11, 5), width=0.8)
_cfg_name = os.path.basename(os.path.normpath(OUTDIR))
_is_duet = df["label"].str.startswith("phase").any()
_method = "DUET" if _is_duet else "SSD (baseline)"
ax.set_ylabel("ms / invocation"); ax.set_title(f"{_method} — {_cfg_name} — per-phase latency (mean)")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
out = f"{OUTDIR}/duet_breakdown.png"
plt.savefig(out, dpi=120)
print(f"\n-> saved {out}")

# Per-event time series (draft): see Phase1 vs Phase2 vs wait vs merge alignment
if (df["proc"] == "draft").any():
    d = df[df["proc"] == "draft"].sort_values("idx")
    fig, ax = plt.subplots(figsize=(11, 4))
    for lb, sub in d.groupby("label"):
        ax.plot(sub["idx"], sub["ms"], ".", markersize=3, label=lb, alpha=0.6)
    ax.set_xlabel("event idx (monotonic, draft proc)"); ax.set_ylabel("ms")
    ax.set_title("Draft-process per-phase latency over time")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    out2 = f"{OUTDIR}/duet_breakdown_over_time.png"
    plt.savefig(out2, dpi=120)
    print(f"-> saved {out2}")

# Per-spec-step aggregation: since phase1/phase2_replay fire K=4 times per step but
# glue/merge/proxy_wait fire once, aggregate by idx buckets to visualize one-step cost.
summary.to_csv(f"{OUTDIR}/duet_breakdown_summary.csv")
print(f"-> saved {OUTDIR}/duet_breakdown_summary.csv")

# Per-step contribution summary (multiply replays by K=4, others by 1)
per_step = []
for (proc, label), grp in df.groupby(["proc", "label"]):
    mult = 4 if label in ("phase1_replay", "phase2_replay", "tree_replay") else 1
    per_step.append({"proc": proc, "label": label, "mean_ms": grp["ms"].mean(),
                     "mult_per_step": mult,
                     "ms_per_step": grp["ms"].mean() * mult})
ps = pd.DataFrame(per_step).sort_values(["proc", "ms_per_step"], ascending=[True, False])
ps.to_csv(f"{OUTDIR}/duet_per_step_contribution.csv", index=False)
print(f"-> saved {OUTDIR}/duet_per_step_contribution.csv")
print("\n=== Per-spec-step contribution (mean × K-multiplier) ===")
print(ps.to_string(index=False))
