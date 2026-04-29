"""Per-config idle/timing summary at cache-hit steps.

Goal: compare K1, K2 candidates for each fanout combo in terms of how well
they align with target's early-exit + verify-end timing — which shows up
as idle time on draft (proxy_wait, draft_recv_cmd) and target (target_spec_wait).

Reads each subdir's mesa_per_step_contribution.csv produced by
plot_mesa_breakdown.py. Writes a markdown summary table sorted by fanout combo.
"""
import os
import re
import sys
from glob import glob

import pandas as pd


WARMUP_EVENTS = 5  # drop first N events per (proc, label) — capture spikes


def _idle_summary(run_dir):
    """Mean per-step contribution of selected labels (cache-hit step focus)."""
    csv = os.path.join(run_dir, "mesa_per_step_contribution.csv")
    if not os.path.isfile(csv):
        return None
    df = pd.read_csv(csv)
    out = {}
    for proc, label in [
        ("draft",  "proxy_wait"),
        ("draft",  "draft_recv_cmd"),
        ("draft",  "phase1_replay"),
        ("draft",  "phase2_hybrid_replay_long"),
        ("draft",  "phase2_hybrid_replay_short"),
        ("draft",  "phase2_hybrid_build"),
        ("draft",  "glue"),
        ("draft",  "draft_glue_replay"),
        ("draft",  "hit_cache_respond"),
        ("target", "target_spec_wait"),
        ("target", "graph_pre"),
        ("target", "graph_post"),
        ("target", "verify_sample_accept"),
        ("target", "proxy_compute_send"),
    ]:
        match = df[(df["proc"] == proc) & (df["label"] == label)]
        out[f"{proc}.{label}"] = float(match["ms_per_step"].iloc[0]) if len(match) else 0.0
    return out


def _parse_tag(tag):
    """Parse 'dfo2_pfo3_K5_K1_3_K2_2' → dict."""
    m = re.match(
        r"dfo(\d+)_pfo(\d+)_K(\d+)_K1_(\d+)_K2_(\d+)", tag,
    )
    if not m:
        return None
    return {
        "dfo": int(m.group(1)),
        "pfo": int(m.group(2)),
        "K": int(m.group(3)),
        "K1": int(m.group(4)),
        "K2": int(m.group(5)),
    }


def _topline(run_dir):
    log = os.path.join(run_dir, "run.log")
    if not os.path.isfile(log):
        return {}
    text = open(log).read()
    pat = lambda r: float(m.group(1)) if (m := re.search(r, text)) else None
    return {
        "TPS": pat(r"Total Throughput:\s*([\d.]+)"),
        "accept": pat(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)"),
        "tok_step": pat(r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)"),
        "P1_hit": pat(r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)"),
        "P2_hit": pat(r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)"),
        "draft_ms": pat(r"Avg draft step time \(ms\):\s*([\d.]+)"),
        "verify_ms": pat(r"Avg target verify time \(ms\):\s*([\d.]+)"),
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: extract_idle.py <results_dir>")
    root = sys.argv[1]
    rows = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        parsed = _parse_tag(sub)
        if not parsed:
            continue
        idle = _idle_summary(d)
        top = _topline(d)
        if idle is None or top.get("TPS") is None:
            continue
        rows.append({"tag": sub, **parsed, **top, **idle})

    if not rows:
        print("No completed runs found.")
        return

    df = pd.DataFrame(rows)
    df["draft_idle"] = df["draft.proxy_wait"] + df["draft.draft_recv_cmd"]
    df["target_idle"] = df["target.target_spec_wait"]
    df["fanout_combo"] = df.apply(
        lambda r: f"dfo={r['dfo']},pfo={r['pfo']}", axis=1)

    # ----- Summary table -----
    print("# Hybrid timing-alignment sweep (exit_layer=40, both AWQ)")
    print()
    print("Per-step idle = `proxy_wait + draft_recv_cmd` on draft, `target_spec_wait` on target.")
    print("Lower is better — closer to 0 means K1/K2 align with target timing for that fanout.")
    print()
    print("| fanout | K | K1 | K2 | TPS | accept | draft_ms | verify_ms | "
          "**proxy_wait** | **draft_recv_cmd** | **draft_idle** | **target_spec_wait** | "
          "phase1_replay | phase2_replay |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    df_sorted = df.sort_values(["dfo", "pfo", "K", "K1"]).reset_index(drop=True)
    for _, r in df_sorted.iterrows():
        # Phase 2 replay total (long-bucket dominant for cache-hit)
        p2 = r["draft.phase2_hybrid_replay_long"] + r["draft.phase2_hybrid_replay_short"]
        print(f"| {r['fanout_combo']} | {r['K']} | {r['K1']} | {r['K2']} | "
              f"{r['TPS']:.2f} | {r['accept']:.2f} | "
              f"{r['draft_ms']:.2f} | {r['verify_ms']:.2f} | "
              f"**{r['draft.proxy_wait']:.2f}** | "
              f"**{r['draft.draft_recv_cmd']:.2f}** | "
              f"**{r['draft_idle']:.2f}** | "
              f"**{r['target_idle']:.2f}** | "
              f"{r['draft.phase1_replay']:.2f} | "
              f"{p2:.2f} |")

    # ----- Best-by-criterion per fanout combo -----
    print()
    print("## Best per fanout combo")
    print()
    print("| fanout | best by TPS | best by min(draft_idle) | best by min(target_idle) | best balanced* |")
    print("|---|---|---|---|---|")
    for combo, sub in df_sorted.groupby("fanout_combo"):
        # combined = draft_idle + target_idle
        sub2 = sub.assign(combined_idle=sub["draft_idle"] + sub["target_idle"])
        best_tps = sub2.loc[sub2["TPS"].idxmax()]
        best_di = sub2.loc[sub2["draft_idle"].idxmin()]
        best_ti = sub2.loc[sub2["target_idle"].idxmin()]
        best_bal = sub2.loc[sub2["combined_idle"].idxmin()]
        def _fmt(r):
            return f"K1={r['K1']},K2={r['K2']} (K={r['K']})"
        print(f"| {combo} | {_fmt(best_tps)} | {_fmt(best_di)} | {_fmt(best_ti)} | {_fmt(best_bal)} |")
    print()
    print("*balanced = minimize draft_idle + target_idle. This proxies for "
          "well-tuned K1/K2 (Phase 1 finishes ~ proxy arrival, Phase 2 finishes "
          "~ verify end).")

    df_sorted.to_csv(os.path.join(root, "timing_summary.csv"), index=False)
    print()
    print(f"-> {root}/timing_summary.csv")


if __name__ == "__main__":
    main()
