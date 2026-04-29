"""Timing-model-based K1/K2 predictor.

Goal: from a single baseline measurement per (dfo, pfo) fanout combo, predict
the (K1, K2) pair that minimizes idle (proxy_wait + draft_recv_cmd +
target_spec_wait) — without brute-force sweeping every (K1, K2) variant.

Model:
  target side: T_target_total = graph_pre + graph_post + verify_sample_accept
                                + small constants
  draft side cache-hit step:
    T_draft_total = T_glue_path
                  + K1 * T_p1
                  + max(0, T_target_proxy_arrival - K1 * T_p1 - T_glue_path)
                  + T_phase2_build
                  + K2 * T_p2

  T_target_proxy_arrival = graph_pre + proxy_compute_send (when proxy hits draft)
  T_glue_path           = hit_cache_respond + glue + draft_glue_replay
                          + phase1_build + phase2_build (one-shot per step)

  Idle:
    proxy_wait        = max(0, T_target_proxy_arrival - K1 * T_p1 - T_glue_path)
    target_spec_wait  = max(0, T_draft_total - T_target_total)
    draft_recv_cmd    = max(0, T_target_total - T_draft_total)

Algorithm (per fanout combo):
  1. From baseline measurement, fit T_p1, T_p2, proxy_arrival, target_total.
  2. K1_opt = (proxy_arrival - T_glue_path) / T_p1  → 2 candidates (floor, ceil).
  3. K2_opt = (target_total - T_glue_path - K1*T_p1 - phase2_build) / T_p2
             → 2 candidates per K1.
  4. Keep top-2 (K1, K2) by predicted total idle.
"""
import os
import re
import sys
import math
from glob import glob

import pandas as pd


def _topline(run_dir):
    log = os.path.join(run_dir, "run.log")
    if not os.path.isfile(log):
        return None
    text = open(log).read()
    pat = lambda r: float(m.group(1)) if (m := re.search(r, text)) else None
    return {
        "TPS": pat(r"Total Throughput:\s*([\d.]+)"),
        "draft_ms": pat(r"Avg draft step time \(ms\):\s*([\d.]+)"),
        "verify_ms": pat(r"Avg target verify time \(ms\):\s*([\d.]+)"),
    }


def _per_phase(run_dir):
    """Read raw mesa_profile JSONs and compute per-event mean + per-step
    counts. Avoids the plot_mesa_breakdown CSV (which hard-codes K=4 for
    replay multipliers — wrong for hybrid).
    """
    import json
    out = {}
    rows_per_proc = {}
    for tag in ("draft", "target_rank0"):
        paths = sorted(glob(os.path.join(run_dir, f"mesa_profile_{tag}_*.json")))
        if not paths:
            continue
        proc = tag.replace("_rank0", "")
        rows = json.load(open(paths[-1]))
        # drop warmup events per label
        df = pd.DataFrame(rows).sort_values(["label", "idx"])
        df["_rank"] = df.groupby("label").cumcount()
        df = df[df["_rank"] >= 5].drop(columns="_rank").reset_index(drop=True)
        rows_per_proc[proc] = df
    if not rows_per_proc:
        return None

    # Use draft's merge_cache count as the "spec step" denominator on draft;
    # target_postprocess on target. Both fire 1× per spec step.
    n_steps_draft = 0
    if "draft" in rows_per_proc:
        n_steps_draft = max(
            1,
            int((rows_per_proc["draft"]["label"] == "merge_cache").sum()),
        )
    n_steps_target = 0
    if "target" in rows_per_proc:
        n_steps_target = max(
            1,
            int((rows_per_proc["target"]["label"] == "target_postprocess").sum()),
        )

    for proc, df in rows_per_proc.items():
        n_steps = n_steps_draft if proc == "draft" else n_steps_target
        for label, sub in df.groupby("label"):
            mean_ms = float(sub["ms"].mean())
            n_events = len(sub)
            events_per_step = n_events / n_steps
            out[(proc, label)] = {
                "mean_ms": mean_ms,
                "events_per_step": events_per_step,
                "ms_per_step": mean_ms * events_per_step,
            }
    return out


def extract_baseline(run_dir):
    """Pull per-forward T_p1, T_p2, idle observations from one measured run."""
    pp = _per_phase(run_dir)
    if pp is None:
        return None
    parsed = re.match(
        r"dfo(\d+)_pfo(\d+)_K(\d+)_K1_(\d+)_K2_(\d+)", os.path.basename(run_dir),
    )
    if not parsed:
        return None
    dfo, pfo, K, K1, K2 = (int(parsed.group(i)) for i in range(1, 6))
    T_p1 = pp.get(("draft", "phase1_replay"), {}).get("mean_ms", 0)
    # Combine long+short hybrid (long fires majority of cache-hit steps)
    T_p2_long = pp.get(("draft", "phase2_hybrid_replay_long"), {}).get("mean_ms", 0)
    T_p2_short = pp.get(("draft", "phase2_hybrid_replay_short"), {}).get("mean_ms", 0)
    n_long = pp.get(("draft", "phase2_hybrid_replay_long"), {}).get("events_per_step", 0)
    n_short = pp.get(("draft", "phase2_hybrid_replay_short"), {}).get("events_per_step", 0)
    if (n_long + n_short) > 0:
        T_p2 = (T_p2_long * n_long + T_p2_short * n_short) / (n_long + n_short)
    else:
        T_p2 = T_p2_long
    T_glue_path = (
        pp.get(("draft", "hit_cache_respond"), {}).get("ms_per_step", 0)
        + pp.get(("draft", "glue"), {}).get("ms_per_step", 0)
        + pp.get(("draft", "draft_glue_replay"), {}).get("ms_per_step", 0)
        + pp.get(("draft", "phase1_build"), {}).get("ms_per_step", 0)
        + pp.get(("draft", "phase2_build"), {}).get("ms_per_step", 0)
    )
    T_phase2_build = pp.get(("draft", "phase2_hybrid_build"), {}).get("ms_per_step", 0)
    proxy_wait = pp.get(("draft", "proxy_wait"), {}).get("ms_per_step", 0)
    draft_recv = pp.get(("draft", "draft_recv_cmd"), {}).get("ms_per_step", 0)
    target_spec_wait = pp.get(("target", "target_spec_wait"), {}).get("ms_per_step", 0)
    target_total = (
        pp.get(("target", "graph_pre"), {}).get("ms_per_step", 0)
        + pp.get(("target", "graph_post"), {}).get("ms_per_step", 0)
        + pp.get(("target", "verify_sample_accept"), {}).get("ms_per_step", 0)
        + pp.get(("target", "verify_setup"), {}).get("ms_per_step", 0)
        + pp.get(("target", "exit_logits"), {}).get("ms_per_step", 0)
        + pp.get(("target", "final_logits"), {}).get("ms_per_step", 0)
        + pp.get(("target", "proxy_compute_send"), {}).get("ms_per_step", 0)
        + pp.get(("target", "target_postprocess"), {}).get("ms_per_step", 0)
    )
    proxy_arrival = (
        pp.get(("target", "graph_pre"), {}).get("ms_per_step", 0)
        + pp.get(("target", "proxy_compute_send"), {}).get("ms_per_step", 0)
    )
    # Cross-validate with actual draft Phase 1 end (for sanity):
    # measured proxy_arrival_inferred = T_glue_path + K1 * T_p1 + proxy_wait
    proxy_arrival_inferred = T_glue_path + K1 * T_p1 + proxy_wait
    return {
        "tag": os.path.basename(run_dir),
        "dfo": dfo, "pfo": pfo, "K": K, "K1": K1, "K2": K2,
        "T_p1": T_p1, "T_p2": T_p2,
        "T_glue_path": T_glue_path, "T_phase2_build": T_phase2_build,
        "proxy_arrival_target": proxy_arrival,
        "proxy_arrival_inferred": proxy_arrival_inferred,
        "target_total": target_total,
        "measured_proxy_wait": proxy_wait,
        "measured_draft_recv": draft_recv,
        "measured_target_spec_wait": target_spec_wait,
        "measured_TPS": _topline(run_dir).get("TPS") if _topline(run_dir) else None,
    }


def predict_idle(K1, K2, T_p1, T_p2, T_glue_path, T_phase2_build,
                  proxy_arrival, target_total):
    T_draft_p1_end = T_glue_path + K1 * T_p1
    proxy_wait = max(0.0, proxy_arrival - T_draft_p1_end)
    T_draft_p2_end = T_draft_p1_end + proxy_wait + T_phase2_build + K2 * T_p2
    target_spec_wait = max(0.0, T_draft_p2_end - target_total)
    draft_recv_cmd = max(0.0, target_total - T_draft_p2_end)
    return {
        "K1": K1, "K2": K2, "K": K1 + K2,
        "T_p1_end": T_draft_p1_end,
        "T_p2_end": T_draft_p2_end,
        "proxy_wait": proxy_wait,
        "target_spec_wait": target_spec_wait,
        "draft_recv_cmd": draft_recv_cmd,
        "total_idle": proxy_wait + target_spec_wait + draft_recv_cmd,
        "step_time": max(T_draft_p2_end, target_total),
    }


def predict_top2(baseline, K_max=12):
    """For a given baseline (one fanout combo), predict the top-2 (K1, K2)
    candidates that minimize total idle.
    """
    T_p1 = baseline["T_p1"]
    T_p2 = baseline["T_p2"]
    T_glue = baseline["T_glue_path"]
    T_p2b = baseline["T_phase2_build"]
    proxy_arr = baseline["proxy_arrival_inferred"]
    target_tot = baseline["target_total"]

    # Step 1: K1 candidates from proxy arrival budget.
    K1_real = (proxy_arr - T_glue) / T_p1
    K1_floor = max(1, int(math.floor(K1_real)))
    K1_ceil = max(1, int(math.ceil(K1_real)))
    if K1_ceil == K1_floor:
        K1_ceil = K1_floor + 1
    # Cap K1 so at least one K2 fits within K_max.
    K1_floor = min(K1_floor, K_max - 1)
    K1_ceil = min(K1_ceil, K_max - 1)
    K1_cands = sorted(set([K1_floor, K1_ceil]))

    # Step 2: For each K1, enumerate K2 ≥ 1 up to K_max - K1, predict idle.
    rows = []
    for K1 in K1_cands:
        K2_max = max(1, K_max - K1)
        for K2 in range(1, K2_max + 1):
            rows.append(predict_idle(
                K1, K2, T_p1, T_p2, T_glue, T_p2b, proxy_arr, target_tot,
            ))
    if not rows:
        # Fallback: at least try K1 = K_max-1, K2 = 1
        rows.append(predict_idle(
            K_max - 1, 1, T_p1, T_p2, T_glue, T_p2b, proxy_arr, target_tot,
        ))
    df = pd.DataFrame(rows)
    df = df.sort_values("total_idle").reset_index(drop=True)
    return df, K1_real


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: predictor.py <results_dir>")
    root = sys.argv[1]

    # Load all measured baselines.
    baselines = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        b = extract_baseline(d)
        if b is None:
            continue
        baselines.append(b)
    if not baselines:
        raise SystemExit(f"No baselines in {root}")

    df = pd.DataFrame(baselines)

    # ============================================================
    # Section 1 — measured per-forward times by fanout
    # ============================================================
    print("# Predictor: validate timing model against measured data\n")
    print("## 1. Per-forward times observed across fanout combos\n")
    print("| fanout | K1, K2 | T_p1 (ms/forward) | T_p2 (ms/forward) | "
          "T_glue_path | T_phase2_build | proxy_arrival_inferred | target_total |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in df.sort_values(["dfo", "pfo", "K1"]).iterrows():
        print(f"| dfo={r['dfo']},pfo={r['pfo']} | K1={r['K1']},K2={r['K2']} | "
              f"{r['T_p1']:.3f} | {r['T_p2']:.3f} | "
              f"{r['T_glue_path']:.2f} | {r['T_phase2_build']:.2f} | "
              f"{r['proxy_arrival_inferred']:.2f} | {r['target_total']:.2f} |")

    # ============================================================
    # Section 2 — validate model: predicted vs measured idle for each row
    # ============================================================
    print("\n## 2. Predicted vs measured idle (model self-consistency)\n")
    print("| run | predicted proxy_wait | measured | predicted spec_wait | measured | "
          "predicted recv_cmd | measured |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in df.sort_values(["dfo", "pfo", "K1"]).iterrows():
        pred = predict_idle(
            r["K1"], r["K2"], r["T_p1"], r["T_p2"],
            r["T_glue_path"], r["T_phase2_build"],
            r["proxy_arrival_inferred"], r["target_total"],
        )
        print(f"| {r['tag']} | {pred['proxy_wait']:.2f} | {r['measured_proxy_wait']:.2f} | "
              f"{pred['target_spec_wait']:.2f} | {r['measured_target_spec_wait']:.2f} | "
              f"{pred['draft_recv_cmd']:.2f} | {r['measured_draft_recv']:.2f} |")

    # ============================================================
    # Section 3 — predict top-2 (K1, K2) per fanout combo
    # ============================================================
    print("\n## 3. Predicted top-2 (K1, K2) per fanout combo\n")
    print("Algorithm: K1 = floor/ceil of (proxy_arrival - T_glue) / T_p1, "
          "then enumerate K2 ≤ K_max-K1 and pick lowest total idle.\n")

    # Use per-combo means of T_p1, T_p2, T_glue, etc.
    grouped = df.groupby(["dfo", "pfo"]).agg({
        "T_p1": "mean", "T_p2": "mean",
        "T_glue_path": "mean", "T_phase2_build": "mean",
        "proxy_arrival_inferred": "mean", "target_total": "mean",
    }).reset_index()

    top2_rows = []
    for _, g in grouped.iterrows():
        baseline = g.to_dict()
        cands, K1_real = predict_top2(baseline, K_max=10)
        best2 = cands.head(2)
        for rank, (_, c) in enumerate(best2.iterrows(), 1):
            top2_rows.append({
                "fanout": f"dfo={g['dfo']},pfo={g['pfo']}",
                "K1_real": round(K1_real, 2),
                "rank": rank,
                **c.to_dict(),
            })
        print(f"### dfo={int(g['dfo'])}, pfo={int(g['pfo'])}: K1_real ≈ {K1_real:.2f}")
        print(f"\n| rank | K1 | K2 | K | T_p1_end | T_p2_end | "
              f"proxy_wait | spec_wait | recv_cmd | **total_idle** |")
        print(f"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rank, (_, c) in enumerate(cands.head(4).iterrows(), 1):
            note = " ⭐ pick" if rank <= 2 else ""
            print(f"| {rank}{note} | {c['K1']:.0f} | {c['K2']:.0f} | {c['K']:.0f} | "
                  f"{c['T_p1_end']:.1f} | {c['T_p2_end']:.1f} | "
                  f"{c['proxy_wait']:.2f} | {c['target_spec_wait']:.2f} | "
                  f"{c['draft_recv_cmd']:.2f} | **{c['total_idle']:.2f}** |")
        print()

    pd.DataFrame(top2_rows).to_csv(
        os.path.join(root, "predictor_top2.csv"), index=False,
    )
    print(f"-> {root}/predictor_top2.csv")


if __name__ == "__main__":
    main()
