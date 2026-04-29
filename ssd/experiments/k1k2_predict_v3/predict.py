"""Single-shot K1/K2 predictor.

Workflow:
  1. Run ONE baseline measurement at K1=2, K2=2 per (dfo, pfo) fanout combo (~1-2 min each).
  2. From that single measurement, directly predict optimal (K1, K2). No brute force.

Constraints applied:
  - K1 capped at K_max/2 (early-exit fraction; for 70B with exit_layer=40 of 80, ratio≈0.5)
  - K1 must absorb proxy arrival: K1 ≥ ⌈(proxy_arr − T_glue) / T_p1⌉ unless capped
  - K2 fills remaining target budget: K2 = ⌊(T_target_extrapolated − K1·T_p1 − T_glue − T_p2b) / T_p2⌋
  - T_target extrapolated linearly in (K+1) positions: T_target(K) ≈ T_target_baseline · (K+1)/(K_b+1)

Usage:
  python predict.py <baseline_dir>
"""
import os
import re
import sys
import math
import json
from glob import glob

import pandas as pd


def per_phase(run_dir):
    out = {}
    rows_per_proc = {}
    for tag in ("draft", "target_rank0"):
        paths = sorted(glob(os.path.join(run_dir, f"mesa_profile_{tag}_*.json")))
        if not paths:
            continue
        proc = tag.replace("_rank0", "")
        rows = json.load(open(paths[-1]))
        df = pd.DataFrame(rows).sort_values(["label", "idx"])
        df["_rank"] = df.groupby("label").cumcount()
        df = df[df["_rank"] >= 5].drop(columns="_rank").reset_index(drop=True)
        rows_per_proc[proc] = df
    if not rows_per_proc:
        return None

    n_steps_draft = max(1, int((rows_per_proc.get("draft", pd.DataFrame())
                                .get("label", pd.Series([])) == "merge_cache").sum())) if "draft" in rows_per_proc else 1
    n_steps_target = max(1, int((rows_per_proc.get("target", pd.DataFrame())
                                 .get("label", pd.Series([])) == "target_postprocess").sum())) if "target" in rows_per_proc else 1

    for proc, df in rows_per_proc.items():
        n_steps = n_steps_draft if proc == "draft" else n_steps_target
        for label, sub in df.groupby("label"):
            mean_ms = float(sub["ms"].mean())
            n_events = len(sub)
            out[(proc, label)] = {
                "mean_ms": mean_ms,
                "events_per_step": n_events / n_steps,
                "ms_per_step": mean_ms * (n_events / n_steps),
            }
    return out


def extract(run_dir):
    pp = per_phase(run_dir)
    if pp is None:
        return None
    parsed = re.match(
        r"dfo(\d+)_pfo(\d+)_K(\d+)_K1_(\d+)_K2_(\d+)", os.path.basename(run_dir),
    )
    if not parsed:
        return None
    dfo, pfo, K, K1, K2 = (int(parsed.group(i)) for i in range(1, 6))
    T_p1 = pp.get(("draft", "phase1_replay"), {}).get("mean_ms", 0)
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
    )
    T_phase2_build = pp.get(("draft", "phase2_hybrid_build"), {}).get("ms_per_step", 0)
    proxy_arr_target = (
        pp.get(("target", "graph_pre"), {}).get("ms_per_step", 0)
        + pp.get(("target", "proxy_compute_send"), {}).get("ms_per_step", 0)
    )
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
    return {
        "tag": os.path.basename(run_dir),
        "dfo": dfo, "pfo": pfo, "K_baseline": K, "K1_b": K1, "K2_b": K2,
        "T_p1": T_p1, "T_p2": T_p2,
        "T_glue_path": T_glue_path, "T_phase2_build": T_phase2_build,
        "proxy_arrival": proxy_arr_target,
        "target_total": target_total,
    }


def predict_k1_k2(b, K_max=10, K1_cap_ratio=0.5):
    """Single-shot prediction.

    K1: smaller of (proxy_arr-fitting K1) and (K_max * K1_cap_ratio cap)
    K2: fills remaining budget at extrapolated T_target

    Returns dict with K1, K2, predicted timings.
    """
    T_p1 = b["T_p1"]
    T_p2 = b["T_p2"]
    T_glue = b["T_glue_path"]
    T_p2b = b["T_phase2_build"]
    proxy_arr = b["proxy_arrival"]
    T_target_b = b["target_total"]
    K_b = b["K_baseline"]

    K1_proxy_real = max(1.0, (proxy_arr - T_glue) / T_p1) if T_p1 > 0 else 1.0
    K1_cap = max(1, int(round(K_max * K1_cap_ratio)))
    K1 = max(1, min(int(round(K1_proxy_real)), K1_cap, K_max - 1))

    # Iterate K2 with extrapolated T_target
    K2 = 1
    for _ in range(8):
        K = K1 + K2
        T_target_K = T_target_b * (K + 1) / (K_b + 1)
        budget = T_target_K - K1 * T_p1 - T_glue - T_p2b
        K2_new = max(1, min(K_max - K1, int(math.floor(budget / T_p2)))) if T_p2 > 0 else K_max - K1
        if K2_new == K2:
            break
        K2 = K2_new

    K = K1 + K2
    T_p1_end = T_glue + K1 * T_p1
    T_p2_end = T_p1_end + max(0, proxy_arr - T_p1_end) + T_p2b + K2 * T_p2
    T_target_K = T_target_b * (K + 1) / (K_b + 1)
    proxy_wait = max(0.0, proxy_arr - T_p1_end)
    spec_wait = max(0.0, T_p2_end - T_target_K)
    recv_cmd = max(0.0, T_target_K - T_p2_end)
    return {
        "K1": K1, "K2": K2, "K": K,
        "K1_proxy_real": round(K1_proxy_real, 2),
        "T_p1_end": T_p1_end, "T_p2_end": T_p2_end,
        "T_target_K": T_target_K,
        "proxy_wait": proxy_wait, "spec_wait": spec_wait, "recv_cmd": recv_cmd,
        "step_time": max(T_p2_end, T_target_K),
        "total_idle": proxy_wait + spec_wait + recv_cmd,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: predict.py <baseline_dir> [K_max] [K1_cap_ratio]")
    root = sys.argv[1]
    K_max = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    K1_cap_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    baselines = []
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        b = extract(d)
        if b is None:
            continue
        baselines.append(b)

    print(f"# Single-shot K1/K2 predictor (K_max={K_max}, K1_cap_ratio={K1_cap_ratio})\n")
    print("## Baseline measurements (one per fanout combo)\n")
    print("| fanout | baseline | T_p1 | T_p2 | T_glue | T_p2b | proxy_arr | target_total |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for b in baselines:
        print(f"| dfo={b['dfo']},pfo={b['pfo']} | K1={b['K1_b']},K2={b['K2_b']} | "
              f"{b['T_p1']:.3f} | {b['T_p2']:.3f} | {b['T_glue_path']:.2f} | "
              f"{b['T_phase2_build']:.2f} | {b['proxy_arrival']:.2f} | {b['target_total']:.2f} |")

    print("\n## Predicted (K1, K2) per fanout\n")
    print("| fanout | K1_proxy_real | K1 | K2 | K | T_p1_end | T_p2_end | T_target_K | proxy_wait | spec_wait | recv_cmd | total_idle |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    out_rows = []
    for b in baselines:
        p = predict_k1_k2(b, K_max=K_max, K1_cap_ratio=K1_cap_ratio)
        print(f"| dfo={b['dfo']},pfo={b['pfo']} | {p['K1_proxy_real']} | "
              f"**{p['K1']}** | **{p['K2']}** | {p['K']} | "
              f"{p['T_p1_end']:.1f} | {p['T_p2_end']:.1f} | {p['T_target_K']:.1f} | "
              f"{p['proxy_wait']:.2f} | {p['spec_wait']:.2f} | {p['recv_cmd']:.2f} | "
              f"**{p['total_idle']:.2f}** |")
        out_rows.append({"dfo": b["dfo"], "pfo": b["pfo"],
                         "K1": p["K1"], "K2": p["K2"], "K": p["K"]})
    pd.DataFrame(out_rows).to_csv(os.path.join(root, "predictions.csv"), index=False)
    print(f"\n-> {root}/predictions.csv")


if __name__ == "__main__":
    main()
