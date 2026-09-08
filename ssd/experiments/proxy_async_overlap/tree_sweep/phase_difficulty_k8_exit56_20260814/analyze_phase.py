#!/usr/bin/env python3
"""Aggregate ordered P1/P2/miss events with question-cluster uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

SOURCES = {0: "Miss (fresh JIT)", 1: "P1", 2: "P2"}
GROUPS = ("mt_bench", "translation", "summarization", "qa",
          "math_reasoning", "rag")


def load_spec(spec: str):
    label, path = spec.split("=", 1)
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return label, Path(path), rows


def percentile(xs, p):
    ys = sorted(xs)
    if not ys:
        return math.nan
    x = (len(ys) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return ys[lo] if lo == hi else ys[lo] * (hi - x) + ys[hi] * (x - lo)


def cluster_ci(question_values, seed=20260814, nboot=10000):
    """Question-balanced mean and question bootstrap CI."""
    values = list(question_values.values())
    if not values:
        return math.nan, math.nan, math.nan
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    boots = [statistics.fmean(rng.choices(values, k=len(values)))
             for _ in range(nboot)]
    return mean, percentile(boots, .025), percentile(boots, .975)


def paired_source_contrast(events, left, right, seed=20260814):
    """Question-balanced LEFT - RIGHT contrast on questions containing both."""
    by_source_q = defaultdict(lambda: defaultdict(list))
    for event in events:
        by_source_q[event["source"]][event["question_key"]].append(
            event["accepted_len"])
    shared = (set(by_source_q[left]) & set(by_source_q[right]))
    differences = {
        q: (statistics.fmean(by_source_q[left][q])
            - statistics.fmean(by_source_q[right][q]))
        for q in shared
    }
    mean, lo, hi = cluster_ci(differences, seed=seed)
    return {
        "contrast": f"{left} - {right}", "questions": len(shared),
        "mean_difference": mean, "ci95_low": lo, "ci95_high": hi,
        "fraction_questions_below_zero": (
            sum(value < 0 for value in differences.values()) / len(differences)
            if differences else math.nan),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="LABEL=/path/to/raw.jsonl")
    ap.add_argument("--out-dir", default="analysis")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_events = []
    run_rows = {}
    for spec in args.inputs:
        label, path, rows = load_spec(spec)
        run_rows[label] = rows
        seen = set()
        for row in rows:
            uid = row["uid"]
            if uid in seen:
                raise RuntimeError(f"{label}: duplicate uid {uid}")
            seen.add(uid)
            events = row.get("phase_events")
            if events is None:
                raise RuntimeError(f"{label}/{uid}: phase_events missing")
            completion_offset = 1  # prefill emits the initial recovery token
            for index, event in enumerate(events):
                al = int(event["accepted_len"])
                # A final suffix can extend beyond max_new_tokens before the
                # scheduler truncates the saved output.  Keep its measured AL
                # but mark it unavailable for realized-prefix replay.
                fully_observed = (
                    completion_offset + al <= len(row["output_token_ids"]))
                all_events.append({
                    "arm": label, "uid": uid,
                    "question_id": row["question_id"],
                    "question_key": f'{row["group"]}:{row["question_id"]}',
                    "group": row["group"], "turn": row.get("turn", 0),
                    "seed": int(row.get("seed", -1)),
                    "event_index": index, "source_id": int(event["source"]),
                    "source": SOURCES[int(event["source"])],
                    "accepted_len": al,
                    "accepted_spec_len": int(event["accepted_spec_len"]),
                    "valid_k": event.get("valid_k"),
                    "completion_offset": completion_offset,
                    "fully_observed": int(fully_observed),
                })
                completion_offset += al

    event_fields = list(all_events[0])
    with (out / "phase_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=event_fields)
        w.writeheader(); w.writerows(all_events)

    summary = []
    by_subtask = []
    by_seed = []
    contrasts = []
    subtask_contrasts = []
    valid_k_sensitivity = []
    for arm in run_rows:
        arm_events = [e for e in all_events if e["arm"] == arm]
        for source_id, source in SOURCES.items():
            ev = [e for e in arm_events if e["source_id"] == source_id]
            by_q = defaultdict(list)
            for e in ev:
                by_q[e["question_key"]].append(e["accepted_len"])
            qmeans = {q: statistics.fmean(v) for q, v in by_q.items()}
            qmean, lo, hi = cluster_ci(qmeans)
            summary.append({
                "arm": arm, "source": source, "events": len(ev),
                "questions": len(by_q),
                "event_pooled_al": (statistics.fmean(
                    e["accepted_len"] for e in ev) if ev else math.nan),
                "question_mean_al": qmean, "ci95_low": lo, "ci95_high": hi,
                "p_al_eq_1": (sum(e["accepted_len"] == 1 for e in ev) / len(ev)
                              if ev else math.nan),
                "p_al_ge_3": (sum(e["accepted_len"] >= 3 for e in ev) / len(ev)
                              if ev else math.nan),
                "p_al_ge_5": (sum(e["accepted_len"] >= 5 for e in ev) / len(ev)
                              if ev else math.nan),
                "mean_valid_k": (statistics.fmean(e["valid_k"] for e in ev
                                                    if e["valid_k"] is not None)
                                 if ev else math.nan),
            })
            for seed in sorted({e["seed"] for e in ev}):
                sev = [e for e in ev if e["seed"] == seed]
                sq = defaultdict(list)
                for e in sev:
                    sq[e["question_key"]].append(e["accepted_len"])
                by_seed.append({
                    "arm": arm, "source": source, "seed": seed,
                    "events": len(sev), "questions": len(sq),
                    "event_pooled_al": statistics.fmean(
                        e["accepted_len"] for e in sev),
                    "question_mean_al": statistics.fmean(
                        statistics.fmean(v) for v in sq.values()),
                })
            for valid_k in sorted({e["valid_k"] for e in ev
                                   if e["valid_k"] is not None}):
                kev = [e for e in ev if e["valid_k"] == valid_k]
                kq = defaultdict(list)
                for e in kev:
                    kq[e["question_key"]].append(e["accepted_len"])
                km, klo, khi = cluster_ci({
                    q: statistics.fmean(values) for q, values in kq.items()
                }, seed=20260914 + int(valid_k))
                valid_k_sensitivity.append({
                    "arm": arm, "source": source, "valid_k": valid_k,
                    "events": len(kev), "questions": len(kq),
                    "question_mean_al": km,
                    "ci95_low": klo, "ci95_high": khi,
                })
            for group in GROUPS:
                gev = [e for e in ev if e["group"] == group]
                gq = defaultdict(list)
                for e in gev:
                    gq[e["question_key"]].append(e["accepted_len"])
                gqm = {q: statistics.fmean(v) for q, v in gq.items()}
                gm, glo, ghi = cluster_ci(gqm, seed=20260814 + GROUPS.index(group))
                by_subtask.append({
                    "arm": arm, "source": source, "group": group,
                    "events": len(gev), "questions": len(gq),
                    "question_mean_al": gm, "ci95_low": glo,
                    "ci95_high": ghi,
                })
        for ci, (left, right) in enumerate((
                ("P2", "P1"), ("Miss (fresh JIT)", "P1"),
                ("P2", "Miss (fresh JIT)"))):
            contrast = paired_source_contrast(
                arm_events, left, right, seed=20260814 + ci)
            contrast["arm"] = arm
            contrasts.append(contrast)
        for gi, group in enumerate(GROUPS):
            group_events = [e for e in arm_events if e["group"] == group]
            contrast = paired_source_contrast(
                group_events, "P2", "P1", seed=20261114 + gi)
            contrast["arm"] = arm
            contrast["group"] = group
            subtask_contrasts.append(contrast)

    arm_contrasts = []
    chain_arms = [arm for arm in run_rows if "chain" in arm.lower()]
    tree_arms = [arm for arm in run_rows if "tree" in arm.lower()]
    if len(chain_arms) == 1 and len(tree_arms) == 1:
        chain_arm, tree_arm = chain_arms[0], tree_arms[0]
        for source_id, source in SOURCES.items():
            by_arm_q = defaultdict(lambda: defaultdict(list))
            for event in all_events:
                if event["source_id"] == source_id:
                    by_arm_q[event["arm"]][event["question_key"]].append(
                        event["accepted_len"])
            shared = set(by_arm_q[tree_arm]) & set(by_arm_q[chain_arm])
            differences = {
                q: (statistics.fmean(by_arm_q[tree_arm][q])
                    - statistics.fmean(by_arm_q[chain_arm][q]))
                for q in shared
            }
            mean, lo, hi = cluster_ci(
                differences, seed=20261014 + source_id)
            arm_contrasts.append({
                "source": source,
                "contrast": f"{tree_arm} - {chain_arm}",
                "questions": len(shared), "mean_difference": mean,
                "ci95_low": lo, "ci95_high": hi,
                "fraction_questions_above_zero": (
                    sum(value > 0 for value in differences.values())
                    / len(differences) if differences else math.nan),
            })

    def write_csv(path, rows):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    write_csv(out / "phase_summary.csv", summary)
    write_csv(out / "phase_by_subtask.csv", by_subtask)
    write_csv(out / "phase_by_seed.csv", by_seed)
    write_csv(out / "phase_contrasts.csv", contrasts)
    write_csv(out / "phase_contrasts_by_subtask.csv", subtask_contrasts)
    write_csv(out / "phase_by_valid_k.csv", valid_k_sensitivity)
    if arm_contrasts:
        write_csv(out / "arm_contrasts.csv", arm_contrasts)

    # A compact Markdown table is kept independent of plotting style.
    lines = [
        "# K1=K2=8 phase-difficulty summary", "",
        "Primary AL includes the one correction/recovery token. Confidence ",
        "intervals resample questions, so long generations do not become ",
        "independent pseudo-replicates.", "",
        "| Arm | Source | Events | Questions | Event-pooled AL | Question-mean AL [95% CI] | P(AL=1) | P(AL≥3) | P(AL≥5) | Mean valid-k |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['arm']} | {r['source']} | {r['events']} | {r['questions']} | "
            f"{r['event_pooled_al']:.3f} | {r['question_mean_al']:.3f} "
            f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] | "
            f"{r['p_al_eq_1']:.3f} | {r['p_al_ge_3']:.3f} | "
            f"{r['p_al_ge_5']:.3f} | {r['mean_valid_k']:.2f} |")
    lines.extend([
        "", "## Within-question phase contrasts", "",
        "Negative `P2 - P1` means P2 has lower accepted length. The CI and ",
        "sign fraction use questions as the independent unit.", "",
        "| Arm | Contrast | Questions | Mean difference [95% CI] | Fraction below zero |",
        "|---|---|---:|---:|---:|",
    ])
    for r in contrasts:
        lines.append(
            f"| {r['arm']} | {r['contrast']} | {r['questions']} | "
            f"{r['mean_difference']:.3f} [{r['ci95_low']:.3f}, "
            f"{r['ci95_high']:.3f}] | "
            f"{r['fraction_questions_below_zero']:.3f} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
