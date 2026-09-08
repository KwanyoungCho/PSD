#!/usr/bin/env python3
"""Aggregate profiler-off raw target latency for instrumentation validation."""
import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE / "profile_off_triad"


def weighted(rows, key):
    den = sum(row["n_verify_steps"] for row in rows)
    return 1000 * sum(row["n_verify_steps"] * row[key] for row in rows) / den


records = []
for path in sorted(RUN.glob("*_r*_s42_o256.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 7:
        continue
    arm = path.stem.split("_r", 1)[0]
    rep = int(path.stem.split("_r", 1)[1].split("_", 1)[0])
    step = weighted(rows, "mean_target_step_s")
    verify = weighted(rows, "mean_target_verify_s")
    records.append({"arm": arm, "repeat": rep, "target_step_ms": step,
                    "target_verify_ms": verify,
                    "outside_verify_ms": step - verify,
                    "verify_steps": sum(row["n_verify_steps"] for row in rows)})

if not records:
    raise SystemExit("no complete profile-off runs")
with (RUN / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=records[0])
    writer.writeheader(); writer.writerows(records)

arms = ("chain", "p2_tree", "full_tree")
lines = [
    "# Profiler-off triad validation", "",
    "Same tiny7/seed42/output256 triad with `SSD_PROFILE_DUET=0`. This table "
    "checks that the detailed CUDA-event instrumentation did not create the "
    "observed target-step gap.", "",
    "| Metric | Chain | P2 tree only | P1+P2 tree |",
    "|---|---:|---:|---:|",
]
for label, key in (("Target step", "target_step_ms"),
                   ("Target verify", "target_verify_ms"),
                   ("Outside verify", "outside_verify_ms")):
    cells = []
    for arm in arms:
        vals = [row[key] for row in records if row["arm"] == arm]
        if not vals:
            cells.append("—")
        else:
            spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.fmean(vals):.3f} ± {spread:.3f}")
    lines.append(f"| {label} | " + " | ".join(cells) + " |")

for right, label in (("p2_tree", "P2 tree − chain"),
                     ("full_tree", "Full tree − chain")):
    lines += ["", f"## {label}", ""]
    for key in ("target_step_ms", "target_verify_ms", "outside_verify_ms"):
        left = statistics.fmean(
            row[key] for row in records if row["arm"] == "chain")
        value = statistics.fmean(
            row[key] for row in records if row["arm"] == right)
        lines.append(f"- `{key}`: {value-left:+.3f} ms")

(RUN / "PROFILE_OFF_VALIDATION.md").write_text("\n".join(lines) + "\n")
print(RUN / "PROFILE_OFF_VALIDATION.md")
