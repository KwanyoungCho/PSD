#!/usr/bin/env python3
"""Summarize target exit/proxy timeline spans by cache status."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


LABELS = (
    "exit_logits",
    "exit_proxy_launch",
    "exit_proxy_side",
    "chain_proxy_graph_replay",
    "tree_proxy_graph_replay",
    "proxy_send_enqueue",
    # Legacy labels remain visible when reading an older profile.
    "proxy_compute_send",
    "proxy_compute",
    "proxy_pack",
    "proxy_send",
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def latest_target_json(directory: Path) -> Path:
    candidates = sorted(
        directory.glob("duet_profile_target_rank0_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no duet_profile_target_rank0_*.json under {directory}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    args = parser.parse_args()
    source = latest_target_json(args.profile_dir)
    rows = json.loads(source.read_text())

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        label = str(row.get("label", ""))
        if label not in LABELS:
            continue
        status = str(row.get("status") or "unknown")
        grouped.setdefault((status, label), []).append(row)

    print(f"source: {source}")
    print("status   label                         n  cuda_p50  cuda_p95  cpu_p50")
    for (status, label), entries in sorted(grouped.items()):
        cuda = [float(r.get("cuda_ms", r.get("ms", 0.0))) for r in entries]
        cpu = [
            (int(r["cpu_dispatch_end_ns"]) - int(r["cpu_dispatch_start_ns"]))
            / 1e6
            for r in entries
            if r.get("cpu_dispatch_start_ns") is not None
            and r.get("cpu_dispatch_end_ns") is not None
        ]
        print(
            f"{status:<8} {label:<29} {len(entries):>4} "
            f"{statistics.median(cuda):>9.3f} "
            f"{percentile(cuda, 0.95):>9.3f} "
            f"{statistics.median(cpu) if cpu else float('nan'):>8.3f}"
        )


if __name__ == "__main__":
    main()
