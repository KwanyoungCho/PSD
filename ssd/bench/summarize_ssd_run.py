"""Summarize one SSD experiment run and generate hit/miss timelines.

This postprocess script intentionally emits only paper-table metrics and two
timeline plots:

  - timeline_cache_hit.png
  - timeline_cache_miss.png

It supports both new profiles with ``target_spec_wait_{hit,miss}`` labels and
older profiles by inferring hit/miss from the bimodal ``hit_cache_respond``
duration distribution.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Aligned-schema (Phase-B) plotter — used when the profile JSON has _anchor
# and wall_start_ns / step_id.  Legacy JSON still falls through to the
# existing approximate plotter below.
try:
    from . import plot_duet_aligned_timeline as aligned
except ImportError:  # running as a script, not as a package
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import plot_duet_aligned_timeline as aligned  # type: ignore[no-redef]


def _latest_profile(outdir: Path, tag: str) -> Path | None:
    # Accepts both new duet_profile_*.json and legacy mesa_profile_*.json
    paths = sorted(outdir.glob(f"duet_profile_{tag}_*.json"))
    if not paths:
        paths = sorted(outdir.glob(f"duet_profile_{tag}.json"))
    if not paths:
        paths = sorted(outdir.glob(f"mesa_profile_{tag}_*.json"))
    if not paths:
        paths = sorted(outdir.glob(f"mesa_profile_{tag}.json"))
    return paths[-1] if paths else None


def _load_profile(outdir: Path, tag: str) -> list[dict]:
    path = _latest_profile(outdir, tag)
    if path is None:
        return []
    with path.open() as f:
        return json.load(f)


def _load_profile_raw(outdir: Path, tag: str) -> list[dict]:
    """Like ``_load_profile`` but preserves the ``_anchor`` row for schema detection."""
    return _load_profile(outdir, tag)


def _metric(pattern: str, text: str, group: int | str = 1, cast=float):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    value = m.group(group)
    return cast(value)


def parse_run_log(outdir: Path) -> dict:
    log = outdir / "run.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    row: dict[str, object] = {
        "run_dir": str(outdir),
        "run_name": outdir.name,
    }

    final = re.search(
        r"Model:\s*(?P<model>[^,]+),\s*Mode:\s*(?P<mode>.*?),\s*"
        r"Total:\s*(?P<tokens>\d+)tok,\s*Time:\s*(?P<time>[\d.]+)s,\s*"
        r"Total Throughput:\s*(?P<tps>[\d.]+)tok/s",
        text,
    )
    if final:
        row.update(
            model=final.group("model"),
            mode=final.group("mode"),
            total_tokens=int(final.group("tokens")),
            wall_time_s=float(final.group("time")),
            total_tps=float(final.group("tps")),
        )

    row.update(
        prefill_tps=_metric(r"Final Prefill Throughput:\s*([\d.]+)tok/s", text),
        decode_tps=_metric(r"Final Decode Throughput:\s*([\d.]+)tok/s", text),
        prefill_total_tokens=_metric(r"Prefill tokens/time:\s*([\d.]+)\s*/", text, cast=int),
        prefill_total_time_s=_metric(r"Prefill tokens/time:\s*[\d.]+\s*/\s*([\d.]+)s", text),
        decode_total_tokens=_metric(r"Decode tokens/time:\s*([\d.]+)\s*/", text, cast=int),
        decode_total_time_s=_metric(r"Decode tokens/time:\s*[\d.]+\s*/\s*([\d.]+)s", text),
        avg_tokens_per_step=_metric(r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)", text),
        accept_fraction=_metric(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)", text),
        target_full_step_ms=_metric(r"Avg target time per full step \(ms\):\s*([\d.]+)", text),
        target_verify_ms=_metric(r"Avg target verify time \(ms\):\s*([\d.]+)", text),
        cache_hit_rate=_metric(r"Avg Cache Hits:\s*([\d.]+)", text),
        p1_hit=_metric(r"Avg Phase 1 \(draft\) Hit Rate:\s*([\d.]+)", text),
        p2_hit=_metric(r"Avg Phase 2 \(proxy\) Hit Rate:\s*([\d.]+)", text),
        p1_avg_accepted_len=_metric(r"Avg Phase 1 Accepted Len:\s*([\d.]+)", text),
        p1_acceptance_ratio=_metric(r"Avg Phase 1 Acceptance Ratio:\s*([\d.]+)", text),
        p2_avg_accepted_len=_metric(r"Avg Phase 2 Accepted Len:\s*([\d.]+)", text),
        p2_acceptance_ratio=_metric(r"Avg Phase 2 Acceptance Ratio:\s*([\d.]+)", text),
        duet_exit_layer=_metric(r"DUET-SSD enabled: exit_layer=(\d+)", text, cast=int),
        duet_proxy_top_k=_metric(r"proxy_top_k=(\d+)", text, cast=int),
        duet_draft_fan_out=_metric(r"draft_fan_out=(\d+)", text, cast=int),
        duet_proxy_fan_out=_metric(r"proxy_fan_out=(\d+)", text, cast=int),
        duet_phase1_k=_metric(r"K1=(\d+)", text, cast=int),
        duet_phase2_k=_metric(r"K2=(\d+)", text, cast=int),
        tokens_per_step_on_hit=_metric(r"Avg Tokens per step on Cache Hit:\s*([\d.]+)", text),
        tokens_per_step_on_miss=_metric(r"Avg Tokens per step on Cache Miss:\s*([\d.]+)", text),
        draft_step_ms=_metric(r"Avg draft step time \(ms\):\s*([\d.]+)", text),
    )
    return row


def _as_df(rows: list[dict], proc: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["idx", "label", "start_ms", "end_ms", "ms", "proc"])
    df = pd.DataFrame(rows)
    df["proc"] = proc
    return df


def _drop_warmup(df: pd.DataFrame, warmup: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values(["proc", "label", "idx"]).copy()
    out["_rank"] = out.groupby(["proc", "label"]).cumcount()
    return out[out["_rank"] >= warmup].drop(columns=["_rank"])


def _mean(df: pd.DataFrame, labels: Iterable[str]) -> float | None:
    labels = set(labels)
    sub = df[df["label"].isin(labels)]
    if sub.empty:
        return None
    return float(sub["ms"].mean())


def _mean_prefix(df: pd.DataFrame, prefix: str) -> float | None:
    sub = df[df["label"].astype(str).str.startswith(prefix)]
    if sub.empty:
        return None
    return float(sub["ms"].mean())


def add_profile_metrics(row: dict, target: pd.DataFrame, draft: pd.DataFrame, k: int, warmup: int) -> None:
    td = _drop_warmup(target, warmup)
    dd = _drop_warmup(draft, warmup)

    row.update(
        prof_target_verify_replay_ms=_mean(td, ["verify_replay"]),
        prof_target_sample_accept_ms=_mean(td, ["verify_sample_accept"]),
        prof_target_spec_wait_ms=_mean_prefix(td, "target_spec_wait"),
        prof_target_proxy_compute_ms=_mean(td, ["proxy_compute"]),
        prof_target_proxy_pack_ms=_mean(td, ["proxy_pack"]),
        prof_target_proxy_send_ms=_mean(td, ["proxy_send"]),
        prof_draft_glue_ms=_mean(dd, ["glue"]),
        prof_draft_glue_replay_ms=_mean(dd, ["draft_glue_replay"]),
        prof_draft_recv_cmd_ms=_mean(dd, ["draft_recv_cmd"]),
        prof_draft_send_response_ms=_mean(dd, ["draft_send_response"]),
        prof_draft_cache_respond_ms=_mean_prefix(dd, "hit_cache_respond"),
        prof_draft_proxy_wait_ms=_mean(dd, ["proxy_wait"]),
        prof_draft_phase1_build_ms=_mean(dd, ["phase1_build"]),
        prof_draft_phase1_prep_ms=_mean(dd, ["phase1_prep"]),
        prof_draft_phase1_replay_ms=_mean(dd, ["phase1_replay"]),
        prof_draft_phase2_build_ms=_mean(dd, ["phase2_build"]),
        prof_draft_phase2_prep_ms=_mean(dd, ["phase2_prep"]),
        prof_draft_phase2_replay_ms=_mean(dd, ["phase2_replay"]),
    )

    tree_prep = _mean(dd, ["tree_prep"])
    tree_replay = _mean(dd, ["tree_replay"])
    if tree_prep is not None:
        row["prof_draft_tree_prep_ms_per_step"] = tree_prep * k
        row["prof_draft_tree_prep_ms_per_invocation"] = tree_prep
    if tree_replay is not None:
        row["prof_draft_tree_replay_ms_per_step"] = tree_replay * k
        row["prof_draft_tree_replay_ms_per_invocation"] = tree_replay


def _status_from_label(label: str) -> str | None:
    if label.endswith("_hit_k1"):
        return "hit_k1"
    if label.endswith("_hit_k2"):
        return "hit_k2"
    if label.endswith("_hit"):
        return "hit"
    if label.endswith("_miss"):
        return "miss"
    if label.endswith("_mixed"):
        return "mixed"
    return None


def _status_matches(actual: str | None, wanted: str) -> bool:
    if actual == wanted:
        return True
    if wanted == "hit" and actual in {"hit", "hit_k1", "hit_k2"}:
        return True
    return False


def _target_wait_statuses(
    target: pd.DataFrame,
    draft: pd.DataFrame,
    threshold_ms: float,
) -> list[str | None]:
    waits = _target_wait_events(target).to_dict("records")
    statuses = [_status_from_label(str(w["label"])) for w in waits]
    if not any(s in ("hit", "miss", "hit_k1", "hit_k2") for s in statuses):
        statuses = _draft_response_statuses(draft, threshold_ms)
    return statuses


def _draft_response_statuses(draft: pd.DataFrame, threshold_ms: float) -> list[str | None]:
    sends = draft[draft["label"] == "draft_send_response"].sort_values("start_ms").to_dict("records")
    responds = draft[draft["label"].astype(str).str.startswith("hit_cache_respond")].sort_values("start_ms").to_dict("records")
    statuses: list[str | None] = []
    for i, _send in enumerate(sends):
        if i >= len(responds):
            statuses.append(None)
            continue
        r = responds[i]
        by_label = _status_from_label(str(r["label"]))
        if by_label:
            statuses.append(by_label)
        else:
            statuses.append("hit" if float(r["ms"]) < threshold_ms else "miss")
    return statuses


def _target_wait_events(target: pd.DataFrame) -> pd.DataFrame:
    return target[target["label"].astype(str).str.startswith("target_spec_wait")].sort_values("start_ms")


def _verify_replay_ms_by_wait_index(target: pd.DataFrame, waits: list[dict]) -> dict[int, float]:
    """Map wait index to the preceding target verify replay duration.

    For handshake-tail timelines, the selected wait is the right edge of the
    window.  The verify replay just before that wait is what visually dominates
    the target row, so timeline examples should avoid choosing hit/miss pairs
    with unusually different preceding replay durations.
    """
    replays = target[target["label"] == "verify_replay"].sort_values("start_ms").to_dict("records")
    out: dict[int, float] = {}
    j = 0
    for i, wait in enumerate(waits):
        lo = float(waits[i - 1]["end_ms"]) if i > 0 else float(wait["start_ms"])
        hi = float(wait["start_ms"])
        while j < len(replays) and float(replays[j]["end_ms"]) < lo:
            j += 1
        k = j
        while k < len(replays) and float(replays[k]["start_ms"]) <= hi:
            r = replays[k]
            if float(r["end_ms"]) >= lo:
                out[i] = float(r["ms"])
                break
            k += 1
    return out


def _choose_pair_index(
    target: pd.DataFrame,
    draft: pd.DataFrame,
    wanted: str,
    warmup_steps: int,
    threshold_ms: float,
) -> int | None:
    waits = _target_wait_events(target).to_dict("records")
    statuses = _target_wait_statuses(target, draft, threshold_ms)

    n = min(len(waits), len(statuses))
    candidates = [i for i in range(warmup_steps, n - 2) if _status_matches(statuses[i], wanted)]
    if not candidates:
        return None
    replay_by_wait = _verify_replay_ms_by_wait_index(target, waits)
    replay_vals = [replay_by_wait[i] for i in range(warmup_steps, n - 2) if i in replay_by_wait]
    if not replay_vals:
        return candidates[len(candidates) // 2]
    median_replay = statistics.median(replay_vals)
    mid_candidate = candidates[len(candidates) // 2]
    return min(
        candidates,
        key=lambda i: (
            abs(replay_by_wait.get(i, median_replay) - median_replay),
            abs(i - mid_candidate),
        ),
    )


COLORS = {
    # Target-side DUET events
    "verify_setup": ("#969696", ".."),
    "graph_pre": ("#8b1a1a", ""),
    "exit_logits": ("#fdae61", ""),
    "proxy_compute_send": ("#1b9e77", ""),
    "proxy_compute": ("#66c2a5", ""),
    "proxy_pack": ("#1b9e77", ".."),
    "proxy_send": ("#006d2c", ""),
    "graph_post": ("#8073ac", ""),
    "final_logits": ("#fee08b", ""),
    "verify_replay": ("#7a0000", ""),
    "verify_sample_accept": ("#fc8d59", ""),
    "target_postprocess": ("#feb24c", ".."),
    "target_spec_wait": ("#ffeda0", ""),
    "target_spec_wait_hit": ("#c7e9c0", ""),
    "target_spec_wait_hit_k1": ("#74c476", ""),
    "target_spec_wait_hit_k2": ("#238b45", ""),
    "target_spec_wait_miss": ("#fdd0a2", ""),
    "target_spec_wait_mixed": ("#d9d9d9", "xx"),
    # Draft-side generic SSD events
    "tree_prep": ("#feb24c", "//"),
    "tree_replay": ("#e6550d", ""),
    "glue": ("#01665e", ""),
    "draft_glue_replay": ("#5ab4ac", ""),
    "hit_cache_respond": ("#80cdc1", "//"),
    "hit_cache_respond_hit": ("#80cdc1", "//"),
    "hit_cache_respond_hit_k1": ("#67a9cf", "//"),
    "hit_cache_respond_hit_k2": ("#2166ac", "//"),
    "hit_cache_respond_miss": ("#fb6a4a", "//"),
    "hit_cache_respond_mixed": ("#bdbdbd", "//"),
    "draft_recv_cmd": ("#4d4d4d", "..."),
    "draft_send_response": ("#252525", ""),
    # Draft-side split-K1/K2 DUET events
    "phase1_build": ("#9ecae1", ".."),
    "phase1_prep": ("#6baed6", "//"),
    "phase1_replay": ("#08519c", ""),
    "proxy_wait": ("#ffd92f", ""),
    "phase2_build": ("#a1d99b", ".."),
    "phase2_prep": ("#74c476", "//"),
    "phase2_replay": ("#006d2c", ""),
    "merge_cache": ("#bcbddc", "\\\\"),
}


def _style(label: str):
    return COLORS.get(label, ("#d9d9d9", ""))


def _in_window(row: dict, lo: float, hi: float) -> bool:
    return float(row["end_ms"]) >= lo and float(row["start_ms"]) <= hi


def plot_timeline_for_pair(
    outdir: Path,
    target: pd.DataFrame,
    draft: pd.DataFrame,
    pair_i: int,
    status: str,
    warmup_steps: int,
    window: str,
) -> Path:
    t_anchor = _target_wait_events(target).to_dict("records")
    d_anchor = draft[draft["label"] == "draft_send_response"].sort_values("start_ms").to_dict("records")
    n = min(len(t_anchor), len(d_anchor))
    if pair_i >= n - 1:
        raise ValueError(f"pair index {pair_i} out of range for {n} handshakes")

    offsets = [float(t_anchor[i]["end_ms"]) - float(d_anchor[i]["end_ms"]) for i in range(n)]
    steady = offsets[min(warmup_steps, max(0, n - 1)) :] or offsets
    offset = sum(steady) / len(steady)

    cur_wait_end_t = float(t_anchor[pair_i]["end_ms"])
    # Default view puts the selected hit/miss handshake at the tail of the
    # timeline: previous response -> verify/tree work -> selected handshake.
    # This keeps hit/miss latency visible without making the plot read
    # "handshake first, then work", which is awkward for pipeline inspection.
    if window == "handshake_tail":
        if pair_i > 0:
            win_lo_t = float(t_anchor[pair_i - 1]["end_ms"])
        else:
            win_lo_t = float(t_anchor[pair_i]["start_ms"])
        win_hi_t = cur_wait_end_t
    elif window == "full_handshake":
        win_lo_t = float(t_anchor[pair_i]["start_ms"])
        if pair_i + 1 < n:
            win_hi_t = float(t_anchor[pair_i + 1]["start_ms"])
        else:
            win_hi_t = cur_wait_end_t
    else:
        win_lo_t = cur_wait_end_t
        if pair_i + 1 < n:
            win_hi_t = float(t_anchor[pair_i + 1]["start_ms"])
        else:
            win_hi_t = cur_wait_end_t
    win_lo_d = win_lo_t - offset
    win_hi_d = win_hi_t - offset

    target_rows = target.to_dict("records")
    draft_rows = draft.to_dict("records")
    t_win = [e for e in target_rows if _in_window(e, win_lo_t, win_hi_t)]
    d_win = [e for e in draft_rows if _in_window(e, win_lo_d, win_hi_d)]
    # In detail profiling, proxy_compute_send is an outer span that contains
    # proxy_compute/proxy_pack/proxy_send. Hide the outer span in timelines so
    # the useful breakdown is not painted over by the parent bar.
    detail_proxy_labels = {"proxy_compute", "proxy_pack", "proxy_send"}
    if any(str(e["label"]) in detail_proxy_labels for e in t_win):
        t_win = [e for e in t_win if str(e["label"]) != "proxy_compute_send"]

    origin = win_lo_t
    fig, ax = plt.subplots(figsize=(16, 5.2))
    bar_h = 0.6

    def plot_row(events: list[dict], y: int, frame_shift: float) -> None:
        for e in events:
            label = str(e["label"])
            x = float(e["start_ms"]) + frame_shift - origin
            w = max(float(e["ms"]), 0.02)
            color, hatch = _style(label)
            ax.barh(
                y,
                w,
                left=x,
                height=bar_h,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
            )

    plot_row(t_win, 1, 0.0)
    plot_row(d_win, 0, offset)

    response_x = cur_wait_end_t - origin
    ax.axvline(response_x, color="black", lw=1.0, alpha=0.55, linestyle=":")
    ax.text(
        response_x + 0.2,
        1.68,
        "response received",
        fontsize=8,
        ha="left",
        va="center",
        color="black",
        alpha=0.75,
    )
    wait_start_x = float(t_anchor[pair_i]["start_ms"]) - origin
    if 0 <= wait_start_x <= max(win_hi_t - origin, 1.0):
        ax.axvline(wait_start_x, color="black", lw=0.8, alpha=0.35, linestyle="--")
    if pair_i + 1 < len(t_anchor) and window != "handshake_tail":
        next_x = float(t_anchor[pair_i + 1]["start_ms"]) - origin
        ax.axvline(next_x, color="black", lw=0.8, alpha=0.35, linestyle="--")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["draft", "target"])
    ax.set_xlabel("time (ms, target frame; draft offset-aligned)")
    title_status = status.replace("_", " ").upper() if status.startswith("hit_") else status
    ax.set_title(f"SSD {window} timeline - cache {title_status} (pair #{pair_i}) - {outdir.name}")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_xlim(0, max(win_hi_t - origin, 1.0))

    present = {str(e["label"]) for e in t_win + d_win}
    handles = []
    for label in sorted(present):
        color, hatch = _style(label)
        handles.append(
            mpatches.Patch(
                facecolor=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                label=label,
            )
        )
    if handles:
        n_cols = min(6, max(3, (len(handles) + 1) // 2))
        ax.legend(
            handles=handles,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=n_cols,
            frameon=True,
            framealpha=0.9,
        )
    plt.subplots_adjust(bottom=0.32)
    suffix = "" if window == "handshake_tail" else f"_{window}"
    path = outdir / f"timeline_cache_{status}{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


PREFERRED_COLUMNS = [
    "run_name",
    "model",
    "mode",
    "total_tokens",
    "wall_time_s",
    "total_tps",
    "decode_tps",
    "prefill_tps",
    "avg_tokens_per_step",
    "accept_fraction",
    "cache_hit_rate",
    "tokens_per_step_on_hit",
    "tokens_per_step_on_miss",
    "target_full_step_ms",
    "target_verify_ms",
    "draft_step_ms",
    "prof_target_verify_replay_ms",
    "prof_target_sample_accept_ms",
    "prof_target_spec_wait_ms",
    "prof_target_proxy_compute_ms",
    "prof_target_proxy_pack_ms",
    "prof_target_proxy_send_ms",
    "prof_draft_tree_replay_ms_per_step",
    "prof_draft_tree_prep_ms_per_step",
    "prof_draft_glue_ms",
    "prof_draft_recv_cmd_ms",
    "prof_draft_cache_respond_ms",
    "run_dir",
]


def _ordered_row(row: dict) -> dict:
    ordered = {k: row.get(k) for k in PREFERRED_COLUMNS if k in row}
    for k in sorted(row):
        if k not in ordered:
            ordered[k] = row[k]
    return ordered


def write_summary(outdir: Path, row: dict, append_index: Path | None) -> None:
    row = _ordered_row(row)
    csv_path = outdir / "summary_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    md_path = outdir / "summary_metrics.md"
    cols = list(row.keys())
    with md_path.open("w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        f.write("| " + " | ".join("" if row.get(c) is None else str(row.get(c)) for c in cols) + " |\n")

    if append_index is not None:
        append_index.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(row.keys())
        old_rows: list[dict] = []
        if append_index.exists():
            with append_index.open(newline="") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames or []
                old_rows = list(reader)
            fieldnames = list(dict.fromkeys([*old_fields, *fieldnames]))
        with append_index.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old_row in old_rows:
                writer.writerow(old_row)
            writer.writerow(row)

    print(f"-> saved {csv_path}")
    print(f"-> saved {md_path}")
    if append_index is not None:
        print(f"-> appended {append_index}")
        plot_script = Path(__file__).with_name("plot_ssd_summary.py")
        if plot_script.exists():
            result = subprocess.run(
                [sys.executable, str(plot_script), str(append_index)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            print(result.stdout.strip())
        table_script = Path(__file__).with_name("plot_ssd_table.py")
        if table_script.exists():
            result = subprocess.run(
                [sys.executable, str(table_script), str(append_index), "--drop-empty-columns"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--k", type=int, required=True, help="Speculative lookahead for per-step multipliers.")
    parser.add_argument("--warmup", type=int, default=5, help="Drop first N events per label for summary stats.")
    parser.add_argument("--timeline-warmup", type=int, default=10, help="Skip first N response handshakes for timeline selection.")
    parser.add_argument(
        "--hit-threshold-ms",
        type=float,
        default=2.0,
        help="Fallback hit/miss threshold for old profiles without explicit labels.",
    )
    parser.add_argument(
        "--timeline-window",
        choices=("handshake_tail", "post_response", "full_handshake"),
        default="handshake_tail",
        help=(
            "Plot the selected cache handshake at the tail, the post-response "
            "critical path, or the full wait/response handshake at the front."
        ),
    )
    parser.add_argument("--append-index", type=Path, default=None)
    args = parser.parse_args()

    outdir = args.outdir
    target_raw = _load_profile_raw(outdir, "target_rank0")
    draft_raw = _load_profile_raw(outdir, "draft")
    # Strip the _anchor sentinel for the legacy DataFrame path (it has no
    # start_ms / end_ms / ms columns and would pollute summary stats).
    target = _as_df([r for r in target_raw if r.get("label") != "_anchor"], "target")
    draft = _as_df([r for r in draft_raw if r.get("label") != "_anchor"], "draft")

    row = parse_run_log(outdir)
    add_profile_metrics(row, target, draft, args.k, args.warmup)

    # Schema-detect: Phase-B aligned JSON uses the new plotter; legacy JSON
    # falls back to the existing approximate handshake-offset plotter.
    use_aligned = aligned.is_aligned_schema(target_raw)
    if use_aligned:
        try:
            results = aligned.plot_all_statuses(outdir)
            for status, path in results.items():
                row[f"timeline_{status}_pair_idx"] = None  # aligned uses step_id, not pair idx
                if path is None:
                    print(f"[skip] no cache {status} step found; skipping timeline_cache_{status}.png")
                else:
                    print(f"-> saved {path}")
        except aligned.AlignedSchemaError as e:
            print(f"[warn] aligned plot failed ({e}); falling back to legacy approximate plotter.")
            use_aligned = False

    if not use_aligned:
        if not aligned.is_aligned_schema(target_raw):
            print(
                "[WARN] legacy DUET profile detected (no _anchor / wall_start_ns); "
                "timeline plots use approximate handshake-offset alignment and should not be "
                "used as the paper source of truth — re-run with the Phase-B aligned trace "
                "to get step-id-joined timelines."
            )
        for status in ("hit", "miss", "hit_k1", "hit_k2"):
            idx = _choose_pair_index(
                target,
                draft,
                status,
                args.timeline_warmup,
                args.hit_threshold_ms,
            )
            row[f"timeline_{status}_pair_idx"] = idx
            if idx is None:
                print(f"[warn] no cache {status} step found; skipping timeline_cache_{status}.png")
                continue
            try:
                path = plot_timeline_for_pair(outdir, target, draft, idx, status, args.timeline_warmup, args.timeline_window)
                print(f"-> saved {path}")
            except ValueError as e:
                print(f"[warn] timeline_cache_{status}.png skipped: {e}")

    write_summary(outdir, row, args.append_index)
    print(pd.DataFrame([_ordered_row(row)]).to_string(index=False))


if __name__ == "__main__":
    main()
