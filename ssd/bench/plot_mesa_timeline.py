"""Per-step Gantt of one MESA spec step: target & draft on a common time axis.

Cross-process alignment: we use NCCL send/recv pairs as anchor.
target.proxy_compute_send.end ≈ draft.proxy_wait.end (identical NCCL completion).
Offset = mean(target.send.end_ms - draft.wait.end_ms) applied to draft events.

Usage: python bench/plot_mesa_timeline.py [OUTDIR] [--step N]
"""
import json, os, sys, argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ap = argparse.ArgumentParser()
ap.add_argument("outdir", nargs="?", default=os.environ.get("SSD_PROFILE_DIR", "/tmp"))
ap.add_argument("--step", type=int, default=50, help="which spec step to plot (0-indexed in steady state)")
ap.add_argument("--warmup", type=int, default=10, help="skip first N spec steps")
args = ap.parse_args()

OUTDIR = args.outdir
import glob
def _load_latest(tag):
    paths = sorted(glob.glob(f"{OUTDIR}/mesa_profile_{tag}_*.json")) or \
            sorted(glob.glob(f"{OUTDIR}/mesa_profile_{tag}.json"))
    if not paths:
        raise FileNotFoundError(f"no mesa_profile_{tag}_*.json in {OUTDIR}")
    return json.load(open(paths[-1]))
draft = _load_latest("draft")
target = _load_latest("target_rank0")

# Unified anchor: target_spec_wait END ≈ draft_send_response END (same NCCL handshake)
# This is a *step boundary*: right after this, target starts the new step's verify.
# Works for both baseline AND MESA identically. Avoids MESA's mid-step proxy anchor confusion.
t_anchor = [e for e in target if e["label"] == "target_spec_wait"]
d_anchor = [e for e in draft if e["label"] == "draft_send_response"]
anchor_kind = "spec_wait/send_response"
if not t_anchor or not d_anchor:
    # Legacy fallback (older profiles w/o step-boundary labels)
    t_anchor = [e for e in target if e["label"] == "proxy_compute_send"]
    d_anchor = [e for e in draft if e["label"] == "proxy_wait"]
    anchor_kind = "proxy (legacy)"
N = min(len(t_anchor), len(d_anchor))
print(f"paired {N} {anchor_kind} handshakes")

# Offset to apply to draft -> target timeframe
offsets = [t_anchor[i]["end_ms"] - d_anchor[i]["end_ms"] for i in range(N)]
# Use robust mean (drop first warmup steps)
offsets_steady = offsets[args.warmup:]
offset = sum(offsets_steady) / len(offsets_steady)
spread = (max(offsets_steady) - min(offsets_steady))
print(f"alignment offset draft->target: mean={offset:.3f} ms, spread={spread:.3f} ms (over {len(offsets_steady)} pairs)")

# Pick a spec step: use proxy_wait #step as the anchor.
step = args.step + args.warmup
assert step < N - 2, f"step {step} >= {N-2} available (need 1.5 steps visible)"
anchor_send = t_anchor[step]["end_ms"]       # target frame
anchor_wait_target = d_anchor[step]["end_ms"] + offset  # draft frame -> target frame

# Window ≈ 1.5 step cycles: handshake[step] → midpoint between handshake[step+1] and [step+2]
win_lo_t = t_anchor[step]["end_ms"]
_s1 = t_anchor[step+1]["end_ms"]
_s2 = t_anchor[step+2]["end_ms"]
win_hi_t = _s1 + 0.5 * (_s2 - _s1)
win_lo_d = win_lo_t - offset
win_hi_d = win_hi_t - offset

def in_window(e, lo, hi):
    return (e["end_ms"] >= lo) and (e["start_ms"] <= hi)

t_win = [e for e in target if in_window(e, win_lo_t, win_hi_t)]
d_win = [e for e in draft if in_window(e, win_lo_d, win_hi_d)]

# Shift everything so the window starts at t=0 (target frame)
t_origin = win_lo_t
fig, ax = plt.subplots(figsize=(16, 5.2))

COLORS = {
    # Target lane — reds for big replays, grays for small logits/setup, gold for handshake
    "verify_setup":         "#d9d9d9",  # light gray — tiny setup
    "graph_pre":            "#e41a1c",  # strong red — biggest target phase
    "exit_logits":          "#bdbdbd",  # mid gray — small lm_head
    "proxy_compute_send":   "#ffbf00",  # gold — handshake indicator
    "graph_post":           "#a50f15",  # dark red
    "final_logits":         "#737373",  # darker gray
    "verify_sample_accept": "#fc9272",  # salmon — postprocessing
    "target_spec_wait":     "#fee090",  # pale yellow — blocked waiting for draft
    "target_postprocess":   "#fdae61",  # light orange — engine bookkeeping
    "verify_replay":        "#b30000",  # dark red — baseline single-graph verify (target)
    "draft_glue_replay":    "#1f78b4",  # blue — draft's own glue decode forward (not verify!)
    # Draft lane — purples for Phase 1, greens for Phase 2 (visually unambiguous per phase)
    "glue":                 "#17becf",  # cyan — boundary / start of step
    "phase1_build":         "#dadaeb",  # very pale purple
    "phase1_prep":          "#807dba",  # medium purple
    "phase1_replay":        "#3f007d",  # deep purple
    "phase2_build":         "#c7e9c0",  # pale green
    "phase2_prep":          "#74c476",  # medium green
    "phase2_replay":        "#00441b",  # deep green
    "proxy_wait":           "#ff7f00",  # orange — cross-proc wait (stands out)
    "merge_cache":          "#252525",  # near-black
    # Baseline draft tree decode (layout=None)
    "tree_prep":            "#fcae91",  # light coral
    "tree_replay":          "#cb181d",  # dark red
    # Draft loop bookkeeping (appears in both MESA and baseline)
    "draft_recv_cmd":       "#bdbdbd",  # light gray — waiting for cmd
    "hit_cache_respond":    "#78c679",  # light green — cache lookup / JIT speculate
    "draft_send_response":  "#006d2c",  # dark green — NCCL send
}

y_target, y_draft = 1, 0
bar_h = 0.6

def plot_row(events, y, frame_shift):
    for e in events:
        x = e["start_ms"] + frame_shift - t_origin
        w = max(e["ms"], 0.02)   # min width so tiny bars are visible
        ax.barh(y, w, left=x, height=bar_h,
                color=COLORS.get(e["label"], "#bdc3c7"),
                edgecolor="black", linewidth=0.3)

plot_row(t_win, y_target, 0.0)
plot_row(d_win, y_draft, offset)

# Step-boundary markers at all handshakes within the window
for s_i in (step, step+1, step+2):
    if s_i < len(t_anchor):
        xt = t_anchor[s_i]["end_ms"] - t_origin
        if xt <= win_hi_t - win_lo_t + 1:
            ax.axvline(xt, color="black", lw=0.8, alpha=0.4, linestyle=":")
ax.text(0, 1.7, f"step #{args.step}", fontsize=8, ha="left", va="center", color="black", alpha=0.8)
mid_x = t_anchor[step+1]["end_ms"] - t_origin
ax.text(mid_x + 0.5, 1.7, f"step #{args.step+1}", fontsize=8, ha="left", va="center", color="black", alpha=0.8)

ax.set_yticks([y_draft, y_target])
ax.set_yticklabels(["draft", "target"])
ax.set_xlabel("time (ms, target frame — draft events offset-aligned)")
_cfg_name = os.path.basename(os.path.normpath(OUTDIR))
# Label: detect baseline vs MESA by presence of phase1/2 labels
_is_mesa = any(e["label"].startswith("phase1_") or e["label"].startswith("phase2_") for e in target + draft)
_method = "MESA" if _is_mesa else "SSD (baseline)"
ax.set_title(f"{_method} — {_cfg_name} — spec step #{args.step} (steady-state)")
ax.grid(axis="x", alpha=0.3, linestyle=":")
ax.set_xlim(0, win_hi_t - t_origin)

# Legend — only labels actually present in the plotted window, placed BELOW plot
present_labels = {e["label"] for e in t_win} | {e["label"] for e in d_win}
handles = [mpatches.Patch(color=COLORS.get(lb, "#bdc3c7"), label=lb)
           for lb in sorted(present_labels)]
n_cols = min(6, max(3, (len(handles) + 1) // 2))
# Legend 배치: x-axis label 과 겹치지 않도록 충분히 아래로 + tight_layout bottom reserve 확대
ax.legend(handles=handles, fontsize=8, loc="upper center",
          bbox_to_anchor=(0.5, -0.30), ncol=n_cols,
          frameon=True, framealpha=0.9, borderaxespad=0.4)

plt.subplots_adjust(bottom=0.32)   # legend 영역 충분히 확보 (axis label 아래)
out = f"{OUTDIR}/mesa_timeline_step{args.step}.png"
plt.savefig(out, dpi=150)
print(f"-> saved {out}")

# Also print the raw events for this step
print("\n=== target events in window ===")
for e in sorted(t_win, key=lambda x: x["start_ms"]):
    print(f"  {e['label']:<22} start={e['start_ms']-t_origin:7.3f} end={e['end_ms']-t_origin:7.3f} ms={e['ms']:6.3f}")
print("\n=== draft events in window (offset-applied) ===")
for e in sorted(d_win, key=lambda x: x["start_ms"]):
    s = e["start_ms"] + offset - t_origin; en = e["end_ms"] + offset - t_origin
    print(f"  {e['label']:<22} start={s:7.3f} end={en:7.3f} ms={e['ms']:6.3f}")
