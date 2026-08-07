"""Per-step Gantt of one DUET spec step: target & draft on a common time axis.

Cross-process alignment: we use NCCL send/recv pairs as anchor.
target.proxy_compute_send.end ≈ draft.proxy_wait.end (identical NCCL completion).
Offset = mean(target.send.end_ms - draft.wait.end_ms) applied to draft events.

Usage: python bench/plot_duet_timeline.py [OUTDIR] [--step N]
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
    paths = sorted(glob.glob(f"{OUTDIR}/duet_profile_{tag}_*.json")) or \
            sorted(glob.glob(f"{OUTDIR}/duet_profile_{tag}.json")) or \
            sorted(glob.glob(f"{OUTDIR}/mesa_profile_{tag}_*.json")) or \
            sorted(glob.glob(f"{OUTDIR}/mesa_profile_{tag}.json"))
    if not paths:
        raise FileNotFoundError(f"no duet_profile_{tag}_*.json in {OUTDIR}")
    return json.load(open(paths[-1]))
draft = _load_latest("draft")
target = _load_latest("target_rank0")

# Unified anchor: target_spec_wait END ≈ draft_send_response END (same NCCL handshake)
# This is a *step boundary*: right after this, target starts the new step's verify.
# Works for both baseline AND DUET identically. Avoids DUET's mid-step proxy anchor confusion.
t_anchor = [e for e in target if e["label"] == "target_spec_wait"]
d_anchor = [e for e in draft if e["label"] == "draft_send_response"]
anchor_kind = "spec_wait/send_response"
if not t_anchor or not d_anchor:
    # Legacy fallback (older profiles w/o step-boundary labels)
    t_anchor = [e for e in target if e["label"] == "proxy_send_enqueue"]
    if not t_anchor:
        t_anchor = [e for e in target
                    if e["label"] == "proxy_compute_send"]
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

# (color, hatch). Hatch disambiguates similar shades when adjacency is unavoidable.
# Families:
#   target verify   : reds        (graph_pre/post, verify_replay)
#   target idle/io  : yellows/golds
#   target logits   : grays (verify_setup / exit_logits / final_logits — all small)
#   draft phase1    : blues       (build / prep / replay = light → medium → dark)
#   draft phase2 split: greens    (build / prep / replay)
#   draft phase2 hybrid: oranges/browns (distinct family from split green so it's
#                                        immediately visible when hybrid is active)
#   draft glue/cache: cyan / teal
#   draft io/idle   : near-black / dark gray
COLORS = {
    # ---- TARGET lane ----
    "graph_pre":            ("#cb181d", ""),     # red
    "graph_post":           ("#67000d", ""),     # dark red
    "verify_replay":        ("#7a0000", ""),     # very dark red (baseline single-graph)
    "verify_sample_accept": ("#fc8d59", ""),     # coral
    "target_postprocess":   ("#feb24c", ".."),   # light orange + dot hatch
    "target_spec_wait":     ("#ffeda0", ""),     # pale yellow (target idle on draft)
    "proxy_compute_send":   ("#fec44f", "xx"),   # gold + cross hatch (handshake marker)
    "exit_proxy_launch":    ("#fdd0a2", "//"),
    "exit_proxy_side":      ("#fdae61", ""),
    "chain_proxy_graph_replay": ("#2ca25f", ""),
    "tree_proxy_graph_replay":  ("#238b45", ".."),
    "proxy_send_enqueue":   ("#006d2c", "//"),
    "verify_setup":         ("#525252", "//"),   # dark gray + slash (tiny setup)
    "exit_logits":          ("#969696", ""),     # mid gray
    "final_logits":         ("#cccccc", "//"),   # light gray + slash
    # ---- DRAFT lane: phase 1 (blues) ----
    "phase1_build":         ("#9ecae1", "//"),   # light blue + slash
    "phase1_prep":          ("#4292c6", "xx"),   # medium blue + cross
    "phase1_replay":        ("#08306b", ""),     # navy
    # ---- DRAFT lane: phase 2 split (greens) ----
    "phase2_build":         ("#a1d99b", "//"),
    "phase2_prep":          ("#41ab5d", "xx"),
    "phase2_replay":        ("#00441b", ""),
    # ---- DRAFT lane: phase 2 hybrid (oranges/browns — visually distinct from split greens) ----
    "phase2_hybrid_build":         ("#fdae6b", "OO"),   # light orange + circle
    "phase2_hybrid_replay_long":   ("#a63603", ""),     # dark orange/brown
    "phase2_hybrid_replay_short":  ("#fd8d3c", "++"),   # mid orange + plus
    "phase2_hybrid_eager_long":    ("#7f2704", "**"),   # very dark brown + star
    "phase2_hybrid_eager_short":   ("#fdd0a2", "\\\\"), # very light orange + backslash
    # ---- DRAFT lane: glue / cache (cyans / teals) ----
    "glue":                 ("#01665e", ""),     # dark teal
    "draft_glue_replay":    ("#5ab4ac", ""),     # mid teal
    "hit_cache_respond":    ("#80cdc1", "//"),   # light teal + slash
    # ---- DRAFT lane: io / idle (dark grays) ----
    "draft_recv_cmd":       ("#4d4d4d", "..."),  # dark gray + dots (idle wait)
    "draft_send_response":  "#252525",
    "proxy_wait":           ("#bdbdbd", "OO"),   # light gray + circle (cross-proc wait)
    "merge_cache":          ("#000000", ""),
    # ---- Baseline tree decode (no DUET) ----
    "tree_prep":            ("#feb24c", "//"),
    "tree_replay":          ("#e6550d", ""),
}


def _style(label):
    """Return (color, hatch) tuple. Tolerates legacy str-only entries."""
    val = COLORS.get(label, ("#d9d9d9", ""))
    if isinstance(val, str):
        return (val, "")
    return val

y_target, y_draft = 1, 0
bar_h = 0.6

def plot_row(events, y, frame_shift):
    for e in events:
        x = e["start_ms"] + frame_shift - t_origin
        w = max(e["ms"], 0.02)   # min width so tiny bars are visible
        color, hatch = _style(e["label"])
        ax.barh(y, w, left=x, height=bar_h,
                color=color, hatch=hatch,
                edgecolor="black", linewidth=0.4)

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
# Label: detect baseline vs DUET by presence of phase1/2 labels
_is_duet = any(e["label"].startswith("phase1_") or e["label"].startswith("phase2_") for e in target + draft)
_method = "DUET" if _is_duet else "SSD (baseline)"
ax.set_title(f"{_method} — {_cfg_name} — spec step #{args.step} (steady-state)")
ax.grid(axis="x", alpha=0.3, linestyle=":")
ax.set_xlim(0, win_hi_t - t_origin)

# Legend — only labels actually present in the plotted window, placed BELOW plot
present_labels = {e["label"] for e in t_win} | {e["label"] for e in d_win}
handles = []
for lb in sorted(present_labels):
    color, hatch = _style(lb)
    handles.append(mpatches.Patch(facecolor=color, hatch=hatch,
                                  edgecolor="black", linewidth=0.4, label=lb))
n_cols = min(6, max(3, (len(handles) + 1) // 2))
# Legend 배치: x-axis label 과 겹치지 않도록 충분히 아래로 + tight_layout bottom reserve 확대
ax.legend(handles=handles, fontsize=8, loc="upper center",
          bbox_to_anchor=(0.5, -0.30), ncol=n_cols,
          frameon=True, framealpha=0.9, borderaxespad=0.4)

plt.subplots_adjust(bottom=0.32)   # legend 영역 충분히 확보 (axis label 아래)
out = f"{OUTDIR}/duet_timeline_step{args.step}.png"
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
