#!/usr/bin/env python3
"""Render breakdown PNGs for middle-of-run steps (instead of duration-median).

For each status (hit_k1, hit_k2, miss), find the step_id closest to the
midpoint of the run (50% progress) that has that status. Renders an aligned
timeline PNG for each.

This gives a more representative view than picking by median full_step_ms,
which can favor edge cases (cold cache early steps, attention-heavy late steps).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench"))
import plot_duet_aligned_timeline as aligned  # noqa: E402


def main():
    outdir = Path(__file__).parent
    # Accepts both new duet_profile_*.json and legacy mesa_profile_*.json
    t_cands = sorted(outdir.glob("duet_profile_target_rank0_*.json"))               or sorted(outdir.glob("mesa_profile_target_rank0_*.json"))
    d_cands = sorted(outdir.glob("duet_profile_draft_*.json"))               or sorted(outdir.glob("mesa_profile_draft_*.json"))
    t_path = t_cands[-1]
    d_path = d_cands[-1]
    target = aligned._strip_anchor(json.loads(t_path.read_text()))
    draft = aligned._strip_anchor(json.loads(d_path.read_text()))

    # Find all step_ids and their status
    step_status: dict[int, str] = {}
    for r in target:
        if r.get("label") == "_anchor":
            continue
        sid = r.get("step_id")
        lbl = r.get("label") or ""
        if sid is None:
            continue
        if lbl.startswith("target_spec_wait"):
            s = r.get("status") or (lbl[len("target_spec_wait_"):] if lbl != "target_spec_wait" else None)
            if s:
                step_status[sid] = s

    # Midpoint
    all_sids = sorted(step_status.keys())
    midpoint = all_sids[len(all_sids) // 2]
    print(f"total steps: {len(all_sids)}, midpoint step_id={midpoint}")

    # For each status, pick step closest to midpoint
    by_status: dict[str, list[int]] = {}
    for sid, s in step_status.items():
        by_status.setdefault(s, []).append(sid)
    for s in by_status:
        by_status[s].sort()

    for status in ("hit_k1", "hit_k2", "miss"):
        candidates = by_status.get(status, [])
        if not candidates:
            print(f"[{status}] no candidates, skip")
            continue
        # Closest to midpoint
        chosen = min(candidates, key=lambda x: abs(x - midpoint))
        out_png = outdir / f"timeline_middle_{status}_step{chosen}.png"
        tr, dr = aligned.select_step(target, draft, chosen)
        try:
            aligned.plot_aligned_step(tr, dr, chosen, out_png, title_suffix=f"(middle, status={status})")
            # Print this step's target_spec_wait duration as sanity
            sw = next((r for r in tr if (r.get("label") or "").startswith("target_spec_wait")), None)
            sw_ms = float(sw.get("cuda_ms", 0)) if sw else 0
            full = sum(float(r.get("cuda_ms", 0)) for r in tr if (r.get("label") or "") != "target_spec_wait" and not (r.get("label") or "").startswith("target_spec_wait_"))
            print(f"[{status}] step_id={chosen}  spec_wait={sw_ms:.2f}ms  target_active={full:.2f}ms  -> {out_png.name}")
        except Exception as e:
            print(f"[{status}] step_id={chosen} FAILED: {e}")


if __name__ == "__main__":
    main()
