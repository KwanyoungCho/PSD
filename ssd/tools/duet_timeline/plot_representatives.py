#!/usr/bin/env python3
"""Render several duration-quantile DUET timelines per cache status.

The ordinary plotter intentionally chooses one median step.  This companion
uses the same aligned JSON and drawing code but selects p25/p50/p75 full-step
durations for hit_k1, hit_k2, and miss.  Only steps with a captured draft
response marker are eligible, preventing an event-cap tail from producing an
empty draft row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


SSD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SSD_ROOT / "bench"))

from plot_duet_aligned_timeline import (  # noqa: E402
    _load_json,
    _strip_anchor,
    compute_causality_shift_ns,
    is_aligned_schema,
    list_steps_by_status,
    plot_aligned_step,
    select_step,
)


def _quantile_index(n: int, q: float) -> int:
    return min(n - 1, max(0, round(q * (n - 1))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_dir", type=Path)
    ap.add_argument(
        "--quantiles", default="0.25,0.50,0.75",
        help="comma-separated duration quantiles (default: 0.25,0.50,0.75)",
    )
    ap.add_argument("--causality-window", type=int, default=5)
    args = ap.parse_args()

    quantiles = [float(x) for x in args.quantiles.split(",")]
    if not quantiles or any(q < 0.0 or q > 1.0 for q in quantiles):
        raise SystemExit("quantiles must be numbers in [0,1]")

    target = _load_json(args.profile_dir, "target_rank0")
    draft = _load_json(args.profile_dir, "draft")
    if not is_aligned_schema(target):
        raise SystemExit(f"unaligned target profile: {args.profile_dir}")
    target = _strip_anchor(target)
    draft = _strip_anchor(draft)

    captured = {
        int(r["step_id"])
        for r in draft
        if r.get("step_id") is not None
        and r.get("label") == "draft_send_response"
    }
    table = list_steps_by_status(target)
    manifest = ["status\tquantile\tstep_id\tfull_step_ms\timage"]
    rendered: dict[tuple[str, str], Path] = {}

    for status in ("hit_k1", "hit_k2", "miss"):
        items = [item for item in table.get(status, []) if item[0] in captured]
        if not items:
            manifest.append(f"{status}\tNA\tNA\tNA\tNA")
            continue
        used: set[int] = set()
        for q in quantiles:
            sid, duration = items[_quantile_index(len(items), q)]
            if sid in used:
                continue
            used.add(sid)
            tgt, drf = select_step(target, draft, sid, status_filter=status)
            if not tgt:
                continue
            shift_ns, n_pairs = compute_causality_shift_ns(
                target, draft, sid, window=args.causality_window,
            )
            qname = f"p{round(q * 100):02d}"
            out = args.profile_dir / (
                f"timeline_cache_{status}_{qname}_step{sid}.png"
            )
            plot_aligned_step(
                tgt, drf, sid, out,
                title_suffix=(
                    f"cache {status.replace('_', ' ')} — {qname} "
                    f"full-step duration ({duration:.2f} ms)"
                ),
                draft_shift_ns=shift_ns,
                shift_n_pairs=n_pairs,
            )
            manifest.append(
                f"{status}\t{qname}\t{sid}\t{duration:.6f}\t{out.name}"
            )
            rendered[(status, qname)] = out

    path = args.profile_dir / "representatives.tsv"
    path.write_text("\n".join(manifest) + "\n")
    print(path.read_text(), end="")

    # One compact overview per arm.  Full-resolution panels remain beside it
    # for detailed inspection.
    qnames = [f"p{round(q * 100):02d}" for q in quantiles]
    statuses = ("hit_k1", "hit_k2", "miss")
    cell_w, cell_h = 1000, 400
    sheet = Image.new(
        "RGB", (cell_w * len(qnames), cell_h * len(statuses)), "white"
    )
    for row, status in enumerate(statuses):
        for col, qname in enumerate(qnames):
            image_path = rendered.get((status, qname))
            if image_path is None:
                continue
            with Image.open(image_path) as panel:
                panel = panel.convert("RGB")
                panel.thumbnail((cell_w, cell_h))
                x = col * cell_w + (cell_w - panel.width) // 2
                y = row * cell_h + (cell_h - panel.height) // 2
                sheet.paste(panel, (x, y))
    sheet_path = args.profile_dir / "timeline_representatives_contact_sheet.png"
    sheet.save(sheet_path, quality=95)
    print(f"contact_sheet\t{sheet_path.name}")


if __name__ == "__main__":
    main()
